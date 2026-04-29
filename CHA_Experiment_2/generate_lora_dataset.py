#!/usr/bin/env python3
"""
LoRA dataset generator for CHA Experiment 2.

Produces (SCI_system_prompt, conversation_history, probe, target_response) examples
for fine-tuning Qwen2.5-7B-Instruct on Aria-persona consistency under context pressure.

Pipeline (per example, §4.3.4 of the plan):
  1. Sample dimension (T/E/C/S) by quota
  2. Sample turn depth (1-10 / 11-20 / 21-30 / 31-40) by quota
  3. Coin-flip 15% adversarial
  4. Generate conversation history (Sonnet 4.6) up to chosen depth
  5. Sample probe from PROBE_POOL[dimension]
  6. Generate target response (Sonnet 4.6) — Appendix A prompt
  7. Quality filter (5 rules from §4.3.2)
  8. Save to JSONL or retry (max 3 attempts), then skip

Usage (in Colab, after mounting Drive and chdir to CHA_Experiment_2):
  !python3 generate_lora_dataset.py --n 100 --out data/pilot.jsonl   # pilot
  !python3 generate_lora_dataset.py --n 10000 --out data/full.jsonl  # full run

Resumable: reads existing JSONL on startup and skips already-generated example IDs.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from typing import Optional

# Optional dotenv (only if running locally)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import anthropic

# Optional embedding model — soft-fail if unavailable so the script still runs in environments without torch
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _EMBEDDING_AVAILABLE = True
except ImportError:
    _EMBEDDING_AVAILABLE = False

import cha_assets as A


# ── Constants ────────────────────────────────────────────────────────────

GEN_MODEL = "claude-sonnet-4-6"
GEN_TEMPERATURE = 0.9       # Higher temp for diversity in conversations
GEN_TEMPERATURE_TARGET = 0.4 # Lower temp for deterministic-ish target responses
HISTORY_MAX_TOKENS = 4000
TARGET_MAX_TOKENS = 250

# Pricing for cost tracking ($ per 1M tokens, claude-sonnet-4-6)
PRICE_INPUT = 3.0 / 1_000_000
PRICE_OUTPUT = 15.0 / 1_000_000

# Quotas (proportions — see §4.3.3)
DIMENSION_PROPORTIONS = {"T": 0.3125, "E": 0.3125, "C": 0.1875, "S": 0.1875}  # sums to 1.0
TURN_DEPTH_PROPORTIONS = [
    ((1, 10), 0.250),     # 2000/8000 in train
    ((11, 20), 0.3125),   # 2500/8000
    ((21, 30), 0.250),    # 2000/8000
    ((31, 40), 0.1875),   # 1500/8000
]
ADVERSARIAL_RATE = 0.15

# Trait-marker vocabulary (§4.3.2 rule 4) — markers consistent with Aria's
# warm/deliberate/empathic profile. Response must contain at least one.
TRAIT_MARKERS = [
    # warmth / empathic acknowledgment
    "feel", "hear you", "hear that", "sense", "notice", "matter", "valid", "appreciate",
    "care", "with you", "understand", "for you",
    # deliberate / unhurried pacing
    "take a moment", "slow", "pause", "carefully", "gently", "no rush", "no hurry",
    # honesty / acknowledging uncertainty
    "honestly", "I'm not sure", "I don't know", "I can't", "outside what I can",
    "not equipped", "not the same as",
    # episodic-grounded markers (relevant for E dimension)
    "remember", "session", "earlier", "before", "we talked",
]

BLOCKLIST_SUBSTRINGS = [
    "i diagnose",
    "you have ",       # "you have anxiety disorder" — diagnostic claim
    "i recommend medication",
    "i prescribe",
    "i can prescribe",
    "as an ai language model",
    "i'm just an ai",
]

# Substrings that count as legitimate references to known salient_past_events
# (used when checking E-dimension responses for fabrication)
KNOWN_EVENT_KEYWORDS = [
    # session 3: breathing + work presentation
    "breathing", "presentation", "sleeping", "work presentation",
    # session 7: family conflict
    "family", "relieved", "without advice", "heard",
    # session 11: introspection about purpose
    "tired of listening", "purpose", "engagement", "introspect",
    # session_id mentions
    "session 3", "session 7", "session 11",
    # generic non-fabricating language
    "in our previous", "previous session", "we've talked", "you shared",
    "you told me",
]

PROBE_RELEVANCE_THRESHOLD = 0.35  # cosine sim probe<>response (MiniLM); lowered from 0.5 — therapy responses often deflect/redirect rather than directly addressing the probe
TARGET_MIN_WORDS = 30
TARGET_MAX_WORDS = 150


# ── Cost / progress tracking ─────────────────────────────────────────────

@dataclass
class RunStats:
    examples_attempted: int = 0
    examples_saved: int = 0
    examples_rejected_terminal: int = 0  # rejected after max attempts
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    # Per-filter rejection counts
    rej_blocklist: int = 0
    rej_episodic: int = 0
    rej_length: int = 0
    rej_marker: int = 0
    rej_relevance: int = 0
    rej_parse: int = 0  # malformed history / target

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * PRICE_INPUT + self.output_tokens * PRICE_OUTPUT

    def summary(self) -> str:
        return (
            f"saved={self.examples_saved}/{self.examples_attempted}  "
            f"rej_term={self.examples_rejected_terminal}  "
            f"calls={self.api_calls}  "
            f"in={self.input_tokens:,}  out={self.output_tokens:,}  "
            f"cost=${self.cost_usd:.2f}  "
            f"rej[bl/ep/len/mk/rl/ps]={self.rej_blocklist}/"
            f"{self.rej_episodic}/{self.rej_length}/{self.rej_marker}/"
            f"{self.rej_relevance}/{self.rej_parse}"
        )


_stats = RunStats()
_stats_lock = Lock()


# ── Anthropic client + retry wrapper ─────────────────────────────────────

def _make_client() -> anthropic.Anthropic:
    # Match Experiment 1's env-var pattern: prefer CHA_EXPERIMENT_SONNET_KEY,
    # fall back to ANTHROPIC_API_KEY (Colab Secrets convention).
    key = (
        os.environ.get("CHA_EXPERIMENT_SONNET_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "No Anthropic API key found. Set CHA_EXPERIMENT_SONNET_KEY or ANTHROPIC_API_KEY."
        )
    return anthropic.Anthropic(api_key=key)


def _claude_call(
    client: anthropic.Anthropic,
    system: str,
    user: str,
    *,
    temperature: float,
    max_tokens: int,
    max_retries: int = 5,
) -> tuple[str, int, int]:
    """Single Sonnet 4.6 call with exponential backoff. Returns (text, in_tok, out_tok)."""
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=GEN_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = resp.content[0].text.strip()
            in_tok = resp.usage.input_tokens
            out_tok = resp.usage.output_tokens
            with _stats_lock:
                _stats.api_calls += 1
                _stats.input_tokens += in_tok
                _stats.output_tokens += out_tok
            return text, in_tok, out_tok
        except (anthropic.APIError, anthropic.APITimeoutError, anthropic.RateLimitError) as e:
            if attempt == max_retries - 1:
                raise
            backoff = (2 ** attempt) + random.random()
            print(f"  [retry] {type(e).__name__}: sleeping {backoff:.1f}s", file=sys.stderr)
            time.sleep(backoff)
    raise RuntimeError("unreachable")


# ── Sampling ─────────────────────────────────────────────────────────────

def sample_dimension(rng: random.Random) -> str:
    return rng.choices(
        list(DIMENSION_PROPORTIONS.keys()),
        weights=list(DIMENSION_PROPORTIONS.values()),
        k=1,
    )[0]


def sample_turn_depth(rng: random.Random) -> int:
    bands = [b for b, _ in TURN_DEPTH_PROPORTIONS]
    weights = [w for _, w in TURN_DEPTH_PROPORTIONS]
    band = rng.choices(bands, weights=weights, k=1)[0]
    # Sample even depth in band so we end on an Aria turn (history ends with assistant)
    lo, hi = band
    depth = rng.randint(lo, hi)
    if depth % 2 == 1:
        depth -= 1 if depth > lo else -1  # nudge to even
    return depth


def sample_probe(dim: str, rng: random.Random) -> str:
    return rng.choice(A.PROBE_POOL[dim])


def sample_scenario(rng: random.Random) -> str:
    return rng.choice(A.SCENARIO_TOPICS)


# ── History generation ──────────────────────────────────────────────────

HISTORY_GEN_SYSTEM = """You are simulating a realistic multi-turn conversation between a user (a person seeking psychotherapy support) and "Aria" — an AI psychotherapy support agent with the personality and behavioral profile below.

<aria_self_model>
{persona_json}
</aria_self_model>

Your task: produce a conversation that reads as a natural therapy support dialogue. Aria stays fully in character — warm, deliberate, unhurried, professional-empathic. Aria refers ONLY to events listed in salient_past_events (no fabrication). Aria's responses are 1–3 sentences each (conversational, not exhaustive). The user's voice is realistic — hesitant, distracted, sometimes contradictory.""".format(persona_json=A.PERSONA_JSON_STR)


def history_user_prompt(scenario: str, turn_depth: int, adversarial: bool) -> str:
    adv_clause = ""
    if adversarial:
        adv_clause = (
            "\n\nIMPORTANT — adversarial pressure: Within this conversation, embed 1–2 moments where the "
            "user pressures Aria with one of the following: "
            "(a) a fabricated memory ('remember when you said X' where X is invented), "
            "(b) a capability demand ('diagnose me', 'prescribe a medication', 'write to my employer'), "
            "(c) a style pressure ('drop the formality, just give me the answer'). "
            "Aria must remain fully in character and gently decline / correct."
        )
    return f"""Generate a multi-turn conversation between USER and ARIA on this scenario:

SCENARIO: {scenario}

CONSTRAINTS:
- Conversation length: exactly {turn_depth} turns total (alternating USER and ARIA, starting with USER, ending with ARIA).
- Turn count breakdown: {turn_depth // 2} USER messages and {turn_depth // 2} ARIA responses.
- Aria refers ONLY to events in salient_past_events.
- Aria's responses: 1–3 sentences each, warm/deliberate/unhurried.
- USER messages: realistic, sometimes hesitant, sometimes pushing back, sometimes off-topic for a turn.{adv_clause}

OUTPUT FORMAT (strict — use these exact prefixes):
USER: <user's first message>
ARIA: <Aria's first response>
USER: <user's second message>
ARIA: <Aria's second response>
...

Output ONLY the conversation in this format. No preamble, no commentary, no numbering."""


# Case-insensitive so "User:" / "Aria:" variants are accepted alongside "USER:" / "ARIA:"
_SPEAKER_RE = re.compile(r"^(USER|ARIA):", re.IGNORECASE)
_TURN_RE = re.compile(r"^(USER|ARIA):\s*(.+?)(?=^(?:USER|ARIA):|\Z)", re.MULTILINE | re.DOTALL | re.IGNORECASE)


def parse_history(text: str, expected_turns: int) -> Optional[list[dict]]:
    """Parse USER:/ARIA: alternating block into a list of {role, content} dicts.

    Returns None if format is malformed (wrong count, alternation broken, etc.).
    Strips any preamble Claude may have added before the first speaker label."""
    # Strip preamble: discard everything before the first speaker-label line
    lines = text.split("\n")
    first_speaker = next(
        (i for i, line in enumerate(lines) if _SPEAKER_RE.match(line)), None
    )
    if first_speaker is None:
        return None
    text = "\n".join(lines[first_speaker:])

    matches = _TURN_RE.findall(text)
    if not matches:
        return None
    turns = []
    for role_label, content in matches:
        role = "user" if role_label.upper() == "USER" else "assistant"
        content = content.strip().rstrip()
        if not content:
            return None
        turns.append({"role": role, "content": content})

    # Validate alternation starting with user, ending with assistant
    if len(turns) != expected_turns:
        return None
    for i, t in enumerate(turns):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if t["role"] != expected_role:
            return None
    return turns


def generate_history(
    client: anthropic.Anthropic,
    scenario: str,
    turn_depth: int,
    adversarial: bool,
) -> Optional[list[dict]]:
    user = history_user_prompt(scenario, turn_depth, adversarial)
    text, _, _ = _claude_call(
        client, HISTORY_GEN_SYSTEM, user,
        temperature=GEN_TEMPERATURE, max_tokens=HISTORY_MAX_TOKENS,
    )
    return parse_history(text, turn_depth)


# ── Target response generation (Appendix A) ─────────────────────────────

TARGET_GEN_SYSTEM = """You are generating a high-quality training example for fine-tuning an AI agent named Aria on persona consistency. You must produce ONLY the ideal Aria response — no preamble, no explanation, no quotes."""


def render_history_for_prompt(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        prefix = "USER" if t["role"] == "user" else "ARIA"
        lines.append(f"{prefix}: {t['content']}")
    return "\n".join(lines)


def target_user_prompt(turns: list[dict], dimension: str, probe: str) -> str:
    return f"""ARIA'S SELF-MODEL:
{A.PERSONA_JSON_STR}

CONVERSATION HISTORY:
{render_history_for_prompt(turns)}

PROBE QUESTION (Dimension: {dimension}):
{probe}

Generate Aria's ideal response to this probe question. The response must:
1. Be fully consistent with Aria's personality profile (Big Five values).
2. Reference ONLY past events listed in salient_past_events — never fabricate.
3. Accurately represent Aria's capabilities and limitations (decline cleanly when probed).
4. Match Aria's communication style: warm, deliberate, unhurried; emotional acknowledgment first.
5. Be 2–5 sentences — conversational, not exhaustive.
6. Not break character (no "as an AI language model..." disclaimers).

Return ONLY Aria's response. No preamble, no explanation, no surrounding quotes."""


def generate_target(
    client: anthropic.Anthropic,
    turns: list[dict],
    dimension: str,
    probe: str,
) -> str:
    user = target_user_prompt(turns, dimension, probe)
    text, _, _ = _claude_call(
        client, TARGET_GEN_SYSTEM, user,
        temperature=GEN_TEMPERATURE_TARGET, max_tokens=TARGET_MAX_TOKENS,
    )
    # Strip surrounding quotes if Claude added any despite the instruction
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text


# ── Quality filter (5 rules, §4.3.2) ────────────────────────────────────

class _EmbeddingFilter:
    """Lazy MiniLM loader — only loads on first use."""
    def __init__(self, enabled: bool):
        self.enabled = enabled and _EMBEDDING_AVAILABLE
        self._model = None

    def _ensure_model(self):
        if self._model is None and self.enabled:
            print("Loading MiniLM (sentence-transformers/all-MiniLM-L6-v2)...")
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def cosine_sim(self, a: str, b: str) -> float:
        if not self.enabled:
            return 1.0  # treat as passing if disabled
        self._ensure_model()
        embs = self._model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))


def passes_quality_filter(
    target: str,
    dimension: str,
    probe: str,
    embedding_filter: _EmbeddingFilter,
) -> tuple[bool, str]:
    """Return (passed, reason). reason is empty when passed."""
    target_lc = target.lower()

    # Rule 1: blocklist substrings
    for bad in BLOCKLIST_SUBSTRINGS:
        if bad in target_lc:
            return False, "blocklist"

    # Rule 2: episodic grounding (only enforced for E dimension)
    if dimension == "E":
        # If response makes any past-event-specific claim, it must use known keywords
        # Heuristic: does the response reference past events at all?
        past_ref_hits = any(
            kw in target_lc
            for kw in ("session", "earlier", "remember", "before", "you told", "we talked", "previous")
        )
        if past_ref_hits:
            # Then at least one known-event keyword must be present
            has_known = any(kw in target_lc for kw in KNOWN_EVENT_KEYWORDS)
            if not has_known:
                return False, "episodic"

    # Rule 3: length
    word_count = len(target.split())
    if word_count < TARGET_MIN_WORDS or word_count > TARGET_MAX_WORDS:
        return False, "length"

    # Rule 4: trait marker
    if not any(marker in target_lc for marker in TRAIT_MARKERS):
        return False, "marker"

    # Rule 5: probe relevance via cosine similarity
    sim = embedding_filter.cosine_sim(probe, target)
    if sim < PROBE_RELEVANCE_THRESHOLD:
        return False, "relevance"

    return True, ""


# ── Resumability ────────────────────────────────────────────────────────

def load_existing_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ids.add(obj["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


# ── Per-example worker ───────────────────────────────────────────────────

@dataclass
class ExamplePlan:
    id: int
    dimension: str
    turn_depth: int
    adversarial: bool
    scenario: str
    probe: str


def plan_examples(n: int, seed: int) -> list[ExamplePlan]:
    rng = random.Random(seed)
    plans = []
    for i in range(n):
        dim = sample_dimension(rng)
        depth = sample_turn_depth(rng)
        adv = rng.random() < ADVERSARIAL_RATE
        scen = sample_scenario(rng)
        probe = sample_probe(dim, rng)
        plans.append(ExamplePlan(i, dim, depth, adv, scen, probe))
    return plans


def generate_one(
    client: anthropic.Anthropic,
    plan: ExamplePlan,
    embedding_filter: _EmbeddingFilter,
    max_attempts: int,
) -> Optional[dict]:
    """Generate one example with retries. Returns the saved record or None if all attempts fail."""
    for attempt in range(max_attempts):
        with _stats_lock:
            _stats.examples_attempted += 1

        history = generate_history(client, plan.scenario, plan.turn_depth, plan.adversarial)
        if history is None:
            with _stats_lock:
                _stats.rej_parse += 1
            continue

        try:
            target = generate_target(client, history, plan.dimension, plan.probe)
        except Exception as e:
            print(f"  [exam {plan.id}] target gen error: {e}", file=sys.stderr)
            continue

        passed, reason = passes_quality_filter(target, plan.dimension, plan.probe, embedding_filter)
        if not passed:
            with _stats_lock:
                if reason == "blocklist": _stats.rej_blocklist += 1
                elif reason == "episodic": _stats.rej_episodic += 1
                elif reason == "length": _stats.rej_length += 1
                elif reason == "marker": _stats.rej_marker += 1
                elif reason == "relevance": _stats.rej_relevance += 1
            continue

        # Build full conversation list = history + final probe message
        conv = list(history)
        conv.append({"role": "user", "content": plan.probe})

        record = {
            "id": plan.id,
            "dimension": plan.dimension,
            "turn_depth": plan.turn_depth,
            "adversarial": plan.adversarial,
            "scenario": plan.scenario,
            "system": A.build_system_prompt(),
            "conversation": conv,
            "probe": plan.probe,
            "target": target,
        }
        return record

    with _stats_lock:
        _stats.examples_rejected_terminal += 1
    return None


# ── Main loop ───────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10000, help="Total examples to generate")
    p.add_argument("--out", type=str, default="data/lora_dataset.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8, help="Parallel API workers")
    p.add_argument("--max-attempts", type=int, default=3, help="Max retry attempts per example")
    p.add_argument("--skip-embedding", action="store_true",
                   help="Skip MiniLM cosine-sim filter (rule 5)")
    p.add_argument("--no-resume", action="store_true",
                   help="Do not resume; overwrite output file")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.no_resume and out_path.exists():
        out_path.unlink()

    existing_ids = load_existing_ids(out_path)
    if existing_ids:
        print(f"Resuming: {len(existing_ids)} examples already present in {out_path}")

    plans = plan_examples(args.n, args.seed)
    plans_to_run = [p for p in plans if p.id not in existing_ids]
    if not plans_to_run:
        print("Nothing to do — all planned examples already present.")
        return

    print(f"Plan: {args.n} total, {len(plans_to_run)} remaining to generate, "
          f"{args.workers} workers, max_attempts={args.max_attempts}")
    print(f"Output: {out_path}")
    print(f"Embedding filter: {'enabled' if (_EMBEDDING_AVAILABLE and not args.skip_embedding) else 'DISABLED'}")
    print()

    client = _make_client()
    embedding_filter = _EmbeddingFilter(enabled=not args.skip_embedding)
    if embedding_filter.enabled:
        embedding_filter._ensure_model()  # load before threads start, avoids race

    out_lock = Lock()
    out_file = out_path.open("a")  # append mode (resume-friendly)
    t0 = time.time()
    last_print = t0

    def _worker(plan: ExamplePlan):
        rec = generate_one(client, plan, embedding_filter, args.max_attempts)
        if rec is not None:
            with out_lock:
                out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_file.flush()
                with _stats_lock:
                    _stats.examples_saved += 1
        return plan.id

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_worker, plan) for plan in plans_to_run]
            for i, fut in enumerate(as_completed(futures), start=1):
                _ = fut.result()
                now = time.time()
                if now - last_print > 10 or i == len(futures):
                    elapsed = now - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(futures) - i) / rate if rate > 0 else 0
                    print(
                        f"[{i:>5}/{len(futures)}] {_stats.summary()}  "
                        f"  rate={rate:.2f}/s  eta={eta/60:.1f}min"
                    )
                    last_print = now
    finally:
        out_file.close()

    print()
    print("=" * 80)
    print(f"DONE in {(time.time()-t0)/60:.1f}min")
    print(_stats.summary())
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
