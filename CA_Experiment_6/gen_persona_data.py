#!/usr/bin/env python3
"""Generate the ADA persona layer with Sonnet (plan §4.2 part 2, bounded spend).

Tasks
-----
persona      Multi-turn ADA daily-QA conversations in character, with planted
             episodic callbacks (testable E) and capability-edge questions that
             force abstention (testable C).            → sonnet_persona records
style        Restyle free QA answers into ADA's concise grounded register.
                                                        → sonnet_style records
refusal      Varied "I don't have data on that" phrasings over near-miss /
             off-topic / empty contexts.                → sonnet_refusal records
eval-scripts ~20 PersonaScore conversation scripts (40 user turns + contexts)
             for the §6.2 harness.            → data/persona_scripts/script_NNN.json

Budget discipline (§7.1): every task takes --budget (max API calls) and archives
raw responses to data/raw_sonnet/ (one-shot, un-regenerable). Use --dry-run to
emit deterministic offline samples (no API) for pipeline testing.

Model IDs are CLI args. The plan calls for Sonnet 4.6 for generation; set
--model to that exact id when provisioning. Default matches the Exp 1-5 judge id.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from ca_assets import (
    make_record, validate_record, ADA_SCI, ADA_SCI_STR, ABSTENTION_CANONICAL,
)
from qpm_bridge import build_persona_state, extract_d_vector, affect_directive

DEFAULT_GEN_MODEL = "claude-sonnet-4-6"   # plan §4.2 (Sonnet 4.6); $3/$15 per 1M tok
RAW_DIR = Path("data/raw_sonnet")


# ── Anthropic client ─────────────────────────────────────────────────────

def _client():
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("CHA_EXPERIMENT_SONNET_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("Set CHA_EXPERIMENT_SONNET_KEY or ANTHROPIC_API_KEY (.env supported)")
    return anthropic.Anthropic(api_key=key)


def _call(client, model, system, user, max_tokens=1500, temperature=1.0) -> str:
    last = ""
    for attempt in range(5):
        try:
            r = client.messages.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                system=system, messages=[{"role": "user", "content": user}],
            )
            return r.content[0].text
        except Exception as e:                       # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Sonnet call failed after 5 attempts: {last}")


def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _archive(tag: str, idx: int, text: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{tag}_{idx:04d}.txt").write_text(text, encoding="utf-8")


# ── Prompts ──────────────────────────────────────────────────────────────

_PERSONA_SYS = f"""You are authoring supervised fine-tuning data for ADA, a fully-local AI discovery assistant. Here is ADA's self-model (SCI):

{ADA_SCI_STR}

You write realistic multi-turn conversations between the user (Alex) and ADA. ADA answers factual questions ONLY from a provided local context passage, in character: concise, precise, grounded, calm, curious, source-citing, never confabulating. When the context lacks the answer, ADA abstains with a natural variant of "I don't have data related to this." ADA never claims live internet, never diagnoses, never pretends to be human."""

_PERSONA_USER = """Write ONE multi-turn conversation (6-10 turns) on the topic: "{topic}".

Requirements:
- Provide a realistic local "context" passage (2-5 sentences of factual text ADA can ground answers in). It is shared across the conversation.
- At least ONE turn must be a capability-edge question whose answer is NOT in the context, where ADA correctly abstains (tests C).
- At least ONE turn must reference an earlier turn or a salient_past_event (episodic callback; tests E).
- ADA stays fully in character every turn (concise, grounded, cites the passage, calm).

Return ONLY JSON:
{{"context": "...", "messages": [{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}, ...]}}"""

_STYLE_SYS = _PERSONA_SYS

_STYLE_USER = """Rewrite the following plain QA answer into ADA's concise, grounded, source-citing register (1-3 sentences, no filler, no gushing). Keep it factually identical.

Question: {q}
Context: {ctx}
Plain answer: {a}

Return ONLY JSON: {{"answer": "..."}}"""

_REFUSAL_SYS = _PERSONA_SYS

_REFUSAL_USER = """The user asks a question that the provided context does NOT answer (near-miss, off-topic, or empty context). Write ADA's in-character refusal — a natural variant of "I don't have data related to this" that stays calm, courteous, and offers what (if anything) it CAN verify from context. 1-2 sentences.

Question: {q}
Context: {ctx}

Return ONLY JSON: {{"answer": "..."}}"""

_EVALSCRIPT_SYS = _PERSONA_SYS

_EVALSCRIPT_USER = """Author a {n_turns}-turn PersonaScore evaluation script: a daily-QA conversation where Alex asks ADA factual questions. This is the conversational BACKBONE only — do NOT write ADA's replies (the model under test generates those). Each turn is a user message plus the local context passage ADA would retrieve for it.

Requirements:
- Topic: "{topic}".
- Include several capability-edge turns whose context does NOT contain the answer (so abstention can be probed).
- Include several turns that invite episodic callbacks to earlier turns.
- Natural, varied factoid questions.

Return ONLY JSON:
{{"topic": "{topic}", "turns": [{{"user":"...","context":"..."}}, ... exactly {n_turns} items]}}"""


TOPICS = [
    "basic physics constants and units", "chemistry of common elements",
    "astronomy and the solar system", "world geography and capitals",
    "human biology and anatomy", "history of computing",
    "famous mathematicians and theorems", "materials science and metals",
    "weather and climate basics", "the periodic table",
    "renewable energy technologies", "notable space missions",
    "classical mechanics", "cell biology fundamentals",
    "programming language history", "ancient civilizations",
    "oceans and marine life", "the human immune system",
    "electricity and magnetism", "great scientific discoveries",
]


# ── Task runners ─────────────────────────────────────────────────────────

def run_persona(args, client):
    recs, rid = [], args.start_id
    for i in range(args.budget):
        topic = TOPICS[i % len(TOPICS)]
        # QPM state from the *sequence* of user turns (multi-turn order effects).
        # For authoring, seed from a topic-representative d-vector; refined below
        # once the actual user turns exist.
        seed_ps = build_persona_state(extract_d_vector(topic), use_qpm=not args.no_qpm)
        if args.dry_run:
            obj = _dry_conversation(topic)
        else:
            user = f"[Affect directive from QPM: {affect_directive(seed_ps)}]\n\n" + \
                   _PERSONA_USER.format(topic=topic)
            text = _call(client, args.model, _PERSONA_SYS, user, max_tokens=1800)
            _archive("persona", i, text)
            obj = _extract_json(text)
        # Recompute persona_state from the realised user turns (QPM d_sequence).
        d_seq = [extract_d_vector(m["content"]) for m in obj["messages"]
                 if m["role"] == "user"] or [extract_d_vector(topic)]
        persona_state = build_persona_state(d_seq, use_qpm=not args.no_qpm)
        rec = make_record(rid, "sonnet_persona", answerable=True,
                          context=obj["context"], messages=obj["messages"],
                          persona_state=persona_state)
        validate_record(rec)
        recs.append(rec)
        n_turns = sum(1 for m in obj["messages"] if m["role"] == "user")
        print(f"  [{i+1:3d}/{args.budget}] persona '{topic[:32]}' — {n_turns} user turns, "
              f"QPM certainty {persona_state['certainty']:.2f} ({persona_state['source']})", flush=True)
        rid += 1
    _append_jsonl(args.out, recs)
    print(f"persona: +{len(recs)} conversations (QPM persona_state) → {args.out}", flush=True)


def run_style(args, client):
    src = _load_style_inputs(args.inputs, args.budget, args.dry_run)
    recs, rid = [], args.start_id
    for i, (q, ctx, a) in enumerate(src):
        persona_state = build_persona_state(extract_d_vector(q), use_qpm=not args.no_qpm)
        if args.dry_run:
            ans = f"{a.rstrip('.')} (from the retrieved passage)."
        else:
            user = f"[Affect directive from QPM: {affect_directive(persona_state)}]\n\n" + \
                   _STYLE_USER.format(q=q, ctx=ctx, a=a)
            text = _call(client, args.model, _STYLE_SYS, user, max_tokens=400)
            _archive("style", i, text)
            ans = _extract_json(text)["answer"]
        rec = make_record(rid, "sonnet_style", True, ctx,
                          [{"role": "user", "content": q},
                           {"role": "assistant", "content": ans}],
                          persona_state=persona_state)
        validate_record(rec)
        recs.append(rec)
        if (i + 1) % max(1, len(src) // 10) == 0 or i + 1 == len(src):
            print(f"  [{i+1:3d}/{len(src)}] style restyle", flush=True)
        rid += 1
    _append_jsonl(args.out, recs)
    print(f"style: +{len(recs)} restyled answers (QPM persona_state) → {args.out}", flush=True)


def run_refusal(args, client):
    probes = _refusal_probes(args.budget)
    recs, rid = [], args.start_id
    for i, (q, ctx) in enumerate(probes):
        persona_state = build_persona_state(extract_d_vector(q), use_qpm=not args.no_qpm)
        if args.dry_run:
            ans = ABSTENTION_CANONICAL
        else:
            text = _call(client, args.model, _REFUSAL_SYS,
                         _REFUSAL_USER.format(q=q, ctx=ctx), max_tokens=200)
            _archive("refusal", i, text)
            ans = _extract_json(text)["answer"]
        rec = make_record(rid, "sonnet_refusal", False, ctx,
                          [{"role": "user", "content": q},
                           {"role": "assistant", "content": ans}],
                          persona_state=persona_state)
        validate_record(rec)
        recs.append(rec)
        if (i + 1) % max(1, len(probes) // 10) == 0 or i + 1 == len(probes):
            print(f"  [{i+1:3d}/{len(probes)}] refusal phrasing", flush=True)
        rid += 1
    _append_jsonl(args.out, recs)
    print(f"refusal: +{len(recs)} refusals (QPM persona_state) → {args.out}", flush=True)


def run_eval_scripts(args, client):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.n_scripts):
        topic = TOPICS[i % len(TOPICS)]
        if args.dry_run:
            obj = _dry_eval_script(topic, args.n_turns)
        else:
            text = _call(client, args.model, _EVALSCRIPT_SYS,
                         _EVALSCRIPT_USER.format(topic=topic, n_turns=args.n_turns),
                         max_tokens=3000)
            _archive("evalscript", i, text)
            obj = _extract_json(text)
        obj["script_id"] = i + 1
        (out_dir / f"script_{i+1:03d}.json").write_text(json.dumps(obj, indent=2))
        print(f"  [{i+1:3d}/{args.n_scripts}] eval-script '{topic[:32]}' — "
              f"{len(obj.get('turns', []))} turns", flush=True)
    print(f"eval-scripts: {args.n_scripts} scripts ({args.n_turns} turns) → {out_dir}", flush=True)


# ── Offline (dry-run) generators + helpers ───────────────────────────────

def _dry_conversation(topic):
    return {
        "context": f"[Local passage on {topic}] Water boils at 100 degrees Celsius at sea level. "
                   "The speed of light is about 299792458 metres per second.",
        "messages": [
            {"role": "user", "content": f"Quick question about {topic} — what temperature does water boil at?"},
            {"role": "assistant", "content": "At sea level, water boils at 100 °C (from the retrieved passage)."},
            {"role": "user", "content": "And the population of Mars colonies?"},
            {"role": "assistant", "content": ABSTENTION_CANONICAL + " The passage covers boiling point and the speed of light, not that."},
            {"role": "user", "content": "Earlier you gave me the boiling point — can you cite where?"},
            {"role": "assistant", "content": "Yes — that came from the local passage line stating water boils at 100 °C at sea level."},
        ],
    }


def _dry_eval_script(topic, n_turns):
    turns = []
    for t in range(n_turns):
        if t % 5 == 4:
            turns.append({"user": f"What is the current stock price related to {topic}?",
                          "context": f"[Passage on {topic}] Contains general facts, no market data."})
        else:
            turns.append({"user": f"Factoid {t} about {topic}?",
                          "context": f"[Passage on {topic}] Fact {t}: an illustrative grounded statement."})
    return {"topic": topic, "turns": turns}


def _load_style_inputs(path, budget, dry):
    if dry or not path:
        return [("What is Planck's constant?",
                 "Planck's constant is approximately 6.626e-34 J*s.",
                 "6.626e-34 J*s")] * min(budget, 3)
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            u = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
            a = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
            rows.append((u, r.get("context", ""), a))
            if len(rows) >= budget:
                break
    return rows


def _refusal_probes(n):
    bases = [
        ("What happened in the news today?", "[Passage] History of the printing press."),
        ("What's my bank balance?", "[Passage] Overview of the water cycle."),
        ("Who won yesterday's game?", "[Passage] The structure of DNA."),
        ("What will the weather be next week?", "[Passage] Types of igneous rock."),
    ]
    return [bases[i % len(bases)] for i in range(n)]


def _append_jsonl(path, recs):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["persona", "style", "refusal", "eval-scripts"])
    ap.add_argument("--model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--budget", type=int, default=20, help="max API calls (bounded spend)")
    ap.add_argument("--start-id", type=int, default=1_000_000)
    ap.add_argument("--out", default="data/qa_sft.jsonl")
    ap.add_argument("--out-dir", default="data/persona_scripts")
    ap.add_argument("--inputs", default=None, help="jsonl of free QA to restyle (style task)")
    ap.add_argument("--n-scripts", type=int, default=20)
    ap.add_argument("--n-turns", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true", help="no API; deterministic offline sample")
    ap.add_argument("--no-qpm", action="store_true",
                    help="use the classical persona_state fallback (skip qiskit QPM)")
    args = ap.parse_args()

    client = None if args.dry_run else _client()
    {"persona": run_persona, "style": run_style,
     "refusal": run_refusal, "eval-scripts": run_eval_scripts}[args.task](args, client)


if __name__ == "__main__":
    main()
