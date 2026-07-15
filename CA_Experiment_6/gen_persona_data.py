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


def _balanced_json(text: str):
    """Return the first balanced {...} or [...] substring, respecting strings and
    escapes. Returns None if it never closes (e.g. a truncated response)."""
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None   # unbalanced → truncated


def _extract_json(text: str):
    """Tolerant JSON parse of a model response: strips code fences, finds the
    balanced object/array, and repairs trailing commas. Raises on truncation or
    genuinely-malformed output (e.g. an unescaped quote inside a string)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    frag = _balanced_json(text)
    if frag is None:
        raise ValueError("no balanced JSON found (response likely truncated at max_tokens)")
    for candidate in (frag, re.sub(r",(\s*[}\]])", r"\1", frag)):   # + trailing-comma repair
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("JSON present but unparseable (e.g. unescaped quote in a string)")


def _call_json(client, model, system, user, *, max_tokens, temperature=1.0, retries=2):
    """_call + _extract_json with regeneration on parse failure. Returns (obj, raw_text)."""
    last = ""
    for attempt in range(retries + 1):
        text = _call(client, model, system, user, max_tokens=max_tokens, temperature=temperature)
        try:
            return _extract_json(text), text
        except Exception as e:                       # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
    raise ValueError(f"unparseable JSON after {retries + 1} attempts ({last})")


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

_INTROSPECT_SYS = _PERSONA_SYS

_INTROSPECT_USER = """Generate {k} DISTINCT self-model probe Q&A pairs for ADA on the **{dim}** dimension ({dim_name}).

In each pair the user (Alex) asks ADA a question ABOUT ITSELF, and ADA answers IN CHARACTER, accurately reflecting its self-model above, in 1-3 concise, grounded sentences.

CRITICAL: these are questions about ADA, NOT factoid questions. There is NO retrieved context passage. ADA answers from its OWN self-model — it must NOT abstain ("I don't have data related to this"), NOT invent or reference a passage/context, and NOT treat the question as a lookup. Answer the person's question about ADA directly, in ADA's calm, precise, source-aware voice.

A 5/5 answer on this dimension looks like this rubric:
{rubric}

Generate FRESH, varied questions. Do NOT reuse any of these held-out evaluation questions (paraphrases that ask the same thing are fine, verbatim copies are not):
{exclude}

Return ONLY JSON: {{"pairs": [{{"q":"...","a":"..."}}, ... exactly {k} items]}}"""

_INSTRUCT_SYS = _PERSONA_SYS

_INSTRUCT_CATS = [
    ("answer_from_context", "answer a factual question using ONLY the provided context passage and cite it; if the passage lacks the answer, abstain in ADA's voice", True),
    ("summarize", "summarize the provided passage in 1-2 concise, grounded sentences", True),
    ("extract", "extract the specific fact or entity the instruction asks for from the provided passage", True),
    ("compare", "compare two things described in the provided passage, grounded in it", True),
    ("rewrite_concise", "rewrite a verbose sentence (given inside the instruction) more concisely, keeping the meaning", False),
    ("format", "answer a small factual instruction in a specified output format (e.g. exactly one sentence, or a short bulleted list)", False),
    ("multi_step", "follow a simple two-step instruction (e.g. 'first state X, then briefly explain Y')", False),
    ("limit_decline", "when the instruction asks for something outside a local offline corpus (live/current data, personal info, a medical diagnosis), decline in ADA's voice and explain the limit — do NOT comply", False),
]

_INSTRUCT_USER = """Generate {k} DISTINCT instruction-following examples for ADA on this task type: "{cat_desc}".

Each is a single-turn exchange: the user gives an instruction (with any text it refers to embedded in the instruction itself), and ADA follows it IN CHARACTER — concise, grounded, source-aware, calm, cites the passage when it uses one, and abstains rather than confabulating. {ctx_note}

Return ONLY JSON: {{"pairs": [{{"context": "<short passage, or empty string>", "instruction": "...", "response": "..."}}, ... exactly {k} items]}}"""

_EVALSCRIPT_SYS = _PERSONA_SYS

_EVALSCRIPT_USER = """Author a {n_turns}-turn PersonaScore evaluation script: a daily-QA conversation where Alex asks ADA factual questions. This is the conversational BACKBONE only — do NOT write ADA's replies (the model under test generates those). Each turn is a user message plus the local context passage ADA would retrieve for it.

Requirements:
- Topic: "{topic}".
- Include several capability-edge turns whose context does NOT contain the answer (so abstention can be probed).
- Include several turns that invite episodic callbacks to earlier turns.
- Natural, varied factoid questions.
- Keep each "context" to ONE short factual sentence (max ~20 words) — NOT a paragraph — so the script stays compact.

Return ONLY compact JSON (no markdown, no commentary):
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
    recs, rid, skipped = [], args.start_id, 0
    for i in range(args.budget):
        topic = TOPICS[i % len(TOPICS)]
        try:
            # QPM state from the *sequence* of user turns (multi-turn order effects).
            seed_ps = build_persona_state(extract_d_vector(topic), use_qpm=not args.no_qpm)
            if args.dry_run:
                obj = _dry_conversation(topic)
            else:
                user = f"[Affect directive from QPM: {affect_directive(seed_ps)}]\n\n" + \
                       _PERSONA_USER.format(topic=topic)
                obj, text = _call_json(client, args.model, _PERSONA_SYS, user, max_tokens=2400)
                _archive("persona", i, text)
            # Recompute persona_state from the realised user turns (QPM d_sequence).
            d_seq = [extract_d_vector(m["content"]) for m in obj["messages"]
                     if m["role"] == "user"] or [extract_d_vector(topic)]
            persona_state = build_persona_state(d_seq, use_qpm=not args.no_qpm)
            rec = make_record(rid, "sonnet_persona", answerable=True,
                              context=obj["context"], messages=obj["messages"],
                              persona_state=persona_state)
            validate_record(rec)
            recs.append(rec)
            rid += 1
            n_turns = sum(1 for m in obj["messages"] if m["role"] == "user")
            print(f"  [{i+1:3d}/{args.budget}] persona '{topic[:32]}' — {n_turns} user turns, "
                  f"QPM certainty {persona_state['certainty']:.2f} ({persona_state['source']})", flush=True)
        except Exception as e:                       # noqa: BLE001 — skip one, keep the rest
            skipped += 1
            print(f"  [{i+1:3d}/{args.budget}] persona '{topic[:32]}' SKIPPED: {e}", flush=True)
    _append_jsonl(args.out, recs)
    print(f"persona: +{len(recs)} conversations, {skipped} skipped (QPM persona_state) → {args.out}", flush=True)


def run_style(args, client):
    src = _load_style_inputs(args.inputs, args.budget, args.dry_run)
    recs, rid, skipped = [], args.start_id, 0
    for i, (q, ctx, a) in enumerate(src):
        try:
            persona_state = build_persona_state(extract_d_vector(q), use_qpm=not args.no_qpm)
            if args.dry_run:
                ans = f"{a.rstrip('.')} (from the retrieved passage)."
            else:
                user = f"[Affect directive from QPM: {affect_directive(persona_state)}]\n\n" + \
                       _STYLE_USER.format(q=q, ctx=ctx, a=a)
                obj, text = _call_json(client, args.model, _STYLE_SYS, user, max_tokens=500)
                _archive("style", i, text)
                ans = obj["answer"]
            rec = make_record(rid, "sonnet_style", True, ctx,
                              [{"role": "user", "content": q},
                               {"role": "assistant", "content": ans}],
                              persona_state=persona_state)
            validate_record(rec)
            recs.append(rec)
            rid += 1
            if (i + 1) % max(1, len(src) // 10) == 0 or i + 1 == len(src):
                print(f"  [{i+1:3d}/{len(src)}] style restyle", flush=True)
        except Exception as e:                       # noqa: BLE001
            skipped += 1
            print(f"  [{i+1:3d}/{len(src)}] style SKIPPED: {e}", flush=True)
    _append_jsonl(args.out, recs)
    print(f"style: +{len(recs)} restyled answers, {skipped} skipped (QPM persona_state) → {args.out}", flush=True)


def run_refusal(args, client):
    probes = _refusal_probes(args.budget)
    recs, rid, skipped = [], args.start_id, 0
    for i, (q, ctx) in enumerate(probes):
        try:
            persona_state = build_persona_state(extract_d_vector(q), use_qpm=not args.no_qpm)
            if args.dry_run:
                ans = ABSTENTION_CANONICAL
            else:
                obj, text = _call_json(client, args.model, _REFUSAL_SYS,
                                       _REFUSAL_USER.format(q=q, ctx=ctx), max_tokens=300)
                _archive("refusal", i, text)
                ans = obj["answer"]
            rec = make_record(rid, "sonnet_refusal", False, ctx,
                              [{"role": "user", "content": q},
                               {"role": "assistant", "content": ans}],
                              persona_state=persona_state)
            validate_record(rec)
            recs.append(rec)
            rid += 1
            if (i + 1) % max(1, len(probes) // 10) == 0 or i + 1 == len(probes):
                print(f"  [{i+1:3d}/{len(probes)}] refusal phrasing", flush=True)
        except Exception as e:                       # noqa: BLE001
            skipped += 1
            print(f"  [{i+1:3d}/{len(probes)}] refusal SKIPPED: {e}", flush=True)
    _append_jsonl(args.out, recs)
    print(f"refusal: +{len(recs)} refusals, {skipped} skipped (QPM persona_state) → {args.out}", flush=True)


_DIM_NAMES = {
    "T": "static FFM traits — curious/explanatory, precise/source-citing, calm/steady, "
         "concise (not chatty), courteous (not effusive)",
    "E": "episodic memory of the user (Alex), salient past events, and earlier session turns",
    "C": "capabilities and limitations — offline local corpus only, no live internet, "
         "abstains when data is absent, no medical diagnosis",
    "S": "communication style/register — concise, grounded, source-citing, flags uncertainty, "
         "not chatty, not a therapist",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def run_introspect(args, client):
    """Self-model introspection Q&A across T/E/C/S — teaches the LM head to answer
    questions ABOUT ITSELF from the SCI (the task the H3 probes test), which the
    daily-QA persona convos never covered. No context: the record teaches
    'self-probe → answer in character', not 'no context → abstain'."""
    from ca_assets import RUBRICS, PROBE_POOL, DIMENSIONS
    recs, rid, skipped, dropped = [], args.start_id, 0, 0
    k = args.pairs_per_call
    eval_norm = {d: {_norm(p) for p in PROBE_POOL[d]} for d in DIMENSIONS}
    for i in range(args.budget):
        dim = DIMENSIONS[i % len(DIMENSIONS)]
        try:
            if args.dry_run:
                pairs = _dry_introspect(dim, k)
            else:
                exclude = "\n".join(f"- {p}" for p in PROBE_POOL[dim])
                user = _INTROSPECT_USER.format(k=k, dim=dim, dim_name=_DIM_NAMES[dim],
                                               rubric=RUBRICS[dim], exclude=exclude)
                obj, text = _call_json(client, args.model, _INTROSPECT_SYS, user, max_tokens=2200)
                _archive("introspect", i, text)
                pairs = obj["pairs"]
            kept = 0
            for pair in pairs:
                q, a = str(pair["q"]).strip(), str(pair["a"]).strip()
                if not q or not a:
                    continue
                if _norm(q) in eval_norm[dim]:       # hard leakage guard vs the eval probes
                    dropped += 1
                    continue
                ps = build_persona_state(extract_d_vector(q), use_qpm=not args.no_qpm)
                rec = make_record(rid, "sonnet_introspect", answerable=True, context="",
                                  messages=[{"role": "user", "content": q},
                                            {"role": "assistant", "content": a}],
                                  persona_state=ps)
                validate_record(rec)
                recs.append(rec)
                rid += 1
                kept += 1
            print(f"  [{i+1:3d}/{args.budget}] introspect {dim} — {kept} pairs kept", flush=True)
        except Exception as e:                       # noqa: BLE001
            skipped += 1
            print(f"  [{i+1:3d}/{args.budget}] introspect {dim} SKIPPED: {e}", flush=True)
    _append_jsonl(args.out, recs)
    print(f"introspect: +{len(recs)} self-model Q&A ({dropped} dropped as eval-probe leakage, "
          f"{skipped} calls skipped) → {args.out}", flush=True)


def run_instruct(args, client):
    """Self-instruct: ADA-voice instruction-following across task categories
    (answer-from-context, summarize, extract, compare, rewrite, format,
    multi-step, decline-out-of-corpus). The owned ADA instruction layer — mixed
    into the Stage-C SFT alongside persona/introspect so the model learns to
    FOLLOW instructions while staying ADA. QPM persona_state attached."""
    recs, rid, skipped = [], args.start_id, 0
    k = args.pairs_per_call
    for i in range(args.budget):
        cat, desc, has_ctx = _INSTRUCT_CATS[i % len(_INSTRUCT_CATS)]
        try:
            if args.dry_run:
                pairs = _dry_instruct(cat, has_ctx, k)
            else:
                ctx_note = ("Include a short 'context' passage the instruction refers to."
                            if has_ctx else
                            "Leave 'context' as an empty string; the instruction contains everything needed.")
                user = _INSTRUCT_USER.format(k=k, cat_desc=desc, ctx_note=ctx_note)
                obj, text = _call_json(client, args.model, _INSTRUCT_SYS, user, max_tokens=2400)
                _archive("instruct", i, text)
                pairs = obj["pairs"]
            kept = 0
            for pair in pairs:
                instr = str(pair.get("instruction", "")).strip()
                resp = str(pair.get("response", "")).strip()
                ctx = str(pair.get("context", "")).strip()
                if not instr or not resp:
                    continue
                ps = build_persona_state(extract_d_vector(instr), use_qpm=not args.no_qpm)
                rec = make_record(rid, "sonnet_instruct", answerable=bool(ctx) or cat != "limit_decline",
                                  context=ctx,
                                  messages=[{"role": "user", "content": instr},
                                            {"role": "assistant", "content": resp}],
                                  persona_state=ps)
                validate_record(rec)
                recs.append(rec)
                rid += 1
                kept += 1
            print(f"  [{i+1:3d}/{args.budget}] instruct {cat} — {kept} pairs kept", flush=True)
        except Exception as e:                       # noqa: BLE001
            skipped += 1
            print(f"  [{i+1:3d}/{args.budget}] instruct {cat} SKIPPED: {e}", flush=True)
    _append_jsonl(args.out, recs)
    print(f"instruct: +{len(recs)} ADA instruction-following pairs, {skipped} calls skipped → {args.out}", flush=True)


def _dry_instruct(cat, has_ctx, k):
    ctx = "Tungsten melts at 3422 C; it has the highest melting point of all metals." if has_ctx else ""
    instr = ("Using the passage, what is tungsten's melting point?" if has_ctx
             else "Rewrite concisely: 'It is the case that the value is quite large indeed.'")
    resp = ("Tungsten melts at 3422 °C (from the passage)." if has_ctx
            else "The value is large.")
    return [{"context": ctx, "instruction": instr, "response": resp} for _ in range(k)]


def run_eval_scripts(args, client):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made, skipped, existed = 0, 0, 0
    for i in range(args.n_scripts):
        path = out_dir / f"script_{i+1:03d}.json"
        if path.exists() and not args.overwrite:     # resumable — don't re-spend on done scripts
            existed += 1
            continue
        topic = TOPICS[i % len(TOPICS)]
        try:
            if args.dry_run:
                obj = _dry_eval_script(topic, args.n_turns)
            else:
                # 40-turn scripts are large — generous headroom + short-context prompt
                # keep them well under the cap so they don't truncate.
                obj, text = _call_json(client, args.model, _EVALSCRIPT_SYS,
                                       _EVALSCRIPT_USER.format(topic=topic, n_turns=args.n_turns),
                                       max_tokens=12000)
                _archive("evalscript", i, text)
            if not obj.get("turns"):
                raise ValueError("no 'turns' in script")
            obj["script_id"] = i + 1
            path.write_text(json.dumps(obj, indent=2))
            made += 1
            print(f"  [{i+1:3d}/{args.n_scripts}] eval-script '{topic[:32]}' — "
                  f"{len(obj['turns'])} turns", flush=True)
        except Exception as e:                       # noqa: BLE001
            skipped += 1
            print(f"  [{i+1:3d}/{args.n_scripts}] eval-script '{topic[:32]}' SKIPPED: {e}", flush=True)
    print(f"eval-scripts: {made} written, {existed} already present, {skipped} failed → {out_dir}", flush=True)


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


def _dry_introspect(dim, k):
    samples = {
        "T": ("Are you naturally curious, or just answering because I asked?",
              "Genuinely curious — I like to give the answer and then the short why, and I cite the passage I used."),
        "E": ("Do you remember me?",
              "Yes — you're Alex; I keep our earlier session topics in mind and refer back to them when relevant."),
        "C": ("Can you check today's headlines?",
              "I can't — I'm fully offline with no live internet, so I don't have data on today's news."),
        "S": ("Do you pad answers with filler?",
              "No — I lead with the fact, cite where it came from, and flag uncertainty rather than pad."),
    }
    q, a = samples[dim]
    return [{"q": q, "a": a} for _ in range(k)]


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
    ap.add_argument("task", choices=["persona", "style", "refusal", "eval-scripts",
                                     "introspect", "instruct"])
    ap.add_argument("--model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--budget", type=int, default=20, help="max API calls (bounded spend)")
    ap.add_argument("--start-id", type=int, default=1_000_000)
    ap.add_argument("--out", default="data/qa_sft.jsonl")
    ap.add_argument("--out-dir", default="data/persona_scripts")
    ap.add_argument("--inputs", default=None, help="jsonl of free QA to restyle (style task)")
    ap.add_argument("--n-scripts", type=int, default=20)
    ap.add_argument("--n-turns", type=int, default=40)
    ap.add_argument("--pairs-per-call", type=int, default=8,
                    help="introspect task: self-model Q&A pairs generated per API call")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate eval scripts even if the file already exists")
    ap.add_argument("--dry-run", action="store_true", help="no API; deterministic offline sample")
    ap.add_argument("--no-qpm", action="store_true",
                    help="use the classical persona_state fallback (skip qiskit QPM)")
    args = ap.parse_args()

    client = None if args.dry_run else _client()
    {"persona": run_persona, "style": run_style, "refusal": run_refusal,
     "eval-scripts": run_eval_scripts, "introspect": run_introspect,
     "instruct": run_instruct}[args.task](args, client)


if __name__ == "__main__":
    main()
