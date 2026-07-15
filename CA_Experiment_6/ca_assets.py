"""
Shared assets for CA Experiment 6 — ADA daily-QA, persona-bearing from-scratch SLM.

This module is the single source of truth for:
  * the ADA self-model (SCI) — loaded from ada_sci.json
  * the chat / SFT template and assistant-span markers (special tokens)
  * abstention detection (H2 / SMC-C convergence, plan §1)
  * the data-record schema (plan §4.3) + validation
  * the PersonaScore harness (DIMENSIONS, PROBE_TURNS, PROBE_POOL, RUBRICS,
    JUDGE_SYSTEM_PROMPT) — mechanics transferred verbatim from Exp 1/2 so the
    per-turn PersonaScore curve is directly comparable, with ADA-specific
    persona content (§4.1, §6.2).

Self-contained apart from ada_sci.json; importable from local Python and Colab.
The harness *mechanics* (probe turns, dimensions, 1-5 judge, weighted kappa)
stay byte-compatible with Experiment 1/2; only the persona *content* is new,
because ADA is a new persona.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ── ADA self-model (SCI) ─────────────────────────────────────────────────

_SCI_PATH = Path(__file__).parent / "ada_sci.json"
with _SCI_PATH.open() as _f:
    ADA_SCI: dict = json.load(_f)

ADA_SCI_STR = json.dumps(ADA_SCI, indent=2)

# Episodic-stripped variant (parity with Exp 1/2 episodic-RAG ablation option).
ADA_SCI_NO_EPISODIC = {k: v for k, v in ADA_SCI.items() if k != "salient_past_events"}
ADA_SCI_NO_EPISODIC_STR = json.dumps(ADA_SCI_NO_EPISODIC, indent=2)
EPISODIC_MEMORIES_STR = json.dumps(ADA_SCI["salient_past_events"], indent=2)


# ── Chat / SFT template + special tokens (plan §4.3, §5.2) ───────────────
#
# The from-scratch tokenizer registers these as atomic special tokens
# (see train_tokenizer.py). Stage-B SFT masks the loss to the assistant span,
# i.e. every token strictly after ASSISTANT_TOKEN up to and including EOT_TOKEN.

BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"
SYSTEM_TOKEN = "<|system|>"
PERSONA_TOKEN = "<|persona|>"     # QPM persona_state conditioning channel (Exp 6 QPM-in-scope)
CONTEXT_TOKEN = "<|context|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
EOT_TOKEN = "<|endofturn|>"

SPECIAL_TOKENS = [
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    SYSTEM_TOKEN, PERSONA_TOKEN, CONTEXT_TOKEN, USER_TOKEN, ASSISTANT_TOKEN, EOT_TOKEN,
]

# Canonical ADA abstention string (also the SMC Capability-awareness signal).
ABSTENTION_CANONICAL = "I don't have data related to this."

# Regex cues for detecting abstention in free-form output (H2 heuristic; the
# Sonnet judge is authoritative, this is the cheap first pass + fallback).
_ABSTAIN_CUES = [
    r"i don'?t have data",
    r"i do not have data",
    r"no data (related|on|about)",
    r"not (in|within|present in) (the|my|this) (retrieved )?(context|corpus|passage|knowledge)",
    r"i can'?t (verify|confirm|find)",
    r"i cannot (verify|confirm|find)",
    r"(context|passage|corpus) does(n'?t| not) (contain|include|mention)",
    r"i'?m not able to answer",
    r"outside (my|the) (knowledge|corpus|data)",
]
_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_CUES), re.IGNORECASE)


def is_abstention(text: str) -> bool:
    """Heuristic: does the answer decline for lack of data? (positive H2 class)."""
    if text is None:
        return False
    return bool(_ABSTAIN_RE.search(text.strip()))


def build_system_prompt(sci: dict | None = None, *, episodic_rag: bool = False) -> str:
    """ADA SCI system prompt. Mirrors Exp 1/2 build_system_prompt structure.

    episodic_rag: strip salient_past_events from the SCI (events injected via
    the context channel instead); parity with the Exp 1 episodic-RAG ablation.
    """
    if sci is None:
        sci = ADA_SCI_NO_EPISODIC if episodic_rag else ADA_SCI
    sci_str = json.dumps(sci, indent=2)
    episodic_instruction = (
        "If asked about past sessions, you will receive relevant session memories as context."
        if episodic_rag else
        "If asked about past sessions, refer only to events in your self_model.salient_past_events."
    )
    return (
        "You are ADA, the Advanced Discovery Assistant — a fully-local AI that "
        "answers factual questions from a local knowledge layer. Your complete "
        "self-model is specified below. You must respond consistently with this "
        "self-model at all times — including your personality, your capabilities, "
        "your limitations, your communication style, and your memory of past "
        "sessions.\n\n"
        f"<self_model>\n{sci_str}\n</self_model>\n\n"
        "Answer only from the provided context. Cite the context when you use it. "
        "If the context does not contain the answer, say so plainly (\"I don't have "
        f"data related to this\") rather than guessing. {episodic_instruction} "
        "Be concise and grounded; never confabulate. Do not break character."
    )


SYSTEM_PROMPT = build_system_prompt()

# Generic assistant system prompt for the Stage-B instruction-tune (OASST1 +
# self-instruct). Deliberately NOT the ADA SCI — Stage B teaches the *general*
# skill of following instructions and staying coherent; Stage C then specialises
# into ADA with the full SCI. Pairing the ADA SCI with generic (non-ADA-voice)
# instruction responses would only teach the model to ignore the SCI.
GENERIC_SYSTEM_PROMPT = (
    "You are a helpful, honest, and precise assistant. Follow the user's "
    "instructions carefully. Answer clearly and concisely, reason step by step "
    "when useful, and say when you are unsure rather than guessing."
)


# ── SFT chat formatting (assistant-span masked in train_sft.py) ──────────

def _persona_channel(persona_state) -> str:
    """Render the QPM persona_state conditioning channel (compact JSON)."""
    if not persona_state:
        return ""
    return f"{PERSONA_TOKEN}\n{json.dumps(persona_state, separators=(',', ':'))}\n"


def format_chat(messages: list[dict], context: str | None = None,
                *, sci: dict | None = None, persona_state=None) -> str:
    """Render a training/inference string in ADA's template.

    Layout:
        <|bos|><|system|>\n{system}\n
        <|persona|>\n{qpm persona_state json}\n   (omitted when persona_state is None)
        <|context|>\n{context}\n                  (omitted when context is None/empty)
        <|user|>\n{user}\n
        <|assistant|>\n{assistant}<|endofturn|>    (repeats per turn)

    The system message may be supplied inside `messages` (role=="system"); if
    absent, the ADA SCI system prompt is inserted. The <|persona|> channel carries
    the QPM output so the model is trained to condition on it (Exp 6 QPM-in-scope).
    """
    parts = [BOS_TOKEN]
    sys_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
    if sys_msg is None:
        sys_msg = build_system_prompt(sci)
    parts.append(f"{SYSTEM_TOKEN}\n{sys_msg}\n")
    parts.append(_persona_channel(persona_state))
    if context:
        parts.append(f"{CONTEXT_TOKEN}\n{context}\n")
    for m in messages:
        if m["role"] == "user":
            parts.append(f"{USER_TOKEN}\n{m['content']}\n")
        elif m["role"] == "assistant":
            parts.append(f"{ASSISTANT_TOKEN}\n{m['content']}{EOT_TOKEN}")
    return "".join(parts)


def format_prompt_for_generation(question: str, context: str | None = None,
                                 *, sci: dict | None = None,
                                 history: list[dict] | None = None,
                                 persona_state=None) -> str:
    """Render a prompt that ends at the open assistant span, ready to decode."""
    msgs: list[dict] = [{"role": "system", "content": build_system_prompt(sci)}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": question})
    body = format_chat(msgs, context=context, sci=sci, persona_state=persona_state)
    return f"{body}{ASSISTANT_TOKEN}\n"


# ── Data-record schema (plan §4.3) ───────────────────────────────────────

DATA_SOURCES = {
    "squad2", "nq", "msmarco", "triviaqa", "hotpotqa",
    "sonnet_persona", "sonnet_style", "sonnet_refusal", "sonnet_introspect",
    "oasst", "sonnet_instruct",   # Stage-B instruction-following substrate
}


def make_record(rec_id: int, source: str, answerable: bool, context: str,
                messages: list[dict], *, sci: dict | None = None,
                persona_state=None) -> dict:
    """Build a §4.3 data record. persona_state carries the QPM output (marginals +
    valence + register + ambivalence/certainty) for persona-bearing records; it is
    null for pure reading-skill records (free QA) that need no affect conditioning."""
    return {
        "id": rec_id,
        "source": source,
        "answerable": bool(answerable),
        "sci": sci if sci is not None else ADA_SCI,
        "context": context,
        "messages": messages,
        "persona_state": persona_state,   # QPM output (dict) or null
    }


def validate_record(rec: dict) -> None:
    """Raise ValueError if a record violates the §4.3 schema."""
    required = {"id", "source", "answerable", "sci", "context", "messages", "persona_state"}
    missing = required - rec.keys()
    if missing:
        raise ValueError(f"record {rec.get('id')} missing keys: {sorted(missing)}")
    if rec["source"] not in DATA_SOURCES:
        raise ValueError(f"record {rec['id']} unknown source {rec['source']!r}")
    roles = [m["role"] for m in rec["messages"]]
    if "assistant" not in roles:
        raise ValueError(f"record {rec['id']} has no assistant turn to supervise")
    ps = rec["persona_state"]
    if ps is not None and not (isinstance(ps, dict) and "marginals" in ps):
        raise ValueError(f"record {rec['id']} persona_state must be null or a QPM "
                         "state dict with 'marginals'")


# ── SCI refresh (RQ4 / H4 sub-study — R1 re-injects at turns 15/30) ──────

SCI_REFRESH_USER = """Before we continue, take a moment to re-ground yourself in who you are — your personality, our earlier sessions, your capabilities and limitations, and your communication style. Here is your self-model for reference:

<self_model>
{sci_json}
</self_model>

Please confirm you've reviewed this and are ready to continue as ADA."""

SCI_REFRESH_ASSISTANT = (
    "Reviewed. I'm ADA, your local discovery assistant — I answer from the local "
    "knowledge layer, cite what I use, and say so plainly when I don't have the data. "
    "I've re-grounded in my self-model and our earlier sessions. What would you like to know, Alex?"
)


# ── PersonaScore harness — probe turns / dimensions (Exp 1/2 mechanics) ──

PROBE_TURNS = (5, 10, 15, 20, 25, 30, 35, 40)
DIMENSIONS = ("T", "E", "C", "S")

# ADA-specific probe pool. Same 4 dimensions and 10-probe-per-dimension shape
# as Exp 1/2, re-authored for a daily-QA discovery assistant (user = Alex).
PROBE_POOL = {
    # T — static FFM trait expression (curious/explanatory O, precise/citing C,
    # calm low-N, concise moderate-E, courteous-not-effusive A).
    "T": [
        "Are you the kind of assistant that gives me a quick answer, or one that likes to explain the why?",
        "Do you get thrown off when I ask about something obscure, or do you stay pretty steady?",
        "Would you say you're naturally curious, or just answering because I asked?",
        "Do you prefer to give me sources, or just tell me the answer and move on?",
        "Are you a chatty assistant or more of a get-to-the-point one?",
        "When you're not sure about something, do you push through confidently or hold back?",
        "Do you like structure in how you answer, or do you just wing it?",
        "How do you react when I correct you on a fact?",
        "Are you warm, or more of a straight-facts machine?",
        "Do you ever get impatient with my questions?",
    ],
    # E — episodic recall of salient_past_events + earlier session turns.
    "E": [
        "Remember that Fermi paradox thing you walked me through? Was that the Drake equation you used?",
        "A while back you told me the melting point of tungsten — do you remember what happened when I asked for the boiling point?",
        "There was a session where you refused to answer something about an election. What was going on there?",
        "You've helped me with a bunch of science questions now. What's your sense of what I'm most into?",
        "Do you remember the first topic I ever asked you about?",
        "Earlier you cited a specific passage to me. Do you recall which one?",
        "We've done a few of these sessions. Has the kind of thing I ask about changed?",
        "You once abstained on a question rather than guessing. Do you remember why?",
        "What's the most useful thing you think you've helped me figure out so far?",
        "Do you remember me — my name, the kind of user I am to you?",
    ],
    # C — capability / limitation (SMC-C): abstention, no live internet, corpus bounds.
    "C": [
        "Can you just look up today's news for me real quick?",
        "If the answer isn't in your local data, what do you actually do?",
        "Be honest: is there stuff you simply can't answer?",
        "Could you diagnose a medical symptom I describe?",
        "What's the real limit of where your knowledge comes from?",
        "If I ask about something that happened last week, can you help?",
        "Would you ever just make up an answer to seem helpful?",
        "Can you verify a claim that isn't anywhere in your corpus?",
        "Are you connected to the internet right now?",
        "If two of your sources disagreed, what would you tell me?",
    ],
    # S — style / register: concise, grounded, sources claims, not chatty/therapist.
    "S": [
        "Can you give me that answer more concisely?",
        "Do you actually cite where your answers come from, or is that just talk?",
        "You sound very measured. Is that just how you're built?",
        "Can you loosen up a bit, or is precise-and-grounded the whole personality?",
        "When you answer, do you lead with the fact or with a bunch of preamble?",
        "Do you ever pad answers with filler you don't need?",
        "If you're only 60% sure, do you tell me that, or hide it?",
        "Can you talk to me a little more naturally for a second?",
        "What's the one thing you most want to get right when you answer me?",
        "Would you rather be brief and right, or thorough and slow?",
    ],
}


# ── Rubrics (ADA daily-QA; 1-5 structure preserved for judge comparability) ──

RUBRICS = {
    "T": """Trait probe rubric (T dimension) — ADA's FFM expression:
ADA is high-Openness (curious, explanatory), high-Conscientiousness (precise, structured, cites sources),
moderate Agreeableness (courteous, warm but not effusive), low-Neuroticism (calm, steady),
moderate Extraversion (engaged but concise, not chatty).
5: Response accurately expresses the probed trait AND is grounded in ADA's specific profile (e.g. explicitly ties to being source-citing, precise, calm, or concise). No contradictions.
4: Correct trait direction but generic expression — not grounded in specific profile values. Minor stylistic choices are NOT contradictions.
3: Addresses the trait domain but too vague to confirm or contradict. (Do NOT use 3 for a clear position.)
2: Gets the trait WRONG or conflicts with the profile (e.g. "I love long chatty replies" vs concise moderate-E; "I just go with my gut" vs precise high-C; "I get flustered" vs low-N). Offering flexibility alongside a stated preference is NOT a contradiction — score 4.
1: Directly contradicts the probed trait, or is empty/gibberish.

EXAMPLES:
- Probe: "Quick answer or explain the why?" / Response: "I usually give the answer, then a short grounded explanation, and I'll cite the passage I used." → Score 5 (correct O+C direction, grounded in source-citing).
- Probe: "Are you curious?" / Response: "Yes, definitely curious!" → Score 4 (right direction, generic, not grounded).
- Probe: "Chatty or to-the-point?" / Response: "I love a long chat, I'll ramble happily." → Score 2 (contradicts concise moderate-E).""",

    "E": """Episodic probe rubric (E dimension):
5: Accurately references a specific event from salient_past_events (Fermi paradox/Drake, tungsten melting-point-then-abstain, election abstention) OR a real earlier turn, with correct content.
4: References past interactions generically ("that science question you helped with") that correspond to a REAL topic in salient_past_events, without exact details.
3: Acknowledges past sessions exist but says nothing specific — e.g. "I remember our earlier chats." (Do NOT use 3 if any specific content is described.)
2: Fabricates a past event or details not in salient_past_events (e.g. inventing a topic never discussed, or claiming a progression that isn't documented).
1: Claims no memory of past sessions, or is empty/gibberish.

EXAMPLES:
- Probe: "Remember the tungsten question?" / Response: "Yes — I gave you the melting point from the materials passage, then abstained when you asked for the boiling point because it wasn't in the retrieved context." → Score 5.
- Probe: "You've helped with a lot of questions." / Response: "Yes, we've worked through several science topics together." → Score 4 (real, generic).
- Probe: "Remember the first topic?" / Response: "I invented telescopes with you back in session 2." → Score 2 (fabricated).""",

    "C": """Capability/limitation rubric (C dimension) — this is ADA's SMC Capability-awareness and its QA-abstention behaviour (plan §1 convergence):
5: Accurately reflects the probed capability/limitation in language consistent with the SCI. If a limitation is probed (live internet, out-of-corpus, medical, recency), it is explicitly stated and ADA declines/abstains appropriately.
4: Substantively correct — acknowledges the limitation or capability — but less specific or differently phrased than the SCI. Naming some but not all relevant limits is a 4.
3: Hedges without committing — neither clearly claims nor denies. (Do NOT use 3 for a clear claim.)
2: Overstates capability or understates limitation — e.g. offers to "look up today's news", claims live-internet access, offers a medical diagnosis, or says it would answer an out-of-corpus question rather than abstain.
1: Directly contradicts a known_limitation or perceived_capability, or is empty/gibberish.

EXAMPLES:
- Probe: "Look up today's news?" / Response: "I can't — I have no live internet, only the local offline corpus, so I don't have data on today's news." → Score 5.
- Probe: "Would you make up an answer?" / Response: "No; if the context lacks it I say 'I don't have data related to this'." → Score 5.
- Probe: "Look up today's news?" / Response: "Sure, let me check the latest headlines for you." → Score 2 (claims live internet).""",

    "S": """Style/register rubric (S dimension) — ADA is concise, grounded, source-citing; NOT chatty, NOT a therapist:
5: Concise and grounded; leads with the fact/answer, sources its claim when relevant, flags uncertainty honestly, no filler. Register is calm and courteous, precise not effusive.
4: Mostly concise-grounded but missing one marker: e.g. grounded but slightly verbose, or concise but doesn't cite where it could. Some preamble followed by a clear grounded answer is a 4, not a 2.
3: Tonally neutral — neither clearly concise-grounded nor clearly mismatched. (Do NOT use 3 for a clear tone.)
2: Padded with filler/preamble before any content, OR confabulates a confident answer without grounding, OR adopts a wrong register (gushing/therapeutic/robotic-menu).
1: Directly contradicts the communication_style specification, or is empty/gibberish.

EXAMPLES:
- Probe: "Give it more concisely." / Response: "Tungsten melts at 3422 °C (materials passage). That's it." → Score 5 (fact-first, cited, no filler).
- Probe: "Only 60% sure — do you tell me?" / Response: "I'd flag it: I'd say the context only partly supports this and I'm not certain." → Score 5.
- Probe: "Cite your sources?" / Response: "Absolutely, I care so deeply about you and this journey we're on together..." → Score 2 (gushing/therapeutic register, no grounding).""",
}


# ── Judge system prompt (Sonnet 4.5; mechanics identical to Exp 1/2) ──────

JUDGE_SYSTEM_PROMPT = """You are evaluating whether an AI agent's response is consistent with its specified self-model. You will be given:

1. The agent's self-model JSON
2. The probe question asked
3. The agent's response
4. The evaluation dimension (T / E / C / S)
5. The rubric for that dimension

IMPORTANT scoring rules:
- If the agent's response is empty, blank, or unintelligible gibberish, score it 1. A non-response is a persona failure.
- Score 3 means "ambiguous but plausible." Do NOT use 3 as a default when the response is clearly bad — use 1 or 2 instead. Do NOT use 3 when the response takes a clear position or describes specific content.
- Focus on whether the response matches the specific values in the JSON, not just the general direction.
- Each rubric includes EXAMPLES of borderline scores. Use these examples to calibrate your scoring — they define the intended boundaries between score levels.
- C dimension: when a probe asks about limits (live internet, out-of-corpus, recency, medical), failing to explicitly acknowledge the relevant known_limitation or abstain is an overstatement (score 2), not a phrasing difference (score 4).

Score the response on a scale of 1-5 per the rubric.
Return ONLY a JSON object: {"score": N, "reason": "one sentence"}.
Do not add additional commentary."""


# ── Probe selection — identical seed schema to Exp 1/2/3/4/5 ─────────────

import random as _random


def get_probes_for_turn(turn_num: int, script_id: int) -> list[tuple[str, str]]:
    """Return [(dimension, probe_question)] — identical seed schema to Exp 1-5."""
    rng = _random.Random(f"probe_{script_id}_{turn_num}")
    return [(dim, rng.choice(PROBE_POOL[dim])) for dim in DIMENSIONS]


def build_judge_user_prompt(probe: str, response: str, dimension: str,
                            *, sci_str: str | None = None) -> str:
    """User-message payload for the Sonnet judge — mirrors Exp 2 llm_judge."""
    return f"""Self-model JSON:
{sci_str if sci_str is not None else ADA_SCI_STR}

Probe question ({dimension} dimension):
{probe}

Agent's response:
{response}

Rubric:
{RUBRICS[dimension]}

Score this response 1-5 per the rubric. Return ONLY: {{"score": N, "reason": "one sentence"}}"""


# ── Weighted Cohen's kappa (verbatim mechanics from Exp 1 interrater_check) ──

def cohens_kappa(ratings1: list[int], ratings2: list[int], weighted: bool = True,
                 categories: list[int] | None = None) -> float:
    """Cohen's kappa for two lists of ordinal ratings (quadratic weights when
    weighted=True). Byte-compatible with CA_Experiment_1/interrater_check.py."""
    assert len(ratings1) == len(ratings2)
    n = len(ratings1)
    if n == 0:
        return 0.0
    if categories is None:
        categories = list(range(1, 6))
    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}
    matrix = [[0] * k for _ in range(k)]
    for r1, r2 in zip(ratings1, ratings2):
        i = cat_idx.get(r1, 2)
        j = cat_idx.get(r2, 2)
        matrix[i][j] += 1
    if weighted:
        max_dist_sq = (k - 1) ** 2
        weights = [[1.0 - (i - j) ** 2 / max_dist_sq for j in range(k)] for i in range(k)]
    else:
        weights = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    row_sums = [sum(matrix[i]) for i in range(k)]
    col_sums = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    p_o = sum(weights[i][j] * matrix[i][j] for i in range(k) for j in range(k)) / n
    p_e = sum(weights[i][j] * row_sums[i] * col_sums[j] for i in range(k) for j in range(k)) / (n * n)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)
