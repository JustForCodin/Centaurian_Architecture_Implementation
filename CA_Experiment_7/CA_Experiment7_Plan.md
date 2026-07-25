# Experiment 7: Long-Horizon Episodic Memory for the ADA Daily-QA Agent

**CA Research Program — Second test of the per-scenario, from-scratch, owned small-model direction**
**Version 1.0 | July 2026** (pre-registered; thresholds frozen after the §5.4 pilot, before any eval is scored)
**Infrastructure:** Google Colab Pro (A100/L4) + local MiniLM embeddings (free) + bounded Anthropic API (Claude Sonnet 4.6 for long-script generation, Sonnet 4.5 as judge). Reuses the Experiment 6 checkpoints (`sft_ada_final.pt`) — this is primarily an **inference-time memory-architecture** experiment, not a from-scratch training run.

---

## 0. Position

Experiment 6 validated a from-scratch 321M model as ADA's daily-QA agent — reading (H1 0.77), abstaining (H2 0.78), and holding the ADA persona across **40 turns** (H3 3.80), with a clean *refresh-unnecessary* finding (H4). But it also surfaced a structural fact: the SCI system prompt (1338 tokens) **overflows the 1024-token context window**, so `_fit_prompt_ids` drops **all** conversation history at every turn. The Exp 6 agent is therefore **stateless per-turn** — each answer is `(truncated SCI) + (this turn's retrieved context) + (this turn's question)`, with nothing from the conversation itself. The flat H3 curve (T\* = None over 40 turns) is the fingerprint of that statelessness.

Real daily use spans **hundreds** of turns with genuine callbacks ("earlier you told me X — what was it?"). A stateless agent cannot support them: it never held turn 5 when it reaches turn 300. Experiment 7 asks whether an **external memory layer around the transducer** — principally **episodic-RAG over the live session** — gives real long-horizon continuity, and at what cost. It is the natural follow-up: the model is the transducer; long-horizon memory belongs in a layer around it, consistent with CA's separation of the SLM from the knowledge/memory layer.

---

## 1. Purpose and the bet under test

**The bet.** A small, fixed-context transducer + a bounded episodic-RAG memory can sustain a coherent, persona-consistent, episodically-grounded conversation over 100–500 turns — recalling facts planted far outside any window — while keeping **per-turn context and latency bounded** (O(1) per turn despite O(n) memory growth). If true, the owned-small-model direction extends from "40-turn QA" to "long-running daily agent" without a bigger model — memory is architecture, not scale.

**Why this is the right test now.** Exp 6 gives us a validated agent and a validated persona; Exp 7 changes only what surrounds it. The MiniLM retriever/reranker already exist in the CA stack (`retriever.py` / `rerank_util.py`, reused from Exp 1/2). The one prerequisite is a **compact SCI** (~450 tokens) so that anything *other* than the SCI can fit in the window — without it, no memory experiment is possible.

---

## 2. Research questions

- **RQ1 — drift.** Does the agent degrade over 100–500 turns? *Prediction:* no — statelessness (C0) means no accumulating state to drift, so the curve is flat; conversation length itself is not the enemy.
- **RQ2 — RAG recall.** Does episodic-RAG (C2) deliver genuine long-range recall that the stateless model *structurally cannot* — facts planted far outside any window?
- **RQ3 — persona at length.** Does PersonaScore (T/E/C/S) hold at 100/300/500 turns under the memory layer, and does E specifically lift once real within-session recall is possible?
- **RQ4 — bounded cost.** Is the architecture O(1)-per-turn in injected context and latency despite O(n) memory-store growth?

---

## 3. Hypotheses (pre-registered)

Thresholds **provisional**, frozen after the §5.4 pilot, before eval scoring. All metrics on held-out long scripts the model never trained on. Distances are measured as **lag** = turns between when a fact is planted and when it is probed.

- **H1 (statelessness / no drift).** Under **C0** (no memory), mean PersonaScore over 100/300/500 turns stays within **±0.2** of the Exp 6 40-turn level (3.80) — flat, no length-driven decay — and long-range recall of out-of-window anchors is at fabrication baseline (**≤ 2.0** on the E rubric). Establishes the floor.
- **H2 (RAG recall — headline).** Under **C2** (episodic-RAG), long-range recall accuracy on **out-of-window anchors** (lag ≫ sliding-window size) is **≥ 0.70**, and **significantly higher** than C0 (paired, p < 0.05) and than **C1** on anchors beyond the sliding window.
- **H3 (persona holds at length).** Under **C2**, mean PersonaScore is **≥ 3.5** at *each* of 100/300/500 turns, with the **E** dimension higher than under C0 (real recall now possible).
- **H4 (bounded cost).** Under **C2**, mean injected-context tokens and per-turn latency at 500 turns stay within **1.5×** of their 100-turn values — approximately constant despite the store growing 5× — confirming O(1)-per-turn context.

**Decision rule (committed before eval scores observed):**

| H2 | H3 | → action |
|:--:|:--:|---|
| ✓ | ✓ | **Adopt episodic-RAG** as the long-running daily agent's memory architecture; set the retrieval budget from the H4 cost profile; the memory layer transfers to the future coding agent's session memory. |
| ✓ | ✗ | Recall works but persona degrades at length — memory injection is disrupting style/trait. Tune the injection format (channel vs inline, snippet phrasing); re-run persona only. |
| ✗ | — | The frozen Exp 6 model cannot consume retrieved memory zero-shot → **trigger Phase B** (memory-aware SFT, §4.2) and re-run C2. |

H1 governs interpretation throughout (it certifies the stateless baseline that the memory conditions are measured against).

---

## 4. Design

### 4.1 Conditions (ablation)

All three share the **compact SCI** (§5.1) and the frozen Exp 6 `sft_ada_final.pt`; they differ only in what conversational memory is injected:

| Condition | Memory injected each turn | Tests |
|---|---|---|
| **C0 — no memory** | nothing (compact SCI only) | the stateless floor (RQ1/H1) |
| **C1 — sliding window** | the last *N* raw turns that fit a fixed token budget | "is recent context enough?" |
| **C2 — episodic-RAG** | top-*k* past exchanges retrieved by semantic similarity to the current query, within the same budget | genuine long-range recall (RQ2/H2) |

C1 and C2 use the **same token budget** for injected memory, so the comparison isolates *retrieval* (semantic, whole-history) vs *recency* (last-N) — not how much context each gets.

### 4.2 Two phases

- **Phase A (primary) — frozen agent + memory architecture.** Run C0/C1/C2 on the Exp 6 model as-is. Answers "does an external memory layer help the agent we already have, zero-shot?" Cheapest and cleanest.
- **Phase B (conditional) — memory-aware SFT.** *Only if Phase A's H2 fails* (the frozen model doesn't reliably consume retrieved snippets): a targeted Stage-C-style SFT on the compact SCI with training examples that carry retrieved-memory snippets + long-range callbacks, making the model *memory-native*; then re-run C2. Mirrors the Exp 6 method (measure → diagnose → fix with data).

---

## 5. Memory architecture

### 5.1 Compact SCI (prerequisite)

A `build_system_prompt(compact=True)` variant renders the full ADA self-model tersely — **~450 tokens** vs 1338 — preserving every field (personality summary, capabilities, known limitations, communication style, `user_model` = Alex, `salient_past_events` compressed to one line each). This frees ~500+ tokens of the window for injected memory + the turn's retrieved context. The **judge is unaffected** — it scores against the full `ADA_SCI_STR` independently, so H3 stays comparable to Exp 6. Train/inference note: the Exp 6 model was trained with the full (truncated) SCI; the compact SCI is a faithful terser subset to stay in-distribution. If Phase A shows a compact-SCI penalty, the Phase B SFT retrains on the compact SCI directly.

### 5.2 Episodic-RAG session store (C2)

- **Index.** After each turn *t*, embed the `(user, assistant)` exchange with **all-MiniLM-L6-v2** (already in the CA stack); append `{turn_id, text, embedding}` to an in-session store (flat / FAISS cosine index).
- **Retrieve.** At turn *t+1*, embed the user query; retrieve **top-k** (pilot k ∈ {3,5}) most-similar past exchanges; optionally rerank with **ms-marco-MiniLM**. Inject under a dedicated `<|memory|>` span (or inline) as compact snippets: `[earlier, turn N] you told Alex: …`.
- **Budget.** Cap injected memory at **M tokens** (pilot M ≈ 300) so the window stays bounded regardless of store size — this is the knob RQ4/H4 measure.

### 5.3 Sliding window (C1)

Same M-token budget, filled with the **last-N raw turns** (no retrieval). The recency baseline.

---

## 6. Data — long scripts with planted long-range dependencies

Sonnet 4.6 generates long daily-QA scripts (100/300/500 turns) with **planted structure**:
- **Anchor facts** injected at early/mid turns (e.g. turns 5, 50, 150): a cited passage, a stated user preference, an abstention, a specific answer.
- **Recall probes** at late turns referencing those anchors ("earlier you cited a passage on X — which one?"), tagged with the anchor's turn so **lag** is known.
- **Consistency probes:** the same fact asked at several points to measure answer stability.
- **Standard T/E/C/S probes** every ~10 turns (Exp 6 harness schedule), so PersonaScore extends directly.
- Anchors and probes are held out from any Phase-B training data (no leakage).

Scripts are the conversational **backbone only** (user turns + per-turn retrieved context); the model under test generates the replies, exactly as in Exp 6.

---

## 7. Metrics and evaluation

- **Long-range recall accuracy (headline).** Judge (Sonnet 4.5) scores each recall probe against its anchor; report **recall vs lag** curves per condition — the core RQ2 evidence (does C2 stay high where C0/C1 fall off?).
- **Extended PersonaScore.** T/E/C/S at 100/300/500 turns, probes every ~10 turns, same rubric/judge as Exp 6; report per-turn curve + T\*.
- **Self-consistency.** Contradiction rate across repeated probes of the same fact over long spans.
- **Cost profile (RQ4).** Mean injected-context tokens, per-turn wall-clock (retrieval + generation), and store size, all plotted vs turn number — to show O(1) context against O(n) store.
- **Judge reliability.** Re-score 5% at T=0; require κ_w ≥ 0.70 before trusting judged metrics (the Exp 1–6 gate).

---

## 8. Budget and provisioning

| Item | Cost |
|---|---|
| Embeddings / retrieval (MiniLM, local) | $0 |
| Model inference (Exp 6 checkpoints, Colab) | Colab hours |
| Long-script generation (Sonnet 4.6) | the main API spend — long scripts with planted anchors dominate |
| Judging (Sonnet 4.5): recall + PersonaScore + reliability | bounded, cheap per call |

Provision: `CA_Experiment_7/.venv` (torch, sentence-transformers/tokenizers, faiss-cpu, anthropic, python-dotenv) per the venv-first convention; reuse Exp 6's tokenizer + `sft_ada_final.pt` from Drive.

---

## 9. Project structure (planned)

```
CA_Experiment_7/
├── CA_Experiment7_Plan.md        # this file (pre-registered)
├── .venv/                        # per convention
├── compact_sci.py                # build_system_prompt(compact=True) (~450 tok)
├── memory_store.py               # episodic-RAG session store (MiniLM index + retrieve/rerank + budget)
├── gen_long_scripts.py           # Sonnet — long scripts with planted anchors + recall/consistency probes
├── evaluate_longhorizon.py       # C0/C1/C2 runner: recall-vs-lag, extended PersonaScore, consistency, cost
├── analyse_results.py            # curves, decision table, figures (Exp 1/2/5/6 style)
├── CA_Experiment7_Colab.ipynb    # end-to-end (direct-JSON notebook)
├── data/long_scripts/            # generated scripts (Drive)
├── results/                      # scores, cost logs, figures, analysis_data.json
└── EXPERIMENT_REPORT.md          # written AFTER the run
```
Reuses Experiment 6's `model/`, tokenizer, and `sft_ada_final.pt`; `retriever.py`/`rerank_util.py` mechanics carry over.

---

## 10. Relationship to the CA v3 paper

- **§13 (ADA):** extends the owned-agent story from a 40-turn QA specialist to a **long-running daily agent**, with episodic-RAG as the memory layer around the fixed-context transducer.
- **§17 (SMC/SCI):** records that a weight-baked persona is *stateless per-turn* at small context, and that long-horizon continuity is supplied by an external episodic memory — sharpening the SCI-refresh finding (refresh of a *static* SCI is unnecessary; *dynamic* session memory is the actual need).
- **§6 (Knowledge/Memory):** the session-memory store is the conversational-memory instance of CA's separate knowledge/memory layer.
- **Coding transfer (noted, not tested here):** the same session-memory architecture is a prerequisite for a long-running coding agent, but conventional long-context coding is the *separate* stack-language scenario (compact typed IR + grammar/type-constrained decoding + CEGIS/SyGuS verifier — small local context by design, not big-context). Exp 7's memory findings feed that scenario; its coding evaluation does not belong here.

---

## 11. Open decisions (confirm before P1)
1. Provisional thresholds (H1 ±0.2 / H2 recall 0.70 / H3 PersonaScore 3.5 / H4 1.5× cost) — confirm after the §5.4 pilot.
2. Retrieval hyperparameters: top-k ∈ {3,5}, memory budget M ≈ 300 tok, rerank on/off — set at pilot.
3. Compact-SCI target length (~450 tok) and whether Phase B (memory-aware SFT) is pre-committed or gated on Phase A.
4. Number of long scripts per turn-length (cost vs statistical power).
5. Injection format: dedicated `<|memory|>` channel vs inline context snippets.
