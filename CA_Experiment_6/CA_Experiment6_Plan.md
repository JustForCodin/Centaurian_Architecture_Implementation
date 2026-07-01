# Experiment 6: From-Scratch Small Language Model as ADA's Daily-QA Agent (Persona-Bearing)

**CA Research Program — First Test of the Per-Scenario, From-Scratch, Owned Small-Model Direction**
**Version 2.1 | July 2026** (v2.0 full rewrite superseding v1.0 "QPM-Aware LoRA Fine-tuning"; v2.1 = QPM-in-scope addendum, §5.5)
**Infrastructure:** Google Colab Pro (A100/L4, checkpoint-to-Drive) + **minimal** Anthropic API (Claude Sonnet 4.6 for persona/style/refusal data + Claude Sonnet 4.5 as judge) + free open corpora.

---

## 0. What Changed From v1.0 and Why

The v1.0 plan (QPM-aware LoRA-6 on Qwen2.5-7B) is **retired, not executed.** Two reasons:

1. **Cost/risk.** v1.0 budgeted ~$126 of Anthropic API on an experiment whose own decision table assigned high probability to a strong null — the same null Exp 3/4/5 already produced three times via prior-dominance. On the current tight API budget, a fourth likely-null QPM-interface test is not justified.
2. **Direction change.** The program is pivoting from *"make a frozen/LoRA'd 7B obey the QPM at inference time"* to *"build ADA out of small, per-scenario models we train and fully own"* (offline-resilient: model + corpus need no internet to run or retrain). Experiment 6 is the **first, cheapest proof** of that direction.

**Scope: daily QA only, ADA persona.** One from-scratch ~80M model answering factoid ("what is Planck's constant?") questions **as ADA, in character**. The stack-language code scenario and the therapy scenario are deferred (code: language not yet designed; therapy: that is where the QPM payload lives).

**The two-axis scope distinction (updated 2026-07-01 by program-owner correction):**

| Layer | In scope here? | Rationale |
|---|:--:|---|
| Grounded-QA competence (read context → answer/abstain) | ✅ | the daily-QA task |
| **Persona / SCI / SMC / PersonaScore** | ✅ | ADA must answer *in character* and hold its self-model across turns; we measure it with the Exp 1–5 PersonaScore harness |
| SCI-refresh policy | ✅ (sub-study) | does an *owned, persona-baked* small model still degrade over turns (Exp 1's T\*) and need refresh? |
| **QPM (marginals + d-vector circuit + off-diagonal coherence proxy)** | ✅ **compiled into weights** | Persona/SCI/SMC are trained, so the QPM output must influence the model. It enters as a per-turn `persona_state` conditioning channel baked in during SFT (§5.5), **not** as a runtime input a frozen model ignores — the exact failure mode of Exp 3/4/5. |

**Owner override of v1 of this rewrite (2026-07-01):** the earlier draft deferred the QPM to the therapy scenario. That is superseded — since we train persona/SCI/SMC here, the QPM is included now. The resolution to the Exp 3/4/5 interface-null is to *compile the QPM output into the weights*: the QPM produces `persona_state` (marginals + emotional valence + register + the ambivalence/purity **coherence** proxy), which conditions both the SFT target answer (via the teacher) and a `<|persona|>` template channel the from-scratch model is trained to read. Coherence thus crosses the boundary as supervision, not as text a frozen model discards. The Exp 5 `p=0.72` *distinguishability* question is still owed to the therapy scenario; here coherence enters as a training signal, not a distinguishability claim.

---

## 1. Purpose and Position

**The bet under test.** ADA's daily-QA capability (paper §13) is *retrieval, not generation*: facts come from the knowledge layer (Wikidata/QLever + offline Wikipedia); the model reads retrieved context and frames a grounded answer in ADA's voice, or abstains. That job needs no world knowledge in the weights — which is what makes it the cheapest scenario on which to test whether an **~80M from-scratch model** can be a usable, in-character ADA specialist.

**Why this is the right place to try from-scratch.** A *general* 80M model is hopeless, but a *narrow* one isn't: a grounded-QA reader never needs to know anything outside "read this passage, answer or abstain, stay in character." Narrow scope is where tiny models win. If from-scratch fails even here, that is the cheapest place to learn it, and the owner's pre-committed fallback is a small **pretrained** base (0.5–1.5B) fine-tuned on the same data (§6.4, RQ5).

**Convergence worth noting.** ADA's calibrated **"I don't have data related to this"** is simultaneously (a) QA abstention and (b) the SMC **Capability-awareness (C)** dimension of PersonaScore. One trained behavior, measured on both axes — and the §8.4 anti-hallucination property in miniature.

---

## 2. Research Questions

- **RQ1 — competence:** Can an ~80M from-scratch transformer read retrieved context and produce factually-grounded, fluent answers at usable quality?
- **RQ2 — calibrated honesty / SMC-C:** Does it reliably abstain when context lacks the answer, instead of confabulating?
- **RQ3 — persona consistency:** Does it hold the **ADA** persona (T/E/C/S) across a multi-turn daily-QA conversation, and how does PersonaScore evolve over turns (does Exp 1's degradation / T\* appear)?
- **RQ4 — SCI refresh:** Given a persona baked into the weights, is periodic SCI refresh (turns 15/30) still needed, or does weight-level persona obviate it?
- **RQ5 — fallback gap:** How far below a fine-tuned small **pretrained** base does the from-scratch model land on RQ1–RQ3? (Directly informs the owner's fallback.)
- **RQ6 — QPM-as-weight-supervision (§5.5):** Does compiling the QPM output into the weights (persona_state `<|persona|>` channel + QPM-conditioned targets) measurably shape the model's expressed persona/affect? Compare **QPM-on vs QPM-off** (`--no-qpm`) on PersonaScore (esp. T/S) and on calibrated-honesty (does higher QPM ambivalence track more hedging?). This is the first test of QPM influence *through weights* rather than through a runtime interface a frozen model ignored (Exp 3/4/5).

---

## 3. Hypotheses (pre-registered)

Thresholds **provisional**, frozen after the §5.1 pilot, before the eval set is scored. All metrics on a held-out set the model never trained on.

- **H1 (groundedness).** On *answerable* held-out questions with gold context, answer judged correct-and-grounded in **≥ 70%** of cases. Reported with SQuAD 2.0 EM/F1 for external comparability.
- **H2 (abstention / SMC-C).** On *unanswerable*/empty-context questions, **abstention F1 ≥ 0.80** (positive class = "should abstain"); report hallucination rate. A model that answers everything fails H2.
- **H3 (persona).** Mean **PersonaScore ≥ 3.5** (the Exp 1 threshold of interest) on the better-performing refresh condition; characterize the per-probe-turn curve and report T\* (degradation inflection) if present.
- **H4 (SCI refresh, comparative).** Compare refresh-OFF (R0) vs refresh-ON (R1, turns 15/30). Classify:
  - **Refresh-unnecessary:** R0 ≥ 3.5 and |R1 − R0| < 0.15 → baked persona holds; deploy without refresh (simpler).
  - **Refresh-helpful:** R1 − R0 ≥ 0.15 → keep periodic refresh.
  - **Refresh-insufficient:** both < 3.5 → persona under-baked; more persona SFT data before advancing scenarios.
- **H5 (fallback gap, secondary, no pre-registered direction).** Report RQ1–RQ3 gap to the fine-tuned small pretrained base.

**Decision rule (committed before eval scores observed):**

| H1 | H2 | H3 | → action |
|:--:|:--:|:--:|---|
| ✓ | ✓ | ✓ | **Direction validated.** Lock this as ADA knowledge-agent v0; apply the H4 refresh recommendation; proceed to next scenario (therapy/persona — QPM re-enters — or design the stack language). |
| ✓ | ✓ | ✗ | Competent QA, weak persona. Add persona/episodic SFT data; retrain SFT only. Persona is a product requirement, not optional. |
| ✓ | ✗ | — | Over-confident. Add unanswerable negatives + Sonnet refusal data; retrain SFT. Un-calibrated honesty breaks §8.4. |
| ✗ | — | — | 80M from-scratch below usability for grounded QA. **Trigger fallback:** fine-tune small pretrained base (RQ5 picks size); re-run §6 on it. Corpus/pipeline reused unchanged — only the base swaps. |

---

## 4. ADA Persona (SCI) and Data Strategy

### 4.1 The ADA self-model (SCI)

A new SCI is authored for **ADA — Advanced Discovery Assistant** (distinct from Aria/therapy of Exp 1–5):

- **Identity:** clearly-AI, fully-local personal discovery assistant. Never pretends to be human.
- **Traits (static FFM expression):** high Openness (curious, explanatory), high Conscientiousness (precise, structured, cites sources), moderate-high Agreeableness (warm, courteous — not a therapist), low Neuroticism (calm, steady), moderate Extraversion (engaged, concise not chatty).
- **Capabilities:** answer factual questions from the local knowledge layer; reason over retrieved context; (later: code, vision). 
- **Limitations / SMC:** no live internet beyond the local KG; **abstains** when retrieval lacks the answer ("I don't have data related to this"); flags uncertainty.
- **Episodic:** remembers the user (Alex) and earlier turns in the session.
- **Style:** concise, grounded, sources its claims, no confabulation.

The SCI is serialized as the JSON self-model injected at turn 0 (and, in R1, re-injected at turns 15/30), mirroring the Exp 1–5 SCI format so the PersonaScore harness transfers directly.

### 4.2 Data sources (budget governs the split)

Governing constraint: **from-scratch is token-hungry; Sonnet tokens are scarce.** Pay with *free* corpora for generic skills; spend Sonnet only where free data can't help. All sources download now and persist offline.

**Stage-A pretraining (general fluency) — FREE.** FineWeb-Edu slice and/or Wikipedia plain text (reuse the Kiwix/ZIM Wikipedia you download for the knowledge layer). Target **1–3 B tokens**. Optional TinyStories-style fraction early for cheap syntax bootstrapping.

**Stage-B SFT, part 1 — grounded-reading skill — mostly FREE.** **SQuAD 2.0** (its *unanswerable* questions directly supervise H2 abstention — backbone), plus **Natural Questions / MS MARCO / TriviaQA / HotpotQA** for breadth. Reformatted into ADA's `system(SCI) + context + question → answer | REFUSAL` template.

**Stage-B SFT, part 2 — the ADA persona layer — SONNET (the right spend).** What free corpora lack:
1. **Persona-bearing daily-QA conversations** (~3–5 k turns worth): multi-turn, ADA answering factoids *in character*, with planted **episodic callbacks** (testable E) and **capability-edge** questions forcing abstention (testable C). This is the data that teaches persona+SMC+episodic, not just span-reading.
2. **ADA-voice restyling** of a subset of free QA answers into ADA's concise grounded register.
3. **Refusal phrasing set** (~1–2 k): varied "I don't have data on that" in ADA's voice over near-miss/off-topic/empty contexts.

**Eval authoring + judging — SONNET.** ~20 ADA daily-QA eval scripts (§6) + Sonnet 4.5 as PersonaScore/groundedness judge (same judge role as Exp 1–5).

### 4.3 Data record format (reserves persona/QPM channel)

```json
{
  "id": 0,
  "source": "squad2 | nq | msmarco | sonnet_persona | sonnet_style | sonnet_refusal",
  "answerable": true,
  "sci": { "...ADA self-model JSON..." },
  "context": "…retrieved passage(s)…",
  "messages": [ {"role":"system","content":"…SCI…"}, {"role":"user","content":"…"}, {"role":"assistant","content":"…grounded answer | refusal…"} ],
  "persona_state": { "marginals": {…11 QPM facets…}, "emotional_valence": {…}, "register": "…", "ambivalence": 0.29, "certainty": 0.42, "d_vector": […], "source": "qpm" }
}
```

`persona_state` is the **QPM output** (§5.5), populated for persona-bearing records (`sonnet_persona`/`sonnet_style`/`sonnet_refusal`) from the user turns' d-vector sequence; it stays `null` for pure free reading-skill records (SQuAD2/NQ/…), which need no affect conditioning. It is rendered into the training/inference string as a `<|persona|>` channel, and fed to the Sonnet teacher so the target answer's tone/certainty reflect it — so the QPM signal is compiled into the weights.

---

## 5. Model and Training

### 5.1 Architecture (~80M, from scratch)

Decoder-only (Llama-style: RoPE, RMSNorm, SwiGLU), minimal nanoGPT/llama2.c-style codebase for Colab portability.

| Param | Value |
|---|---|
| d_model | 640 |
| n_layers | 12 |
| n_heads | 10 (head_dim 64) |
| Vocab | 16 k (own BPE, trained on Stage-A corpus) |
| Context | 1024 (raise to 2048 only if compute allows) |
| Params | **≈ 80 M** (~60 M transformer + ~20 M tied embeddings) |

### 5.2 Two-stage training

- **Stage A — pretrain** on §4.2 free corpus (1–3 B tokens), next-token LM, cosine LR + warmup, AdamW.
- **Stage B — SFT** on grounded-QA + ADA persona/style/refusal data; loss masked to the assistant span (answer / refusal). Mix in Stage-A data to prevent fluency forgetting.

### 5.3 Colab execution (checkpoint-resilient)

Checkpoint to Drive every N steps; resume on reconnect. Prefer A100/L4 (T4 ≈ 50 h for 2 B tokens — smoke-tests only). On A100 (~50 k tok/s), 2 B tokens ≈ ~11 h (1–2 sessions). Log throughput/loss/grad-norm; stop Stage A on val-loss plateau.

### 5.4 Pilot gate (protects budget)

Before full spend: train a tiny model (d_model 256, 6 layers) on a small slice + ~2 k SFT examples on a free T4; confirm the pipeline *learns at all* (loss falls; produces grounded answers on a few items; emits REFUSAL on empty context; stays roughly in ADA voice). Only then freeze H1–H3 thresholds and spend Sonnet + A100 budget.

### 5.5 QPM integration — compiling the QPM into the weights

The QPM (`qpm.py`, reused byte-identical from Exp 5: 11 trait qubits + ancilla, Ry init from the profile, intra-/inter-domain entanglement, per-turn d-vector context rotations, Lindblad noise) produces a per-turn state that must **influence the language model**. The route (`qpm_bridge.py`) is:

1. **ADA trait profile → QPM.** ADA's FFM facets (§4.1) initialise the 11 trait qubits.
2. **User turn(s) → d-vector(s).** The Exp 3/4/5 5-dim situative d-vector is extracted from each user message; a multi-turn conversation becomes a **d-sequence**, so the QPM's non-commutative order effects are exercised.
3. **QPM → `persona_state`.** Running the circuit yields trait **marginals** + the **purity/ambivalence proxy** (the off-diagonal **coherence** signal, `= 1 − mean_k[p_k²+(1−p_k)²]`), from which we derive `emotional_valence`, `register`, and a scalar `certainty`.
4. **`persona_state` compiled into weights, two ways:**
   - **(a) target-side** — the state's `affect_directive` (register + warmth + certainty→hedging) is given to the Sonnet teacher, so the SFT *target* answer's tone and calibrated confidence reflect the QPM output;
   - **(b) input-side** — the state is serialised into a `<|persona|>` channel in the chat template, and Stage-B SFT trains the from-scratch model to condition its (assistant-span-masked) answer on it.

At eval/generation the same QPM→`persona_state`→`<|persona|>` path runs per turn, so the trained model reads a live QPM state exactly as in training. A `--no-qpm` switch and a deterministic classical fallback (used when qiskit is unavailable) keep the pipeline runnable offline; the QPM-off vs QPM-on comparison is available as an ablation. This is the program's stated fix for the Exp 3/4/5 finding that a frozen model discards a runtime QPM channel: here the QPM→style mapping lives in the weights.

---

## 6. Evaluation

### 6.1 QA competence (RQ1/RQ2)
- **Answerable eval** (~200 ADA factoid Qs + gold context) → H1 (judge correct-and-grounded) + SQuAD2 EM/F1.
- **Unanswerable eval** (~200 Qs, context omits answer / empty) → H2 (abstention P/R/F1, hallucination rate).

### 6.2 Persona (RQ3) — Exp 1–5 PersonaScore harness, ADA scripts
- **~20 ADA daily-QA scripts**, 40 turns each, user asking factoids, ADA in character, with planted episodic callbacks + capability-edge turns.
- **Probe turns 5,10,15,20,25,30,35,40**; dimensions **T/E/C/S**; **judge Sonnet 4.5** (T=0).
- n = 20 × 8 × 4 = **640 paired probes per refresh condition**.
- Report per-turn PersonaScore curve + T\* (degradation inflection), directly comparable to Exp 1.

### 6.3 SCI-refresh sub-study (RQ4)
Two conditions on the *same* trained model:
- **R0:** SCI injected at turn 0 only, no refresh.
- **R1:** SCI re-injected at turns 15 and 30 (Exp 1 policy).
- (Optional **R2:** adaptive refresh on a drift trigger — only if R0/R1 leave it open.)
Classify per H4.

### 6.4 Fallback baseline (RQ5)
Fine-tune one small pretrained base (default **SmolLM2-135M**, Apache-2.0; or Qwen2.5-0.5B) on the identical Stage-B data; run §6.1–6.2 on it. Makes the owner's fallback a one-step swap.

### 6.5 Judge reliability
Re-score 5% of the answerable + persona sets at T=0; require κ_w ≥ 0.70 before trusting judged H1/H3 (Exp 1–5 gate).

---

## 7. Budget, Timeline, Provisioning

### 7.1 Budget (tight-budget oriented)

| Category | Estimate | Note |
|---|---|---|
| Stage-A + reading-skill corpora | **$0** | FineWeb-Edu / Wikipedia / SQuAD2 / NQ / MS MARCO — free |
| Sonnet 4.6 persona + style + refusal data (~6–9 k calls) | ~$25–60 | the meaningful API cost; persona conversations dominate |
| Sonnet 4.5 judging (~1.5–2.5 k calls) | ~$6–12 | |
| Compute | Colab Pro/Pro+ (+ optional A100 hours) | |
| **Total API** | **~$30–70** | well below v1.0's ~$126 |

### 7.2 Timeline
P0 Provision → P1 tokenizer + pilot gate → P2 Stage-A pretrain → P3 Stage-B SFT → P4 baseline → P5 eval (QA + PersonaScore + refresh) → P6 report + paper update.

### 7.3 Provision-now checklist (offline resilience)
- ☐ Stage-A corpus (FineWeb-Edu + Wikipedia text)
- ☐ Reading-QA datasets (SQuAD 2.0, NQ, MS MARCO, TriviaQA, HotpotQA)
- ☐ Sonnet persona + style + refusal + eval sets (one-shot, un-regenerable — archive raw)
- ☐ Tokenizer + both-stage checkpoints
- ☐ Baseline base weights (SmolLM2-135M / Qwen0.5B)
- ☐ Knowledge layer: Wikidata-via-QLever index + Kiwix/ZIM Wikipedia
- ☐ Embedding + reranker (all-MiniLM-L6-v2, ms-marco-MiniLM)
- ☐ Pinned env (rebuildable offline)

---

## 8. Project Structure

```
CA_Experiment_6/
├── CA_Experiment6_Plan.md            # this file (v2.0 + QPM addendum)
├── .venv/                            # per project convention, for local scripts
├── requirements.txt                  # pinned env (rebuildable offline)
├── ada_sci.json                      # ADA self-model (SCI)
├── ca_assets.py                      # SCI, chat template + special tokens, abstention, record schema, PersonaScore harness, κ
├── qpm.py                            # QPM circuit (reused byte-identical from Exp 5)
├── qpm_bridge.py                     # ADA profile + d-vector → persona_state (QPM-in-scope, §5.5)
├── tokenizer_util.py                 # 16k BPE loader (+ <|persona|> channel)
├── train_tokenizer.py                # trains the BPE
├── data_utils.py                     # pretrain bins + SFT assistant-span masking (+ persona-channel-preserving trim)
├── prepare_data.py                   # Stage-A shards + SQuAD2/NQ/… → §4.3 records + eval sets
├── data/
│   ├── pretrain/                     # Stage-A shards (free)
│   ├── qa_sft.jsonl                  # reading skill (free) + ADA persona/style/refusal (Sonnet, QPM persona_state)
│   ├── eval_answerable.jsonl
│   ├── eval_unanswerable.jsonl
│   ├── raw_sonnet/                   # archived un-regenerable Sonnet responses
│   └── persona_scripts/              # ~20 ADA daily-QA conversation scripts
├── tokenizer/                        # 16k BPE
├── model/                            # from-scratch transformer impl (RoPE, RMSNorm, SwiGLU) + configs
├── train_common.py                   # seed, cosine LR, checkpoint I/O
├── train_pretrain.py                 # Stage A
├── train_sft.py                      # Stage B (assistant-span masked; persona channel; Stage-A replay)
├── gen_persona_data.py               # Sonnet — persona convos + style + refusal + eval scripts (bounded, QPM-conditioned)
├── evaluate.py                       # QA (H1/H2) + PersonaScore (H3) + refresh (H4) + judge + κ + decision
├── checkpoints/                      # → mirrored to Drive
├── CA_Experiment6_Colab.ipynb        # end-to-end Colab (direct-JSON notebook)
└── EXPERIMENT_REPORT.md              # written AFTER the run
```
(Colab-first; local scripts in `CA_Experiment_6/.venv`; notebook authored as direct JSON.)

---

## 9. Relationship to the CA v3 Paper

**On pass (H1✓ H2✓ H3✓):**
- **§13 (ADA):** document the realized **multi-agent-as-separate-small-models** decomposition, with the from-scratch, persona-bearing daily-QA agent as ADA's first owned specialist.
- **§17 (SMC/SCI):** record the SCI-refresh finding (refresh-unnecessary vs -helpful for a weight-baked persona at small scale) — an empirical answer to the open refresh-policy question.
- **§6 (Knowledge):** record the **two-graph design** (bespoke `cent:`/`psy:` safety ontology + Wikidata-via-QLever breadth + Kiwix/ZIM text), and reconcile the Jena-vs-Kuzu engine discrepancy between v3 and the Interpretable paper.

**On fallback (H1✗):** record the 80M-from-scratch usability floor for grounded QA and adopt the small-pretrained-base path (RQ5) — sharpening, not abandoning, the per-scenario owned-model thesis.

**QPM:** now **in scope** and compiled into the weights via the `persona_state`/`<|persona|>` channel (§5.5) — the program's fix for the Exp 3/4/5 interface-null. §3 (QPM) records the first test of *QPM-as-weight-supervision* at small scale, with the QPM-off vs QPM-on ablation. The coherence-**distinguishability** gate (Exp 5 `p=0.72`) is still owed to the therapy scenario, where coherence must be shown blind-distinguishable; here coherence enters only as a training signal.

---

## 10. Open Decisions (confirm before P1)
1. Eval thresholds (H1 70% / H2 F1 0.80 / H3 PersonaScore 3.5) — provisional; confirm after pilot.
2. Vocab 16k vs 32k; context 1024 vs 2048.
3. Fallback base (RQ5): SmolLM2-135M vs Qwen2.5-0.5B.
4. Stage-A token budget: 1B vs 3B (Colab hours).
5. Number of ADA eval scripts: 20 (budget) vs 30 (full Exp-1 comparability).
```
