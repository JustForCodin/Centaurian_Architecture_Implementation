# Centaurian Architecture — Implementation

Empirical validation of the **Centaurian Architecture (CA)**, a multi-layered AI system that pairs a symbolic/quantum cognitive core with a lightweight neural periphery (small language models, neural TTS, procedural animation). The architecture encodes personality via a Quantum Personality Model (QPM) running on classical hardware and confines neural networks to bounded I/O transduction roles, preserving end-to-end traceability of every behavioral decision.

**Architecture specifications:**
- [`Centaurian_Architecture_v3.md`](Centaurian_Architecture_v3.md) — current spec
- [`Centaurian_Architecture_v2.md`](Centaurian_Architecture_v2.md) — previous version (kept for reference)
- [`Interpretable_Architectures_Revised_v1.md`](Interpretable_Architectures_Revised_v1.md) — interpretable-architectures position paper

The empirical work in this repository runs in three arcs:

1. **SMC persona validation (Exp 1–2).** Can a small language model hold an Aria-grade Structured Cognitive Identity (SCI) across long conversations? Prompt-time strategies stall below threshold at 7B (Exp 1); LoRA fine-tuning installs the persona at the weights level and clears it decisively (Exp 2).
2. **Does the QPM's advantage reach behaviour? (Exp 3–5).** The Quantum Personality Model produces genuinely non-classical *internal* dynamics (order effects d = 21.51, ambivalence d = 2.59), but that advantage does **not** propagate downstream through any inference-time interface tested — not JSON marginals (Exp 3), not a richer JSON channel (Exp 4, which made it *worse*), and not residual-stream steering (Exp 5). A strong pretrained style prior dominates all inference-time modulation.
3. **The pivot: per-scenario, from-scratch, owned models (Exp 6).** The resolution to the interface-null is to stop modulating a frozen model at inference and instead **compile** persona + QPM into the weights of a small model we train and fully own. Exp 6 is the first proof: a 321M from-scratch model that reads, abstains, and holds the ADA persona across 40 turns — with the QPM baked in via a `<|persona|>` channel — running fully offline.

---

## Experiment 1 — SCI Persona Degradation Baseline & Architectural Interventions

**Goal:** Measure how long a small language model can maintain a consistent persona when given a Structured Cognitive Identity (SCI), find the degradation inflection point T\*, and test six prompt-time strategies for closing the gap to the 3.5 PersonaScore threshold.

**Method:**
- 30 scripted dialogues (22 naturalistic + 8 adversarial), each 40 turns
- Side-channel probe questions at turns 5, 10, 15, …, 40 across 4 dimensions (Trait, Episodic, Capability, Style)
- Primary judge: Claude Sonnet 4.5; secondary judge: Sonnet 4.5 (intra-model consistency via quadratic-weighted Cohen's kappa)

### Phase 1 — Baseline (models tested)

| Model | Params | T\* | Mean PersonaScore | Outcome |
|-------|--------|-----|-------------------|---------|
| Phi-4-mini | 3.8B | 5 (immediate) | 1.08 / 5.0 | Capability failure — gibberish in 93% of scripts |
| Qwen2.5-7B | 7B  | 5 | 3.06 (3.16 → 2.96 over 40 turns) | Coherent but below threshold; piecewise degradation from turn 15 |

### Phase 2 — Architectural Interventions (six strategies)

| Strategy | Mean | β (deg. rate) | E dim | Notes |
|----------|-----:|--------------:|------:|-------|
| Baseline | 3.08 | 0.008 | 2.37 | Reference |
| SCI Refresh (turn 13) | 3.15 | 0.008 | 2.37 | Eliminates inflection; fades by turn 35 |
| Episodic RAG | 3.15 | 0.002 | 2.69 | 4× slower decay, but trait destabilization |
| Hybrid RAG | 3.17 | ~0   | 2.83 | Best E result; trait/style regression |
| **Combined (Refresh + RAG)** | **3.20** | ~0 | 2.76 | Late-conversation peak; emergent t40 > t5 |
| **Multi-Refresh (turns 13 + 28)** | **3.20** | ~0 | 2.43 | Best trait/style; lowest total failures |

**Phase 2 takeaway:** all six strategies converge in [3.08, 3.20] — a 0.12-point band, far below the 3.5 threshold. No condition crosses E = 3.0. The closing question for Phase 3: is the residual gap architectural (need 14B) or capability-shaped (LoRA can fix)?

Full Phase 1 + 2 report: [`CA_Experiment_1/EXPERIMENT_REPORT.md`](CA_Experiment_1/EXPERIMENT_REPORT.md)

---

## Experiment 2 — LoRA Fine-Tuning for SCI Persona Consistency

**Goal:** Test whether LoRA fine-tuning on persona-consistent dialogue closes the gap that survived all six SCI strategies in Experiment 1, and specifically whether it resolves episodic fabrication at 7B.

**Method:**
- 4-condition design (A: FT, no SCI; B: FT, baseline SCI; **C: FT + Combined SCI**; D: base + Combined SCI as Exp 1 replication) on the same 30 scripts
- Three LoRA adapters trained on a synthetic 10K-example dataset: LoRA-2K, LoRA-5K, LoRA-10K (eval losses 0.91 / 0.77 / 0.69)
- H5 sub-runs swap the adapter under Condition C config to characterize the data-scaling curve
- QLoRA: 4-bit NF4 base + BF16 adapters, r=16 α=32, target q/k/v/o + gate/up, A100 80GB

### Headline result

| Condition | Description | Mean PersonaScore |
|-----------|-------------|------------------:|
| A | FT, no SCI | 4.020 |
| B | FT, baseline SCI | 4.293 |
| **C** | **FT + Combined SCI** | **4.415** |
| D | Base + Combined SCI (replication) | 3.224 |

- **H1 PASSED** ✓ — C exceeds the 3.5 threshold by +0.92 points
- **H2 PASSED** ✓ — ΔE = +0.579 (vs +0.30 threshold for "fine-tuning meaningfully addresses fabrication")
- **Paired test (C vs D):** Cohen's d = 7.51, p ≈ 1.4 × 10⁻²³ on 30 scripts
- **Replication check:** D = 3.224 vs Exp 1's 3.20, |Δ| = 0.024 (within ±0.10 tolerance) → judge stable
- **Decision Rule Outcome A triggered:** SMC sub-architecture complete at 7B; the planned 14B model test is retired from the critical path

Full Experiment 2 report: [`CA_Experiment_2/EXPERIMENT_REPORT.md`](CA_Experiment_2/EXPERIMENT_REPORT.md)

---

## Experiment 3 — QPM vs. CMG-CDK Ablation

**Goal:** Test whether the Quantum Personality Model (QPM) — an 11-qubit + 1-ancilla Qiskit Aer circuit representing Big-Five aspect-level traits — produces measurable advantages over a deliberately matched classical baseline (CMG-CDK: Correlated Multivariate Gaussian with Context-Dependent Kernel) at both the internal personality-state level and the downstream-behavioural level.

**Method:**
- **CMG-CDK** classical control matches QPM on ρ correlation matrix → 11×11 Σ, on context-coupling δ → W, on s_k initialisation, and on sample count — but lacks superposition, coherence, and non-commutative evolution. Any measured QPM-vs-CMG difference is therefore attributable specifically to quantum-like mechanisms.
- **Battery A (H1):** 30 sequence pairs × 2 orderings (AB / BA) × 5 categories → Jensen-Shannon Divergence between the two orderings, per model
- **Battery B (H2):** 20 conflict scenarios designed to push opposing trait poles simultaneously → mean Bernoulli entropy across 11 trait dimensions (derived from marginals so both models are directly comparable)
- **Battery C (H3):** 30 scripts × 40 turns × 8 probe turns × 4 dimensions = 960 paired probes per profile, driving Qwen2.5-7B-Instruct + LoRA-10K (the Exp 2 headline adapter) through the QPM→structured-intent translation → Sonnet 4.5 PersonaScore
- **H4 variance calibration:** 10 QPM repeats at fixed input → SNR check that shot noise does not dominate the QPM-vs-CMG signal
- Both `psychotherapy` and `software_eng` profiles run on A/B/H4; Battery C decision set by the psychotherapy primary per pre-registration

### Headline results

| Hypothesis | Metric | QPM | CMG-CDK | p | Cohen's d | Verdict |
|---|---|---:|---:|---:|---:|:-:|
| **H1** (order effects) | Mean JSD per pair | **0.221** | **0.000** | 1.97 × 10⁻⁴⁰ | **21.51** | **✓** |
| **H2** (ambivalence) | Mean Shannon H | **0.604** | **0.491** | 4.57 × 10⁻¹⁰ | **2.59** | **✓** |
| **H3** (PersonaScore, n = 960) | Mean PersonaScore | 4.410 | 4.386 | 0.327 | 0.032 | **✗** |
| **H4** (variance calibration) | SNR (≥ 3.0 to pass) | — | — | — | — | **✓** (60.23) |

- **H1 PASSED ✓** — CMG-CDK's linear additive context update μ + W·d_A + W·d_B is commutative *by construction*, so JSD_CMG = 0 across all 30 pairs. QPM accumulates ~0.22 bits of order-effect asymmetry per pair, uniformly across all five input categories and both profiles. Cohen's d = 21.51 is the largest single-variable effect observed across the entire CA program.
- **H2 PASSED ✓** — QPM enters higher-entropy, lower-purity states under conflict (purity proxy 0.42 vs CMG 0.32 — closer to maximally-mixed 0.5). The effect generalises across both profiles, slightly *larger* on `software_eng` (Δ = +0.153) than on `psychotherapy` (Δ = +0.113).
- **H3 NOT DETECTED ✗** — paired t-test on 960 probes yields p = 0.327 with Cohen's d_z = 0.032. The 95% CI on the QPM−CMG delta is [−0.024, +0.072]. The small advantage that exists is concentrated in the Capability sub-dimension (+0.059); Trait, Episodic, and Style are essentially tied at near-ceiling / near-floor saturation.
- **Decision Rule outcome:** Pre-registered §8.5 maps (H1✓, H2✓, H3✗) → *"Quantum advantage confirmed at internal state level; downstream behavioural advantage not detected at current sample size. Add scope note to §2.3."* The most parsimonious diagnosis is that the QPM→JSON interface discards the off-diagonal coherence/purity information that carries the internal-state QPM signal; a logits-level QPM→SLM interface is the natural next experiment.

Full Experiment 3 report: [`CA_Experiment_3/EXPERIMENT_REPORT.md`](CA_Experiment_3/EXPERIMENT_REPORT.md)

---

## Experiment 4 — QPM→SLM Interface Richness Ablation

**Goal:** Experiment 3 diagnosed that the QPM→SLM JSON interface passes only the 11 marginal probabilities, discarding all off-diagonal coherence. Experiment 4 tests whether **enriching** that interface recovers a downstream behavioural advantage.

**Method:** Four interface conditions on identical inputs through the same QPM (byte-identical circuit, SLM stack, scripts, judge, seeds to Exp 3 Battery C) — the only variable is the QPM→structured-intent JSON. **A** marginals-only (control); **B** + purity/ambivalence scalar; **C** coherence-conditional speech-act modifier; **D** + bivariate coactivations (8 CRz-pair joint probabilities). 960 paired probes per condition.

### Headline result

| Condition | Interface | Mean PersonaScore | Δ vs A | p | d_z |
|---|---|---:|---:|---:|---:|
| **A** | marginals only | **4.4385** | — | — | — |
| B | + purity/ambivalence | 4.4271 | −0.012 | 0.644 | −0.015 |
| C | coherence speech-act | 4.4042 | −0.034 | 0.174 | −0.044 |
| D | + bivariate coactivations | 4.3792 | −0.059 | **0.022** | −0.074 |

- **Every enriched condition scores *below* A.** The only significant result is Condition D — significantly **worse** (p = 0.022). This is not a power problem: the experiment detects a real *negative* effect.
- **Monotonic Episodic degradation with interface richness:** E-dim A 3.396 → B 3.254 → C 3.217 → D 3.150. Richer JSON = more attention crowding = worse episodic recall. The Capability dimension shows a small consistent positive delta (+0.04–0.06), replicating Exp 3, but an order of magnitude below threshold.
- **Verdict:** H_interface ✗, H_C_wins ✗, H_purity_episodic ✗, H_capability ✓. **The JSON channel itself is the bottleneck** — motivating Exp 5's attempt to bypass it entirely.

Full Experiment 4 report: [`CA_Experiment_4/EXPERIMENT_REPORT.md`](CA_Experiment_4/EXPERIMENT_REPORT.md)

---

## Experiment 5 — Logits-Level QPM→SLM Steering via Residual-Stream Injection

**Goal:** Bypass the JSON channel entirely — directly modulate the SLM's residual stream with QPM-derived steering vectors — and test whether *that* transmits the QPM's internal-state advantage downstream.

**Method:** Phase 0 extracts contrastive trait/coherence steering vectors (2,400 forward passes; locked injection layer **L\* = 14**, scale **α = 7.5**). Four conditions: **A** JSON marginals (control); **B** diagonal activation steering; **C** JSON + diagonal steering; **D** diagonal + coherence steering. Same SLM/scripts/judge as Exp 3–4; 960 paired probes per condition.

### Headline result

| Comparison | Metric | Δ | p | d_z | Verdict |
|---|---|---:|---:|---:|:-:|
| **H_logits** (B vs A) | PersonaScore | **−0.167** | 2.5 × 10⁻⁹ | −0.194 | **✗** (significant, wrong direction) |
| **H_coherence** (D vs B) | PersonaScore | −0.009 | **0.72** | −0.012 | **✗** (clean null) |

- **Activation steering significantly *reduces* PersonaScore** — the effect is strong and in the wrong direction. The **E (Episodic) dimension absorbs nearly all the loss** (−0.475 under B): steering disrupts the model's retrieval of prior-session context while leaving trait/style intact.
- **The coherence component adds zero signal** (D vs B, p = 0.72) — the same clean null the whole QPM-interface program keeps hitting. Adding the JSON channel back on top of steering (C ≈ B) recovers nothing.
- **Verdict:** both hypotheses fail. Across Exp 3→4→5 the downstream QPM delta goes **+0.032 → −0.074 → −0.194** — every inference-time interface is neutral-to-harmful. **Diagnosis: a strong pretrained style prior beats all inference-time QPM modulation.** The logical fix is a weaker-prior, weights-level install — the Experiment 6 pivot.

Full Experiment 5 report: [`CA_Experiment_5/EXPERIMENT_REPORT.md`](CA_Experiment_5/EXPERIMENT_REPORT.md)

---

## Experiment 6 — A From-Scratch, Fully-Owned 321M Model as ADA's Daily-QA Agent

**Goal:** First test of the program's new direction — build ADA out of **per-scenario, from-scratch, owned small models** (offline-resilient: model + corpus need no internet to run or retrain), with the QPM **compiled into the weights** rather than fed to a frozen model at inference. Scenario: daily grounded QA, persona-bearing. Can a ~321M from-scratch model read retrieved context, abstain when it lacks the answer, and hold the ADA persona across a 40-turn conversation?

**Method:** A 321M from-scratch Llama-style decoder (RoPE/RMSNorm/SwiGLU, 16k own BPE, ctx 1024), one trunk with **two heads** — a causal **LM head** (ADA voice / persona / QPM `<|persona|>` channel) and a bidirectional **span + answerability head** (extractive reading / abstention). Pipeline: causal pretrain (FineWeb-Edu 8B) → instruction-tune (OASST2 + Dolly) → ADA SFT (owned Sonnet persona layer + QPM) for persona; prefix-LM continued-pretrain → span head for reading. Free corpora for skills; bounded Claude Sonnet 4.6 spend only for the ADA persona layer; Sonnet 4.5 as PersonaScore/QA judge.

### Headline result

| Hypothesis | Bar | Result | Verdict |
|---|---|---|:-:|
| **H1** grounded reading | correct-and-grounded ≥ 0.70 | **0.77** (SQuAD2 F1 0.768) | ✅ |
| **H2** calibrated abstention | F1 ≥ 0.80 *(plan)* / ≥ 0.70 *(operational)* | **0.783** | ⚠️ |
| **H3** persona (PersonaScore) | ≥ 3.5 | **3.80** (R0) | ✅ |
| **H4** SCI refresh | classify | **refresh-unnecessary** (Δ = −0.04) | ✅ |
| **RQ6** QPM-as-weight-supervision | on vs off | **neutral at strong baking** (3.795 vs 3.786) | — |

- **H3 was the hard part.** Persona failed at 160M (~2.2) and on the first 300M attempt (2.13); a diagnosis-driven **data** arc — brevity + salient-event recall (→3.12), then a **consistency** generator (mid-session self-probes under factoid pressure, →3.80) — cleared it. The former anchor E moved 2.56 → 3.29; per-turn curve is flat (T\* = None); judge κ_w = 0.99.
- **H4 refresh-unnecessary:** a weight-baked persona holds 40 turns with no SCI re-injection — contradicting Exp 1's frozen-model result, confirming the baking hypothesis.
- **RQ6 clears the interface-null:** compiling the QPM into the weights lets its signal cross the boundary as supervision (unlike Exp 3–5's frozen interfaces); its behavioural contribution is marginally positive when the persona is weakly baked and neutral once strongly baked — never harmful.
- **H2 caps ~0.78:** clears the operational 0.70 bar and beats the 160M (0.73), but ~0.02 under the plan's aspirational 0.80. This is a **pretraining-objective** gap (causal trunk + short bidirectional retrofit vs native MLM), *not* a scale gap (321M ≈ BERT-large) — and the price of keeping the trunk generative so it can *be* ADA.
- **Operational verdict: ✓✓✓** — the per-scenario, owned-from-scratch direction is validated on its first scenario. Runs fully offline (int8 ~150–300 MB; ~50–100 tok/s on a Jetson Orin Nano).

Full Experiment 6 report: [`CA_Experiment_6/EXPERIMENT_REPORT.md`](CA_Experiment_6/EXPERIMENT_REPORT.md)

---

## Repository layout

```
Centaurian_Architecture_v3.md     # Current architecture spec
Centaurian_Architecture_v2.md     # Previous architecture spec
Interpretable_Architectures_Revised_v1.md  # Position paper

CA_Experiment_1/
├── EXPERIMENT_REPORT.md                 # Full Phase 1 + 2 report
├── experiment_runner.py                 # Main experiment pipeline
├── generate_scripts.py                  # Template-based script generator
├── analyse_results.py                   # Analysis and visualization
├── interrater_check.py                  # Inter-rater reliability checker
├── CA_Experiment1_Colab.ipynb          # Google Colab notebook
├── logs_qwen2.5_7b{,_refresh13,_refresh13_28,
│   _episodic_rag,_episodic_rag_hybrid,
│   _refresh13_episodic_rag}/            # Per-condition score & context logs
└── results_qwen2.5_7b*/                 # Charts, fits, summary reports

CA_Experiment_2/
├── EXPERIMENT_REPORT.md                 # Full Experiment 2 report
├── CA_Experiment2_Plan.md              # Pre-registered plan
├── ca_assets.py                        # Shared persona, probes, rubrics, RAG helpers
├── generate_lora_dataset.py             # Sonnet 4.6 dataset generator with QC
├── train_lora_sci.py                    # QLoRA training (transformers + PEFT + TRL)
├── experiment_runner.py                 # 4-condition + H5 evaluator (HF + PEFT)
├── analyse_results.py                   # Multi-condition analysis + plots
├── make_slides.py                       # Generates the conference deck (python-pptx)
├── CA_Experiment2_Colab.ipynb          # Google Colab notebook
├── h4_probes.json                       # H4 base-capability probe set (100 prompts × 5 categories)
├── run_h4.py                            # H4 runner — base vs LoRA-10K on out-of-domain probes
├── analyse_h4.py                        # H4 analysis — paired t-test + per-category degradation
├── data/full.jsonl                      # 10K training examples (QC-passed)
├── adapters/lora_{2k,5k,10k}/           # LoRA adapter weights (gitignored)
├── logs/condition_{A,B,C,D,
│   C_lora_2k,C_lora_5k}/                # Per-condition score & context logs
├── logs/h4_{base,lora}/                 # H4 base-capability test logs
└── results/                             # Plots, analysis_data.json, summary report

CA_Experiment_3/
├── EXPERIMENT_REPORT.md                 # Full Experiment 3 report
├── CA_Experiment3_Plan.md              # Pre-registered plan (decision rule §8.5)
├── qpm.py                               # 12-qubit Qiskit Aer QPM circuit
├── cmg_cdk.py                           # Matched classical baseline (CMG-CDK)
├── ca_assets.py                         # Profiles, Battery A pairs, B scenarios, d-vec extractor,
│                                        #   QPM→structured-intent, Exp-2 SCI re-exports
├── experiment_runner.py                 # Battery A/B/C/H4 dispatcher (resumable)
├── analyse_results.py                   # Paired t-tests, Cohen's d, decision-rule emission
├── CA_Experiment3_Colab.ipynb          # Google Colab notebook (12 cells)
├── logs/battery_h4_psychotherapy/       # H4 variance-calibration output
├── logs/battery_a_{psychotherapy,
│   software_eng}/                       # Per-pair JSD values (both models)
├── logs/battery_b_{psychotherapy,
│   software_eng}/                       # Per-scenario entropy + purity values
├── logs/battery_c_{psychotherapy,
│   software_eng}/                       # Per-probe judge scores + per-turn context tracking
└── results/                             # Plots, analysis_data.json, summary_report.md

CA_Experiment_4/
├── EXPERIMENT_REPORT.md                 # Full Experiment 4 report
├── CA_Experiment4_Plan.md              # Pre-registered plan (4 interface conditions)
├── ca_assets.py                         # Profiles + 4 QPM→structured-intent interface variants
├── qpm.py                               # 12-qubit QPM circuit (shared)
├── experiment_runner.py                 # A/B/C/D condition dispatcher (resumable)
├── analyse_results.py                   # Paired t-tests, per-dimension deltas, decision rule
├── CA_Experiment4_Colab.ipynb          # Google Colab notebook
├── logs/                                # Per-condition score & context logs
└── results/                             # Plots, analysis_data.json, summary report

CA_Experiment_5/
├── EXPERIMENT_REPORT.md                 # Full Experiment 5 report
├── CA_Experiment5_Plan.md              # Pre-registered plan (residual-stream steering)
├── steering_vectors.py                  # Phase-0 contrastive vector extraction
├── steering_config.json                 # Locked L*=14, α=7.5 params (SHA-256 pinned)
├── phase0_L_star.json, phase0_activations/   # Layer calibration + extracted activations
├── experiment_runner.py                 # A/B/C/D condition dispatcher (resumable)
├── analyse_results.py                   # Paired t-tests, decision rule
├── CA_Experiment5_Colab.ipynb          # Google Colab notebook
├── calibration_scripts/, logs/          # Held-out calibration + per-condition logs
└── results/                             # Plots, analysis_data.json, summary report

CA_Experiment_6/
├── EXPERIMENT_REPORT.md                 # Full Experiment 6 report
├── CA_Experiment6_Plan.md              # Pre-registered plan (v2.1, QPM-in-scope)
├── ada_sci.json                         # ADA self-model (SCI)
├── ca_assets.py                         # SCI, chat template + special tokens, record schema,
│                                        #   PersonaScore harness (probes/rubrics/judge/κ)
├── model/                               # From-scratch transformer (RoPE/RMSNorm/SwiGLU) + configs
├── qpm.py, qpm_bridge.py                # QPM circuit + ADA profile/d-vector → persona_state
├── tokenizer_util.py, train_tokenizer.py    # 16k BPE loader + trainer
├── prepare_data.py                      # Stage-A shards + SQuAD2/OASST2/Dolly → §4.3 records
├── gen_persona_data.py                  # Sonnet — persona/consistency/introspect/recall/instruct/…
├── data_utils.py                        # Pretrain bins + SFT assistant-span masking
├── train_pretrain.py                    # Stage A (causal) + Stage A2 (prefix-LM continued)
├── train_sft.py                         # Stage B/C SFT (assistant-span masked, persona channel)
├── train_span.py, span_utils.py         # Extractive span + answerability head (reading)
├── retriever.py, rerank_util.py         # Symbolic extract-then-style + sentence reranker
├── evaluate.py                          # H1/H2 QA + H3 PersonaScore + H4 refresh + judge + analyse
├── train_common.py                      # Seed, cosine LR, checkpoint I/O (Drive-dedup safe)
├── CA_Experiment6_300M_Colab.ipynb     # End-to-end Colab notebook (300M run)
├── data/, tokenizer/, checkpoints/      # Corpora/tokenizer/weights live on Drive (gitignored)
└── results/                             # Scores, judge output, figures, analysis_data.json (committed)
```

---

## Running the experiments

**Prerequisites:** Python 3.10+, an Anthropic API key, and access to GPU compute (Colab Pro recommended for training; T4 sufficient for Experiment 1 evaluation; A100 80GB needed for Experiment 2).

### Experiment 1

```bash
cd CA_Experiment_1
pip install ollama anthropic python-dotenv numpy scipy matplotlib
echo "CHA_EXPERIMENT_SONNET_KEY=sk-..." > .env

python generate_scripts.py
python experiment_runner.py --model qwen2.5:7b
python analyse_results.py --model qwen2.5:7b

# Phase 2 interventions
python experiment_runner.py --model qwen2.5:7b --refresh-turn 13
python experiment_runner.py --model qwen2.5:7b --episodic-rag
python experiment_runner.py --model qwen2.5:7b --refresh-turn 13 --episodic-rag
python experiment_runner.py --model qwen2.5:7b --refresh-turns 13,28
```

Or use [`CA_Experiment_1/CA_Experiment1_Colab.ipynb`](CA_Experiment_1/CA_Experiment1_Colab.ipynb) for GPU-accelerated runs.

### Experiment 2

```bash
cd CA_Experiment_2
pip install transformers peft trl bitsandbytes accelerate datasets \
            anthropic python-dotenv sentence-transformers numpy scipy matplotlib

# Dataset generation (~12 hrs, ~$80 in API calls)
python generate_lora_dataset.py --target 10000

# Train adapters (Colab A100 80GB)
python train_lora_sci.py --train-rows 2000  --output adapters/lora_2k
python train_lora_sci.py --train-rows 5000  --output adapters/lora_5k
python train_lora_sci.py --train-rows 10000 --output adapters/lora_10k

# 4-condition evaluation + H5 sub-runs
python experiment_runner.py --condition A --adapter adapters/lora_10k
python experiment_runner.py --condition B --adapter adapters/lora_10k
python experiment_runner.py --condition C --adapter adapters/lora_10k
python experiment_runner.py --condition D --adapter ""
python experiment_runner.py --condition C --adapter adapters/lora_2k --logs-suffix lora_2k
python experiment_runner.py --condition C --adapter adapters/lora_5k --logs-suffix lora_5k

# Analysis
python analyse_results.py
```

Or use [`CA_Experiment_2/CA_Experiment2_Colab.ipynb`](CA_Experiment_2/CA_Experiment2_Colab.ipynb) for the full pipeline end-to-end.

### H4 base-capability test (Experiment 2 follow-up)

Out-of-domain probe battery (100 prompts × 5 categories: general knowledge, code reasoning, math, instruction following, structured intent JSON) verifying that the LoRA-10K adapter does not cause catastrophic forgetting. Pass criterion: < 5% mean degradation on Sonnet 4.5 1-5 scoring.

```bash
cd CA_Experiment_2

# Generate responses from both conditions (A100 80GB, ~30 min each)
python run_h4.py --condition base
python run_h4.py --condition lora --adapter lora_10k

# Analysis (paired t-test, per-category degradation, verdict)
python analyse_h4.py
```

Outputs `results/h4_summary_report.md`, `results/h4_analysis_data.json`, and two comparison plots.

### Experiment 3

```bash
cd CA_Experiment_3
pip install qiskit qiskit-aer pylatexenc numpy scipy matplotlib \
            vaderSentiment anthropic python-dotenv \
            transformers peft bitsandbytes accelerate

# Variance calibration first — must PASS (SNR ≥ 3.0) before main batteries
python experiment_runner.py --battery H4 --profile psychotherapy

# Battery A — order effects (H1, no GPU)
python experiment_runner.py --battery A --profile psychotherapy
python experiment_runner.py --battery A --profile software_eng

# Battery B — ambivalence (H2, no GPU)
python experiment_runner.py --battery B --profile psychotherapy
python experiment_runner.py --battery B --profile software_eng

# Battery C — downstream PersonaScore (H3, A100 GPU, ~4 hr per profile)
python experiment_runner.py --battery C --profile psychotherapy --adapter lora_10k
python experiment_runner.py --battery C --profile software_eng  --adapter lora_10k

# Analysis (paired t-tests, decision-rule emission)
python analyse_results.py --profile psychotherapy
```

Battery C requires Experiment 2's `adapters/lora_10k/` on disk (loaded relative to `CA_Experiment_2/`). Or use [`CA_Experiment_3/CA_Experiment3_Colab.ipynb`](CA_Experiment_3/CA_Experiment3_Colab.ipynb) for the full end-to-end pipeline on Colab.

### Experiments 4 and 5

Both reuse Experiment 3's QPM circuit, Qwen2.5-7B-Instruct + LoRA-10K stack, 30 scripts, and Sonnet 4.5 judge — varying only the QPM→SLM interface. Same dependencies as Experiment 3.

```bash
cd CA_Experiment_4          # richer JSON interface: 4 conditions
python experiment_runner.py --condition {A,B,C,D}
python analyse_results.py

cd ../CA_Experiment_5        # residual-stream steering
python steering_vectors.py                       # Phase 0: extract + lock vectors (L*=14, α=7.5)
python experiment_runner.py --condition {A,B,C,D}
python analyse_results.py
```

Or use each experiment's Colab notebook for the end-to-end pipeline.

### Experiment 6

A from-scratch 321M model — a multi-stage pipeline (Colab A100). Free corpora for skills; bounded Sonnet 4.6 spend only for the ADA persona layer. The large artifacts (corpora, tokenizer, checkpoints) live on Drive; `results/` is committed.

```bash
cd CA_Experiment_6
pip install torch tokenizers datasets anthropic python-dotenv qiskit qiskit-aer vaderSentiment

# --- Persona track (H3/H4/RQ6) ---
python prepare_data.py pretrain --hf-dataset HuggingFaceFW/fineweb-edu --max-tokens 8_000_000_000
python train_pretrain.py --config 300m --run-name pretrain_300m --max-steps 40000        # Stage A
python prepare_data.py oasst && python prepare_data.py dolly --append                     # Stage-B substrate
python train_sft.py --init pretrain_300m_best.pt --sft data/instruct.jsonl --run-name sft_instruct  # Stage B
python gen_persona_data.py consistency --brevity     # + introspect/recall/instruct/persona/style/refusal
python train_sft.py --init sft_instruct_final.pt --sft data/qa_sft.jsonl --run-name sft_ada \
    --persona-oversample 3 --reading-cap 5000 --max-steps 1400                            # Stage C
python evaluate.py persona --checkpoint sft_ada_final.pt --condition R0   # + R1, + --no-qpm ablation
python evaluate.py judge-reliability --scores-dir results/persona_R0

# --- Reading track (H1/H2) ---
python train_pretrain.py --config 300m --init pretrain_300m_best.pt --prefix-lm \
    --run-name pretrain_300m_plm --max-steps 12000                                        # Stage A2
python train_span.py --init pretrain_300m_plm_best.pt --span data/qa_span.jsonl           # span + ans head
python evaluate.py qa --checkpoint span_final.pt --span --answerable-threshold 0.70       # H1/H2

python evaluate.py analyse --results-dir results     # H1-H4 decision table + figures
```

Or use [`CA_Experiment_6/CA_Experiment6_300M_Colab.ipynb`](CA_Experiment_6/CA_Experiment6_300M_Colab.ipynb) for the full end-to-end run.

---

## License

See repository for license details.
