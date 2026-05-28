# Centaurian Architecture — Implementation

Empirical validation of the **Centaurian Architecture (CA)**, a multi-layered AI system that pairs a symbolic/quantum cognitive core with a lightweight neural periphery (small language models, neural TTS, procedural animation). The architecture encodes personality via a Quantum Personality Model (QPM) running on classical hardware and confines neural networks to bounded I/O transduction roles, preserving end-to-end traceability of every behavioral decision.

**Architecture specifications:**
- [`Centaurian_Architecture_v3.md`](Centaurian_Architecture_v3.md) — current spec
- [`Centaurian_Architecture_v2.md`](Centaurian_Architecture_v2.md) — previous version (kept for reference)
- [`Interpretable_Architectures_Revised_v1.md`](Interpretable_Architectures_Revised_v1.md) — interpretable-architectures position paper

The empirical work in this repository validates the **Self-Model Component (SMC)** sub-architecture — i.e., whether a 7B SLM can serve as the linguistic transducer **and** hold an Aria-grade Structured Cognitive Identity (SCI) reliably across long conversations.

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

---

## License

See repository for license details.
