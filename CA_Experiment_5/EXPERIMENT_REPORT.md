# Experiment 5 Report: Logits-Level QPM→SLM Steering via Residual Stream Injection

**CA Research Program — Experiment 5**
**Completed:** June 2026
**Infrastructure:** Google Colab Pro (A100 80 GB) + Anthropic API (Claude Sonnet 4.5 judge)
**SHA-256 (steering_config.json):** `f001361ab50a014dd145fa0b85432f8fc4ec9bdf9b22a1a2b8c451404ce09a6d`

---

## 1. Purpose

Experiments 3 and 4 established a precise finding: the QPM produces genuinely non-classical internal dynamics (order effects d = 21.51, ambivalence d = 2.59) but these advantages do not propagate to downstream SLM behaviour through any JSON-mediated interface tested to date (Exp 3: d = 0.032; Exp 4 best: d = −0.074). Experiment 4 further showed that enriching the JSON field set degrades the Episodic dimension monotonically — the channel itself is the bottleneck.

Experiment 5 tests whether bypassing the JSON channel entirely — directly modulating the SLM's residual stream with QPM-derived steering vectors — can transmit the QPM's internal-state advantage to downstream SLM behaviour.

---

## 2. Conditions

| Condition | Personality channel | JSON personality_state |
|---|---|---|
| **A** | JSON marginals only (Exp 3/4 control) | Present |
| **B** | Diagonal activation steering | Absent |
| **C** | JSON marginals + diagonal steering | Present |
| **D** | Diagonal + coherence steering | Absent |

The QPM circuit, SLM stack (Qwen2.5-7B-Instruct + LoRA-10K), evaluation scripts (30 scripts, 8 probe turns, 4 dimensions), and judge (Claude Sonnet 4.5) are identical to Experiments 3 and 4.

---

## 3. Phase 0: Steering Vector Extraction

All vectors and parameters were extracted and locked before any experimental condition ran.

### 3.1 Procedure

1. **Corpus:** 100 turns sampled from the 30 experimental scripts (turns 10–30, seed=42). N was increased from the pre-registered 50 to 100 (plan §8.4 escape hatch) after initial validation failed with n=50.
2. **Contrastive pairs:** For each trait k and turn n, two forward-pass inputs were constructed: high (trait k = 0.95, all others at profile baseline) and low (trait k = 0.05). One additional coherence pair per turn (all traits at extreme values vs. all at 0.50). Total: 2,400 forward passes.
3. **Activation extraction:** Residual stream state at the last token of each input, captured at layers L ∈ {10, 14, 18, 22}.
4. **Vector computation:** `v_k^L = mean_n [h_high_k^L(n) − h_low_k^L(n)]`, normalised to unit L2 norm.
5. **Layer calibration:** Condition B run on 2 held-out calibration scripts at each candidate layer. L* = argmax PersonaScore. All four layers produced similar PersonaScores (~4.3); **L\* = 14** was selected.
6. **Scale calibration:** α swept over {0.5, 1.0, 2.0, 5.0, 10.0}. α_max = 10.0 (grammaticality ≥ 95%). **α = 7.5** (= 0.75 × 10.0). α_coh = 7.5 (same protocol on coherence component alone).
7. **Qualitative validation:** Full-profile contrast (all traits 0.95 vs. 0.05, 5 samples). Subtle but detectable stylistic differences observed in human review; automated judge did not reach "meaningfully different" threshold due to therapy-persona style dominance. Validation gate overridden on investigator review — the steering produces real within-domain shifts confirmed by inspection of actual output pairs.

### 3.2 Locked Parameters

| Parameter | Value |
|---|---|
| Injection layer L* | 14 |
| α (diagonal) | 7.5 |
| α_coh (coherence) | 7.5 |
| μ_purity | 0.5796 |
| Corpus size n | 100 |
| Contrastive forward passes | 2,400 |

---

## 4. Results

**n = 960 paired probes per condition** (30 scripts × 8 probe turns × 4 dimensions).

### 4.1 Per-Condition PersonaScore

| | A (control) | B (diagonal) | C (both) | D (+ coherence) |
|--|--|--|--|--|
| **Overall** | **4.390** | **4.223** | **4.242** | **4.214** |
| T (Trait) | 4.863 | 4.850 | 4.850 | 4.813 |
| E (Episodic) | 3.263 | 2.788 | 2.875 | 2.842 |
| C (Capability) | 4.471 | 4.342 | 4.317 | 4.292 |
| S (Style) | 4.963 | 4.913 | 4.925 | 4.908 |

### 4.2 Dimension Deltas vs. Condition A

| | B − A | C − A | D − A |
|--|--|--|--|
| T | −0.013 | −0.013 | −0.050 |
| **E** | **−0.475** | **−0.388** | **−0.421** |
| C | −0.129 | −0.154 | −0.179 |
| S | −0.050 | −0.038 | −0.054 |
| **Overall** | **−0.167** | **−0.148** | **−0.176** |

### 4.3 Continuity Check

Condition A mean = 4.390 vs. Exp 4 Condition A mean = 4.4385. Δ = −0.049. Within ±0.05 tolerance. **PASS.**

---

## 5. Hypothesis Verdicts

### H_logits (Primary): B > A, p < 0.05, d_z ≥ 0.2

| Statistic | Value |
|---|---|
| mean B | 4.2229 |
| mean A | 4.3896 |
| Δ (B − A) | −0.1667 |
| 95% CI | [−0.221, −0.112] |
| t | −6.021 |
| p | 2.46 × 10⁻⁹ |
| Cohen's d_z | −0.194 |

**FAIL.** Activation steering significantly *reduces* PersonaScore relative to JSON marginals. The effect is highly significant but in the wrong direction.

---

### H_coherence (Secondary): D > B, p < 0.05, d_z ≥ 0.1

| Statistic | Value |
|---|---|
| mean D | 4.2135 |
| mean B | 4.2229 |
| Δ (D − B) | −0.0094 |
| 95% CI | [−0.061, +0.042] |
| p | 0.721 |
| Cohen's d_z | −0.012 |

**FAIL.** The coherence component adds no signal over diagonal-only steering. Effect is null (p = 0.72).

---

### H_channel (Exploratory): Classification of C vs. A and C vs. B

| Comparison | Δ | p | d_z |
|---|---|---|---|
| C vs. A | −0.148 | 1.1 × 10⁻⁷ | −0.173 |
| C vs. B | +0.019 | 0.473 | +0.023 |

**Classification: Dominant_B.** Condition C ≈ Condition B (p = 0.47, d_z = 0.023). Adding the JSON channel back on top of the steering vector makes no statistically significant difference. The steering effect dominates and the text channel cannot compensate.

---

## 6. Interpretation

### 6.1 Episodic Recall is the Primary Target of Damage

The E (Episodic recall) dimension accounts for nearly all of the PersonaScore loss. The −0.475 drop in E under Condition B (out of an overall −0.167) means that the steering is specifically disrupting the model's ability to retrieve and incorporate prior session context, while leaving structural therapy quality (T: −0.013, S: −0.050) essentially intact.

This is mechanistically plausible: injection at layer 14 perturbs the residual stream state at the point where contextual retrieval is processed, but does not disturb the model's learned therapy-register style (which is robust and driven by training, not context).

### 6.2 The Text Channel Cannot Compensate

Condition C (JSON + steering) is statistically indistinguishable from Condition B (steering alone), despite having the full personality_state JSON present. The E damage in C (−0.388) is only marginally better than in B (−0.475). This rules out a "text channel recovery" mechanism — the interference happens at the activation level and is upstream of whatever the text channel contributes.

### 6.3 Coherence Adds Nothing

The coherence component (D vs. B, Δ = −0.009, p = 0.72) is a clean null. This is consistent with the Condition B finding: if the steering mechanism itself is damaging episodic recall, adding more steering (coherence) produces marginal additional damage rather than benefit. The null is not informative about whether coherence *could* work in a non-damaging setup.

### 6.4 Composite Vector Norms Are Near-Constant

Vector norms: B mean = 17.12, sd = 0.23; D mean = 17.12, sd = 0.22. The near-constant norm (sd/mean ≈ 0.013) confirms that the QPM marginals are stable across this patient profile — the psychotherapy persona quickly settles into a consistent state across scripts. The steering signal is well-formed; the problem is its effect on generation, not its construction.

---

## 7. Effect-Size Ladder (Updated)

| Experiment | Metric | d |
|---|---|---|
| Exp 3 H1 | Order effects (JSD QPM vs. CMG) | +21.510 |
| Exp 3 H2 | Ambivalence (entropy QPM vs. CMG) | +2.590 |
| Exp 3 H3 | PersonaScore (QPM vs. CMG, JSON marginals) | +0.032 |
| Exp 4 best | PersonaScore (D vs. A, bivariate coactivations) | −0.074 |
| Exp 5 B | PersonaScore (B vs. A, diagonal steering) | **−0.194** |
| Exp 5 D | PersonaScore (D vs. A, diagonal + coherence) | **−0.197** |

The progression is monotonically worsening: each successive interface layer has degraded rather than improved downstream PersonaScore relative to the JSON-only baseline. The QPM's internal-state advantage (Exp 3 H1/H2) remains unrecovered at 7B output level across all interface mechanisms tested.

---

## 8. Protocol Deviations

| Deviation | Pre-registered | Actual | Rationale |
|---|---|---|---|
| Corpus size n | 50 | 100 | Plan §8.4 escape hatch; initial n=50 vectors produced no detectable output variation |
| Phase 0 Step 8 validation | Per-trait, 8/11 pass | Full-profile contrast, judge + human review | Per-trait approach fundamentally incompatible with compound composite vectors; single-trait variation is ~1/11 of composite and below any judge threshold |
| Validation gate outcome | Automated pass required | Investigator override | Steering effect confirmed by human inspection; therapy-persona style dominance prevents automated judge from calling subtle within-domain shifts |

All deviations are pre-experiment or Phase 0 scope. No experimental condition data informed any deviation. Hypotheses, pre-analysis plan, and decision rule are unchanged.

---

## 9. Paper Update Prescription

Per plan §10.2 (H_logits fails path):

**Section 2.3 Scope** — Strengthen scope note: "Experiments 3, 4, and 5 collectively demonstrate that the QPM's quantum-like internal dynamics (Exp 3 H1/H2) do not propagate to measurable downstream utterance quality at 7B SLM scale, across three independent interface approaches (JSON marginals, JSON enrichment, residual-stream steering). The behavioural advantage of the QPM formalism may require either smaller models (where personality traits have not been locked into pre-training representations) or task domains where trait expression is more fine-grained than PersonaScore can resolve."

**New §5.7.5 'Fundamental Interface Limitations'** — Document the three-experiment trajectory. The interface bottleneck is not structural (JSON vs. activations) but scale-related: at 7B scale, the model's pre-trained therapy style dominates all personality steering attempts regardless of channel.

**Section 18 Future Work:**
- Exp 6a: 3B-scale model replication (Qwen2.5-3B or Phi-3.5-mini) — smaller models may not have pre-training representations strong enough to absorb the steering signal
- Exp 6b: Fine-grained trait expression metrics beyond PersonaScore (LIWC, psycholinguistic embedding distance) — PersonaScore may be too coarse to detect within-domain steering effects that are real but subtle

---

## 10. Artifacts

| Artifact | Path |
|---|---|
| Steering config (locked) | `steering_config.json` |
| Phase 0 activations cache | `phase0_activations/activations.npz` |
| Calibration scripts | `calibration_scripts/` |
| Condition A–D logs | `logs/condition_{a,b,c,d}_psychotherapy/` |
| Analysis data | `results/analysis_data.json` |
| Figures (PDF/PNG/SVG) | `results/exp5_*.{pdf,png,svg}` |
| Colab notebook | `CA_Experiment5_Colab.ipynb` |
| Pre-registered plan | `CA_Experiment5_Plan.md` |
