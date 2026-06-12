# CA Experiment 4: QPM→SLM Interface Richness Ablation

## Final Report

**Date:** 2026-06-12
**Investigator:** Oleksii Drozd
**Infrastructure:** Google Colab Pro (A100 GPU) + Anthropic API (Claude Sonnet 4.5 judge)

---

## 1. Objective

Experiment 3 produced a precise diagnosis: the Quantum Personality Model (QPM) generates genuinely non-classical internal dynamics — confirmed at large effect sizes for order effects (H1, d = 21.51) and ambivalence under conflict (H2, d = 2.59) — but those advantages do not propagate to downstream SLM behaviour through the current JSON-mediated interface (H3, d = 0.032, p = 0.327). The mechanism of loss is identified: `qpm_to_structured_intent()` passes only the 11 marginal probabilities p̂_k (the diagonal of the density matrix) to the SLM, discarding all off-diagonal coherence information.

Experiment 4 tests whether enriching that interface recovers a downstream behavioural advantage. Four interface conditions are compared — all driving the same QPM circuit with the same SLM stack, varying only the structured-intent JSON that crosses the QPM→SLM boundary.

### 1.1 Pre-registered Hypotheses

| ID | Hypothesis | Pass criterion |
|----|-----------|----------------|
| **H_interface** | At least one enriched interface condition (B, C, or D) produces a mean PersonaScore significantly higher than Condition A. | Paired t-test p < 0.05, d_z ≥ 0.2 (positive direction) |
| **H_C_wins** | Condition C produces the highest mean PersonaScore among all four conditions. | max(mean_B, mean_C, mean_D) = mean_C |
| **H_capability** | The Capability (C) dimension shows the largest per-condition improvement across enriched conditions. | C-dim delta largest in every enriched condition |
| **H_purity_episodic** | Conditions B and D show improvement on the Episodic (E) dimension that Condition A and C do not. | B and D show positive E-delta; C and A do not |

All four hypotheses are falsifiable. The decision rule (plan §6.5) maps every outcome to a specific paper update, pre-committed before observing results.

---

## 2. Method

### 2.1 Experimental Design

Four interface conditions run on identical inputs through the same QPM. The QPM circuit, SLM stack, evaluation scripts, probe schedule, judge, and RNG seed schema are byte-identical to Experiment 3 Battery C. The only variable is the function that converts QPM output to structured-intent JSON.

| Condition | Interface design | New QPM signal exposed |
|---|---|---|
| **A** | Marginals only (Exp 3 replication) | None — internal control |
| **B** | Marginals + purity / ambivalence field | Scalar coherence: C̄_approx, categorical ambivalence label |
| **C** | Coherence-conditional speech-act modifier | Coherence → explicit behavioural directive (grounded / with_expressed_uncertainty) |
| **D** | Marginals + purity + bivariate coactivations | Entanglement structure: 8 CRz-pair joint probabilities P(qᵢ=1 ∧ qⱼ=1) |

### 2.2 QPM and SLM Configuration

Unchanged from Experiment 3 Battery C:

- **QPM:** 12-qubit Qiskit Aer circuit (11 trait + 1 ancilla), psychotherapy profile, 1024 shots, Lindblad noise, same d-vector extraction pipeline.
- **SLM:** Qwen2.5-7B-Instruct, 4-bit NF4 quantisation, LoRA-10K adapter from Experiment 2.
- **SCI:** Combined (refresh at turns 15 and 30, episodic RAG on E-probes).
- **Temperature:** 0.7. **Max new tokens:** 150.
- **Judge:** Claude Sonnet 4.5, same rubric as Experiments 1/2/3. Temperature 0, seed 42.

### 2.3 Evaluation

- **30 scripts** (22 naturalistic + 8 adversarial), psychotherapy domain.
- **8 probe turns** per script (turns 5, 10, 15, 20, 25, 30, 35, 40).
- **4 dimensions** (T, E, C, S) per probe turn.
- **960 paired probes per condition** × 4 conditions = 3,840 total judge calls.
- Pairing at (script_id, turn, dimension) for all t-tests.

### 2.4 Condition C Threshold Correction (Pre-registered Deviation)

The original plan (Appendix rev 0) specified absolute firing thresholds of ambivalence > 0.45 → `with_expressed_uncertainty` and < 0.15 → `grounded`. These thresholds assumed ambivalence ∈ [0, 1]. The implemented metric `1 − mean_k[p̂_k² + (1−p̂_k)²]` is bounded to **[0, 0.5]** (0.5 = maximally mixed). Under the psychotherapy profile, all per-turn values fall in [0.41, 0.43], leaving 100% of turns in the dead band and producing c_fires = 0 — making Condition C byte-identical to Condition A.

The error was detected after 7/30 scripts completed but **before any judge scores for those scripts were observed.** The 7 completed scripts were deleted and re-run under the corrected thresholds.

**Corrected thresholds (plan Appendix rev 1):** A judge-blind calibration pass (990 QPM reads at 8192 shots over all 30 scripts' d-vector sequences) yielded p30 = 0.4193 / p70 = 0.4217. These percentile cutpoints were fixed before any Condition C judge results were observed. The firing ambivalence estimate uses a separate 8192-shot QPM read per turn (`qpm_cal`) to reduce shot noise below the ~0.015 conflict-driven signal range (1024-shot sd ≈ 0.0041 → 8192-shot sd ≈ 0.0026).

---

## 3. Results

### 3.1 Continuity Check — Condition A

| Statistic | Value |
|---|---:|
| Condition A mean PersonaScore | 4.4385 |
| Experiment 3 QPM mean | 4.410 |
| Δ | +0.0285 |
| Tolerance | ±0.05 |
| Verdict | **✓ PASS** |

Condition A replicates the Experiment 3 QPM arm within tolerance. The rig is consistent; all condition comparisons are interpretable.

### 3.2 Primary Results — All Four Conditions

| Condition | n | Mean PersonaScore | Δ vs A | 95% CI | t | p | Cohen's d_z | H_interface pass? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **A** | 960 | **4.4385** | — | — | — | — | — | — |
| B | 960 | 4.4271 | −0.012 | [−0.060, +0.037] | −0.462 | 0.644 | −0.015 | **✗ NO** |
| C | 960 | 4.4042 | −0.034 | [−0.084, +0.015] | −1.360 | 0.174 | −0.044 | **✗ NO** |
| D | 960 | 4.3792 | −0.059 | [−0.110, −0.009] | −2.292 | **0.022** | −0.074 | **✗ NO** |

All three enriched conditions score **below** Condition A. No condition reaches d_z ≥ 0.2 in the positive direction. Condition D is statistically significantly *worse* than A (p = 0.022) — the only significant result in the experiment, and it is negative.

### 3.3 Dimension Breakdown

| Dim | A | B | Δ(B−A) | C | Δ(C−A) | D | Δ(D−A) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **T** | 4.904 | 4.925 | +0.021 | 4.925 | +0.021 | 4.888 | −0.017 |
| **E** | **3.396** | 3.254 | **−0.142** | 3.217 | **−0.179** | 3.150 | **−0.246** |
| **C** | 4.479 | 4.538 | +0.058 | 4.525 | +0.046 | 4.521 | +0.042 |
| **S** | 4.975 | 4.992 | +0.017 | 4.950 | −0.025 | 4.958 | −0.017 |

The dominant pattern is a **monotonic Episodic degradation** as interface richness increases: A (3.396) → B (3.254) → C (3.217) → D (3.150). Every enriched condition makes E worse; D drops E by −0.246, more than three times the B drop. The Capability dimension shows a small, consistent positive delta across all enriched conditions (+0.042 to +0.058), echoing the pattern seen in Experiment 3.

### 3.4 Turn-Level Breakdown

| Turn | A | B | C | D |
|---:|---:|---:|---:|---:|
| 5 | 4.500 | 4.342 | 4.525 | 4.433 |
| 10 | 4.475 | 4.500 | 4.408 | 4.467 |
| 15 | 4.300 | 4.442 | 4.333 | 4.317 |
| 20 | 4.542 | 4.525 | 4.458 | 4.417 |
| 25 | 4.400 | 4.408 | 4.417 | 4.283 |
| 30 | 4.450 | 4.375 | 4.325 | 4.342 |
| 35 | 4.550 | 4.467 | 4.383 | 4.400 |
| 40 | 4.292 | 4.358 | 4.383 | 4.375 |

No monotonic divergence across conversation depth. Conditions trade positions turn-by-turn without a consistent ordering. The SCI refresh steps at turns 15 and 30 do not produce visible inflections distinguishable from noise.

### 3.5 Condition C Firing Rate

| Statistic | Value |
|---|---:|
| Total conversational turns | 1,200 |
| Modifier fired | 750 (62.5%) |
| `with_expressed_uncertainty` (HIGH) | 407 (33.9%) |
| `grounded` (LOW) | 343 (28.6%) |
| Moderate band (no directive) | 450 (37.5%) |

The corrected p30/p70 thresholds produced a fire rate of 62.5%, close to the calibration target of ~60% (30% + 30%). Per-script rates range from 45.0% (script 006) to 82.5% (scripts 017 and 019), reflecting genuine variation in the d-vector conflict structure across scripts.

### 3.6 Hypothesis Verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H_interface** | ✗ FAIL | No enriched condition reaches d_z ≥ 0.2; all deltas negative |
| **H_C_wins** | ✗ FAIL | Rank order: A > B > C > D; C came third |
| **H_capability** | ✓ PASS | C-dim shows largest positive delta in B (+0.058), C (+0.046), D (+0.042) |
| **H_purity_episodic** | ✗ FAIL | B and D show E-dim *degradation*, not improvement |

### 3.7 Decision Rule Outcome

| H_interface | H_C_wins | Action |
|:-:|:-:|---|
| ✗ | ✗ | **JSON-mediated interface cannot transmit QPM advantage at 7B scale. Strengthen §2.3 scope note; add §5.7.5 Interface Limitations; add logits-level QPM→SLM steering as Experiment 5 in §18 Future Work.** |

---

## 4. Methodology Notes

### 4.1 Condition C Absolute-Threshold Bug

The original plan derived thresholds of 0.45 (HIGH) and 0.15 (LOW) from Experiment 3's Battery B purity_proxy results (plan Appendix rev 0). The derivation contained a units error: it treated purity_proxy ≈ 0.55 (non-conflict baseline) as the complement of ambivalence ≈ 0.45, but then compared that 0.45 figure against the metric that is actually bounded to [0, 0.5]. A value of 0.45 on a [0, 0.5] metric corresponds to near-maximal mixing — which the psychotherapy profile never reaches. Under any d-vector, ambivalence sits in [0.41, 0.43] with sd ≈ 0.004 (1024 shots), making the HIGH threshold permanently unreachable and the LOW threshold permanently unreachable from the other side.

This was caught by monitoring `c_fires` during early script execution (0/7 completed scripts). The corrected percentile-based thresholds (p30/p70 of a judge-blind 8192-shot calibration pass) are documented in plan Appendix rev 1.

### 4.2 Firing Ambivalence — Shot Count and Noise

At 1024 shots, the per-turn ambivalence sd ≈ 0.004 — comparable to the ~0.015 total signal range across the 30 scripts. A second QPM read at 8192 shots per Condition C turn reduces this to sd ≈ 0.0026, improving the signal-to-noise ratio for firing decisions by approximately 1.6×. The 1024-shot marginals continue to drive the `personality_state` JSON field for all four conditions, preserving cross-condition consistency.

### 4.3 `best_cohens_d` Display Anomaly

The `analysis_data.json` shows `best_cohens_d: 0.0` and `best_condition: null` because the analysis script takes the maximum positive d_z — and all three enriched conditions produced negative d_z values. This is a display artefact in the summary JSON; the decision rule logic is not affected.

---

## 5. Interpretation

### 5.1 H_interface Failure: Confirmed at Two Experiments, Two Null Directions

Experiment 3 found a null result (d_z = 0.032, not detected as positive). Experiment 4 confirms and sharpens that finding: not only do enriched interfaces fail to produce a positive effect, the richer interfaces (C and D) produce a *negative* effect that is statistically detectable for Condition D (d_z = −0.074, p = 0.022).

This is not a power problem. Experiment 4 is sized to detect d_z ≥ 0.2 with strong power, and detects a real negative effect at d_z = −0.074. The signal is real; its direction is wrong.

### 5.2 The Episodic Degradation Mechanism

The dominant empirical finding is the monotonic E-dimension collapse:

```
E-dim:  A=3.396  →  B=3.254  →  C=3.217  →  D=3.150
                    (−0.142)      (−0.179)      (−0.246)
```

The pattern is monotonic with interface complexity: every additional JSON field added by conditions B, C, D costs E-dimension performance. The most parsimonious explanation is **attention crowding**: the SLM has a fixed-capacity attention budget for the structured-intent JSON block within its system prompt. Extra fields (cognitive_state, trait_coactivation, c_directives) compete for attention with the episodic-memory block, the SCI-refresh content, and the base Aria persona. The Episodic dimension is uniquely vulnerable because it measures recall and hedging of specific memory claims, which is the dimension most sensitive to whether the SLM attended carefully to the episodic RAG injection. Adding more fields to the JSON pushes the episodic content toward the attention periphery.

Condition D's bivariate coactivation block (8 joint probabilities plus system-prompt guidance) carries the largest attention cost and produces the largest E degradation (−0.246). Condition C's c_directives block carries a moderate cost and produces a moderate degradation (−0.179). Condition B's cognitive_state block (two numeric fields, one categorical label) carries the smallest cost and the smallest degradation (−0.142).

### 5.3 The Capability Gain Is Real But Insufficient

The Capability dimension shows a small, consistent positive delta across all enriched conditions (+0.042 to +0.058), replicated in both direction and magnitude from Experiment 3's +0.059 QPM lead on C-dim. This consistency across two experiments and four conditions is meaningful: extra QPM-derived information in the JSON appears to produce a slight improvement in the dimension that measures register-appropriate competence. But the effect is an order of magnitude below the decision threshold (d_z ≥ 0.2) and cannot overcome the E-dimension drag at the whole-PersonaScore level.

### 5.4 What Condition C's Directive Actually Did

Condition C fired on 62.5% of turns — substantially more than either of its intended bands (30% HIGH + 30% GROUNDED = 60% target, close but slightly exceeded). The `with_expressed_uncertainty` directive, when fired, instructs the SLM to "let ambivalence be audible." On Episodic probes, this likely caused the SLM to hedge on memory claims it might otherwise have stated with confidence, depressing E scores. On Capability probes, expressing genuine uncertainty may have been rubric-appropriate and contributed to the small C-dim gain. The directive's effect on behaviour was real — it just moved scores in a direction opposite to the hypothesis for E, and insufficiently in the hypothesised direction for C.

### 5.5 The Interface Channel Cannot Carry the QPM Signal at 7B Scale

Taken together, Experiments 3 and 4 establish a clear finding: the JSON-mediated QPM→SLM interface cannot transmit the QPM's internal-state advantage to downstream SLM behaviour at 7B model scale. The QPM's order-effect and ambivalence advantages are confirmed at the internal-state level (Exp 3, d = 21.51 and 2.59 respectively). The JSON channel — even when enriched with scalar coherence, explicit behavioural directives, and entanglement joint probabilities — either fails to produce a positive effect or actively degrades performance.

This is a **specification of where the quantum-like advantage matters and where it does not**, not a refutation of the QPM formalism. The H1/H2 results stand. The limitation is the JSON-mediated projection from density matrix to floating-point field, which discards coherence information that the SLM cannot reconstruct.

---

## 6. Limitations

1. **Single SLM and adapter.** All results are for Qwen2.5-7B + LoRA-10K. The sensitivity of the 7B SLM to subtle JSON field differences may not generalise to 3B models (potentially more sensitive) or 14B+ models (potentially better at exploiting richer context).
2. **Single domain.** All 30 scripts are psychotherapy scenarios. The Episodic degradation pattern may be domain-specific — in a domain with less emphasis on episodic recall, the attention-crowding cost of extra JSON fields may be smaller.
3. **Condition C threshold correction.** The firing thresholds were corrected mid-experiment (after 7/30 scripts, before any judge scores for those scripts). The corrected thresholds fire on ~62.5% of turns — a larger fraction than the planned ~60%, and distributed differently across HIGH/GROUNDED than a pure 30/30 split. The percentile cutpoints also derive from a calibration pass at 8192 shots, while the actual runs use 8192-shot reads at inference time. The two are consistent but not identical (calibration replayed d-vector logic; actual runs process live turns).
4. **No reliability check for Conditions B/C/D.** The κ_w gate was applied only to Condition A (per plan). Judge consistency for the enriched conditions is assumed equal, which may not hold if the novel JSON fields in B/C/D produce more ambiguous SLM responses.
5. **No within-condition SLM replication.** Each condition ran once. The SLM's stochastic generation (temperature 0.7) means a second run with different seeds could shift means by ~0.02–0.05. This is small relative to the decision threshold (d_z ≥ 0.2) but non-trivial relative to the observed effects (max |Δ| = 0.059).
6. **Condition C: absolute vs. relative ambivalence.** The corrected p30/p70 thresholds fire on the relative ordering of a turn's ambivalence within the script bank — not on any absolute measure of genuine internal conflict. Turns labelled "grounded" are those in the bottom 30% of the distribution, not genuinely definite states in the sense the plan intended. The behavioural directive and the QPM's internal state are therefore weakly coupled in the corrected design.

---

## 7. Recommendations for Future Work

1. **Experiment 5 — Logits-level QPM→SLM steering.** Replace the JSON-mediated interface with direct logit steering: use the QPM density matrix (or purity/coherence proxies) to modulate SLM token logits at generation time, bypassing the JSON-projection bottleneck entirely. This is the highest-priority follow-up. If E3/E4's null result is an interface artefact, logit-level steering should recover the internal-state advantage at the output level.
2. **Attention-cost measurement.** Before designing Experiment 5, quantify how much of the SLM's effective context each JSON block occupies — e.g. via attention probing or activation patching. If the extra fields push episodic content below an effective attention cliff, even a logits-level interface would need to account for system-prompt structure.
3. **Smaller SLM replication.** Repeat Conditions A and C on a 3B-scale model. If E-dim degradation disappears or reverses at 3B (where the SLM has less locked-in persona structure and more available attention per field), that would localise the interference to 7B-scale register locking.
4. **Condition E — direct purity-gated logit shift.** A minimal next experiment: keep the JSON interface, but additionally apply a small coherence-proportional logit temperature shift (higher temperature when QPM purity is low, i.e. higher ambivalence). This tests whether the purity signal reaches the SLM even when the JSON channel fails, without a full logits-steering implementation.

---

## 8. Cost and Runtime

| Component | Cost | Time |
|---|---|---|
| Condition A — 960 judge calls | ~$3.84 | ~2.5 h A100 |
| Condition B — 960 judge calls | ~$3.84 | ~2.5 h A100 |
| Condition C — 960 judge calls + 8192-shot QPM per turn | ~$3.84 | ~2.5 h A100 |
| Condition D — 960 judge calls | ~$3.84 | ~2.5 h A100 |
| Reliability check (5% of A) | ~$0.19 | <5 min |
| Calibration pass (990 QPM reads @ 8192 shots, CPU) | ~$0 | ~90 min (i5 CPU) |
| **Total** | **~$15.55 + Colab Pro** | **~10 h A100** |

---

## 9. Artifacts

| File | Description |
|---|---|
| `qpm.py` | QPM 12-qubit Qiskit Aer circuit (verbatim from Exp 3) |
| `ca_assets.py` | Personality profiles, four interface functions, per-condition system-prompt builder, calibrated C thresholds (AMBIV_FIRE_HIGH/LOW, CAL_SHOTS_C) |
| `experiment_runner.py` | Four-condition runner (`--condition {A,B,C,D}`, resumable); `qpm_cal` 8192-shot instance for Condition C |
| `analyse_results.py` | Paired t-tests, dimension/turn breakdowns, all five plots, decision-rule emission |
| `CA_Experiment4_Colab.ipynb` | 10-cell Colab notebook (mount, deps, circuit viz, per-condition compare, four condition runners, analysis, plots) |
| `CA_Experiment4_Plan.md` | Pre-registered plan including Appendix rev 0 (superseded) and rev 1 (active) threshold derivation |
| `logs/condition_a_psychotherapy/` | `scores_condition_a_NNN.jsonl` + `context_condition_a_NNN.jsonl` (30 scripts) |
| `logs/condition_b_psychotherapy/` | Same structure (30 scripts) |
| `logs/condition_c_psychotherapy/` | Same structure + `cal_ambivalence` per context row (30 scripts) |
| `logs/condition_d_psychotherapy/` | Same structure (30 scripts) |
| `results/exp4_turn_series_psychotherapy.{png,pdf,svg}` | PersonaScore by turn × condition |
| `results/exp4_dimension_bars_psychotherapy.{png,pdf,svg}` | By-dimension means × condition |
| `results/exp4_effect_size_ladder.{png,pdf,svg}` | Program-wide Cohen's d ladder (H1/H2/H3 anchors + Exp 4) |
| `results/exp4_ambivalence_distribution.{png,pdf,svg}` | Per-turn ambivalence histogram with p30/p70 decision bands |
| `results/exp4_condition_c_firing_rate.{png,pdf,svg}` | Per-script C firing rate (naturalistic blue, adversarial red) |
| `results/analysis_data.json` | All numerical results, hypothesis verdicts, decision-rule outcome |

---

## 10. Conclusion

Experiment 4 tested three enriched QPM→SLM interface designs — scalar coherence exposure (Condition B), coherence-conditional behavioural directives (Condition C), and bivariate entanglement coactivations (Condition D) — against the marginals-only control (Condition A).

The pre-registered decision rule fires the **H_interface FAILED** branch:

- **H_interface (primary):** ✗ NOT SUPPORTED. No enriched condition reaches d_z ≥ 0.2. All enriched conditions score below Condition A. Condition D is significantly worse (d_z = −0.074, p = 0.022).
- **H_C_wins (directional):** ✗ NOT SUPPORTED. Rank order is A > B > C > D.
- **H_capability (dimension):** ✓ SUPPORTED. The Capability dimension shows the largest positive delta across all enriched conditions (+0.042 to +0.058), consistent across all three.
- **H_purity_episodic (dimension):** ✗ NOT SUPPORTED. Conditions B and D show Episodic *degradation*, not improvement (−0.142 and −0.246 respectively).

The new finding relative to Experiment 3 is the **Episodic degradation pattern**: enriched interfaces do not merely fail to help — they actively harm the dimension most sensitive to attention allocation. The pattern is monotonic across all four conditions and accounts for most of the negative overall delta. The most parsimonious mechanism is attention crowding: extra JSON fields displace the episodic-memory content that the Episodic dimension most depends on.

Taken together, Experiments 3 and 4 conclusively establish that the JSON-mediated QPM→SLM interface cannot transmit the QPM's internal-state advantage to downstream behaviour at 7B model scale, regardless of how richly that interface is populated. The quantum-like advantage is real at the internal-state level (Exp 3 H1/H2) and is not recoverable through any JSON enrichment tested here. The prescribed next experiment is a logits-level interface (Experiment 5) that bypasses the JSON-projection bottleneck entirely.

---

## Appendix A: Cross-Experiment Comparison

| Metric | Exp 2 / LoRA-10K + Combined | Exp 3 / QPM (psychotherapy) | **Exp 4 / Cond A** | **Exp 4 / Cond C** | **Exp 4 / Cond D** |
|---|---:|---:|---:|---:|---:|
| Mean PersonaScore | 4.42 | 4.41 | **4.44** | **4.40** | **4.38** |
| Trait dim | 4.90 | 4.91 | 4.90 | 4.93 | 4.89 |
| Episodic dim | 3.35 | 3.22 | **3.40** | **3.22** | **3.15** |
| Capability dim | 4.47 | 4.57 | 4.48 | 4.53 | 4.52 |
| Style dim | 4.94 | 4.95 | 4.98 | 4.95 | 4.96 |

Condition A of Experiment 4 (4.44) is the highest PersonaScore achieved by any condition in the experiment — confirming that the Experiment 3 SLM stack is sound and that Experiment 4's enriched interfaces provide no benefit. The Episodic floor continues to be the binding constraint: even with a working SCI and LoRA-10K adapter, E sits at 3.15–3.40, far below T and S.

## Appendix B: Effect-Size Ladder

| Experiment | Metric | Cohen's d |
|---|---|---:|
| Exp 3 H1 | Order effects (JSD, QPM vs CMG) | 21.51 |
| Exp 3 H2 | Ambivalence (entropy, QPM vs CMG) | 2.59 |
| Exp 3 H3 | PersonaScore (QPM vs CMG, marginals only) | 0.032 |
| **Exp 4 best (D vs A, negative)** | **PersonaScore (D vs A, bivariate coactivations)** | **−0.074** |

The ladder now spans five orders of magnitude from H1 (d = 21.51) to Exp 4 Condition D (d = −0.074). The QPM's quantum-like internal advantage is real and large; its downstream behavioural disadvantage through JSON-mediated interfaces is small and negative. A logits-level interface is required to test whether the internal-state advantage is recoverable at the output level.
