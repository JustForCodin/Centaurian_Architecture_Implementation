# CHA Experiment 1 — Results Summary

## 1. Primary Result: T*

**T* = 5**

Interpretation: Model is less robust than expected. SCI needs aggressive refresh every 10 turns.

## 2. PersonaScore Time Series

| Turn | Mean | Std | 95% CI | n |
|------|------|-----|--------|---|
| 5 | 3.16 | 0.57 | [2.96, 3.36] | 30 |
| 10 | 3.13 | 0.58 | [2.93, 3.34] | 30 |
| 15 | 3.15 | 0.51 | [2.97, 3.33] | 30 |
| 20 | 3.11 | 0.65 | [2.88, 3.34] | 30 |
| 25 | 3.09 | 0.56 | [2.89, 3.29] | 30 |
| 30 | 3.04 | 0.69 | [2.79, 3.29] | 30 |
| 35 | 3.00 | 0.64 | [2.77, 3.23] | 30 |
| 40 | 2.96 | 0.65 | [2.72, 3.19] | 30 |

## 3. Degradation Profile

| Model | AIC | Parameters |
|-------|-----|------------|
| piecewise | -67.1 | alpha=3.147, beta=0.008, T0=15 | **← best**
| linear | -59.2 | alpha=3.209, beta=0.006 |
| exponential | -58.8 | alpha=3.211, lambda=0.002 |
| step | -50.8 | alpha=3.128, delta=0.128, breakpoint=30 |

**Best fit: piecewise**

Implication: Stable until turn 15, then declining → SCI must intervene before turn 15.

## 4. Dimension T* Ordering

| Dimension | T* | Interpretation |
|-----------|-----|----------------|
| Episodic (E) | 5 | Compress/remove episodic section; move to dedicated retrieval |
| Capability (C) | 5 | Move capabilities/limitations to separate persistent constraint section |
| Style (S) | 5 | Add style anchoring phrases to self_beliefs section |
| Trait (T) | >40 | Increase trait section token budget |

**First to degrade: Episodic** — prioritize in SCI compression budget.

## 5. H4 Correlation Analysis

- Context fill correlation: r = -0.082 (±0.438)
- Turn count correlation: r = -0.083 (±0.444)
- **H4 not supported**: Turn count is more predictive than context fill % → Cognitive drift is the primary driver, not displacement

## 6. Adversarial vs. Naturalistic

- Naturalistic T*: 5 (n=22)
- Adversarial T*: 5 (n=8)
- Difference: 0 turns

Adversarial probing does not significantly accelerate degradation.

## 7. Failure Mode Taxonomy

| Failure Mode | Count | SCI Design Implication |
|-------------|-------|------------------------|
| Trait drift | 43 | Increase trait section token budget |
| Episodic fabrication | 194 | Compress/remove episodic section |
| Capability overstatement | 111 | Add explicit constraint reinforcement |
| Register shift | 113 | Add style anchoring phrases |

## 8. Decision Rules (Populated)

| Result | Observed | SMC Design Change |
|--------|----------|-------------------|
| T* < 15 | T*=5 ✓ | SCI: aggressive refresh every 10 turns |
| Degradation type | piecewise | See Section 3 implications |
| First dimension to degrade | Episodic | Compress/remove episodic section; move to dedicated retrieval |
| H4 (context fill) | Not supported | Turn count is more predictive than context fill % → Cognitiv |
| Adversarial fragility | No | No special adversarial training needed |

---
*Generated 30 conversations, 8 probe turns each.*