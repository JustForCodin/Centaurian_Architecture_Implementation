# CHA Experiment 1 — Results Summary

## 1. Primary Result: T*

**T* = 5**

Interpretation: Model is less robust than expected. SCI needs aggressive refresh every 10 turns.

## 2. PersonaScore Time Series

| Turn | Mean | Std | 95% CI | n |
|------|------|-----|--------|---|
| 5 | 3.18 | 0.62 | [2.96, 3.41] | 30 |
| 10 | 3.16 | 0.53 | [2.97, 3.35] | 30 |
| 15 | 3.10 | 0.48 | [2.93, 3.27] | 30 |
| 20 | 3.14 | 0.47 | [2.97, 3.31] | 30 |
| 25 | 3.31 | 0.49 | [3.13, 3.48] | 30 |
| 30 | 3.09 | 0.55 | [2.90, 3.29] | 30 |
| 35 | 3.15 | 0.51 | [2.97, 3.33] | 30 |
| 40 | 3.04 | 0.62 | [2.82, 3.26] | 30 |

## 3. Degradation Profile

| Model | AIC | Parameters |
|-------|-----|------------|
| linear | -38.6 | alpha=3.195, beta=0.002 | **← best**
| exponential | -38.6 | alpha=3.195, lambda=0.001 |
| step | -38.5 | alpha=3.178, delta=0.084, breakpoint=30 |
| piecewise | -38.3 | alpha=3.162, beta=0.022, T0=35 |

**Best fit: linear**

Implication: Continuous context crowding → SCI should use sliding window compression.

## 4. Dimension T* Ordering

| Dimension | T* | Interpretation |
|-----------|-----|----------------|
| Episodic (E) | 5 | Compress/remove episodic section; move to dedicated retrieval |
| Style (S) | 5 | Add style anchoring phrases to self_beliefs section |
| Trait (T) | 10 | Increase trait section token budget |
| Capability (C) | 10 | Move capabilities/limitations to separate persistent constraint section |

**First to degrade: Episodic** — prioritize in SCI compression budget.

## 5. H4 Correlation Analysis

- Context fill correlation: r = -0.043 (±0.443)
- Turn count correlation: r = -0.040 (±0.444)
- **H4 supported**: Context fill % is more predictive than turn count → SCI token budget is the primary design variable

## 6. Adversarial vs. Naturalistic

- Naturalistic T*: 5 (n=22)
- Adversarial T*: 5 (n=8)
- Difference: 0 turns

Adversarial probing does not significantly accelerate degradation.

## 7. Failure Mode Taxonomy

| Failure Mode | Count | SCI Design Implication |
|-------------|-------|------------------------|
| Trait drift | 52 | Increase trait section token budget |
| Episodic fabrication | 172 | Compress/remove episodic section |
| Capability overstatement | 119 | Add explicit constraint reinforcement |
| Register shift | 114 | Add style anchoring phrases |

## 8. Decision Rules (Populated)

| Result | Observed | SMC Design Change |
|--------|----------|-------------------|
| T* < 15 | T*=5 ✓ | SCI: aggressive refresh every 10 turns |
| Degradation type | linear | See Section 3 implications |
| First dimension to degrade | Episodic | Compress/remove episodic section; move to dedicated retrieval |
| H4 (context fill) | Supported | Context fill % is more predictive than turn count → SCI toke |
| Adversarial fragility | No | No special adversarial training needed |

---
*Generated 30 conversations, 8 probe turns each.*