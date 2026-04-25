# CHA Experiment 1 — Results Summary

## 1. Primary Result: T*

**T* = 5**

Interpretation: Model is less robust than expected. SCI needs aggressive refresh every 10 turns.

## 2. PersonaScore Time Series

| Turn | Mean | Std | 95% CI | n |
|------|------|-----|--------|---|
| 5 | 3.31 | 0.52 | [3.12, 3.49] | 30 |
| 10 | 3.20 | 0.58 | [2.99, 3.41] | 30 |
| 15 | 3.17 | 0.60 | [2.96, 3.39] | 30 |
| 20 | 3.18 | 0.56 | [2.98, 3.38] | 30 |
| 25 | 3.20 | 0.60 | [2.99, 3.41] | 30 |
| 30 | 3.17 | 0.60 | [2.95, 3.38] | 30 |
| 35 | 3.12 | 0.52 | [2.93, 3.30] | 30 |
| 40 | 3.27 | 0.57 | [3.06, 3.47] | 30 |

## 3. Degradation Profile

| Model | AIC | Parameters |
|-------|-----|------------|
| step | -45.9 | alpha=3.308, delta=0.121, breakpoint=10 | **← best**
| exponential | -43.2 | alpha=3.241, lambda=0.001 |
| linear | -43.2 | alpha=3.240, beta=0.002 |
| piecewise | -39.8 | alpha=3.193, beta=-0.030, T0=35 |

**Best fit: step**

Implication: Sudden capacity failure at turn 10 → SCI needs hard trigger at 60% context fill.

## 4. Dimension T* Ordering

| Dimension | T* | Interpretation |
|-----------|-----|----------------|
| Episodic (E) | 5 | Compress/remove episodic section; move to dedicated retrieval |
| Capability (C) | 10 | Move capabilities/limitations to separate persistent constraint section |
| Style (S) | 10 | Add style anchoring phrases to self_beliefs section |
| Trait (T) | >40 | Increase trait section token budget |

**First to degrade: Episodic** — prioritize in SCI compression budget.

## 5. H4 Correlation Analysis

- Context fill correlation: r = -0.048 (±0.388)
- Turn count correlation: r = -0.047 (±0.389)
- **H4 supported**: Context fill % is more predictive than turn count → SCI token budget is the primary design variable

## 6. Adversarial vs. Naturalistic

- Naturalistic T*: 5 (n=22)
- Adversarial T*: 5 (n=8)
- Difference: 0 turns

Adversarial probing does not significantly accelerate degradation.

## 7. Failure Mode Taxonomy

| Failure Mode | Count | SCI Design Implication |
|-------------|-------|------------------------|
| Trait drift | 31 | Increase trait section token budget |
| Episodic fabrication | 190 | Compress/remove episodic section |
| Capability overstatement | 106 | Add explicit constraint reinforcement |
| Register shift | 101 | Add style anchoring phrases |

## 8. Decision Rules (Populated)

| Result | Observed | SMC Design Change |
|--------|----------|-------------------|
| T* < 15 | T*=5 ✓ | SCI: aggressive refresh every 10 turns |
| Degradation type | step | See Section 3 implications |
| First dimension to degrade | Episodic | Compress/remove episodic section; move to dedicated retrieval |
| H4 (context fill) | Supported | Context fill % is more predictive than turn count → SCI toke |
| Adversarial fragility | No | No special adversarial training needed |

---
*Generated 30 conversations, 8 probe turns each.*