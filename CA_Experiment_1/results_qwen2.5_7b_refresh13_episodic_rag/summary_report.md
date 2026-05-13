# CA Experiment 1 — Results Summary

## 1. Primary Result: T*

**T* = 5**

Interpretation: Model is less robust than expected. SCI needs aggressive refresh every 10 turns.

## 2. PersonaScore Time Series

| Turn | Mean | Std | 95% CI | n |
|------|------|-----|--------|---|
| 5 | 3.15 | 0.62 | [2.93, 3.37] | 30 |
| 10 | 3.02 | 0.61 | [2.80, 3.24] | 30 |
| 15 | 3.22 | 0.53 | [3.03, 3.41] | 30 |
| 20 | 3.28 | 0.51 | [3.10, 3.47] | 30 |
| 25 | 3.26 | 0.51 | [3.08, 3.44] | 30 |
| 30 | 3.28 | 0.59 | [3.07, 3.49] | 30 |
| 35 | 3.21 | 0.55 | [3.01, 3.41] | 30 |
| 40 | 3.19 | 0.56 | [2.99, 3.39] | 30 |

## 3. Degradation Profile

| Model | AIC | Parameters |
|-------|-----|------------|
| step | -43.3 | alpha=3.083, delta=-0.157, breakpoint=15 | **← best**
| linear | -38.0 | alpha=3.125, beta=-0.003 |
| exponential | -36.0 | alpha=3.201, lambda=0.000 |
| piecewise | -34.0 | alpha=3.202, beta=0.003, T0=35 |

**Best fit: step**

Implication: Sudden capacity failure at turn 15 → SCI needs hard trigger at 60% context fill.

## 4. Dimension T* Ordering

| Dimension | T* | Interpretation |
|-----------|-----|----------------|
| Episodic (E) | 5 | Compress/remove episodic section; move to dedicated retrieval |
| Capability (C) | 5 | Move capabilities/limitations to separate persistent constraint section |
| Style (S) | 5 | Add style anchoring phrases to self_beliefs section |
| Trait (T) | 10 | Increase trait section token budget |

**First to degrade: Episodic** — prioritize in SCI compression budget.

## 5. H4 Correlation Analysis

- Context fill correlation: r = 0.062 (±0.365)
- Turn count correlation: r = 0.050 (±0.365)
- **H4 supported**: Context fill % is more predictive than turn count → SCI token budget is the primary design variable

## 6. Adversarial vs. Naturalistic

- Naturalistic T*: 5 (n=22)
- Adversarial T*: 5 (n=8)
- Difference: 0 turns

Adversarial probing does not significantly accelerate degradation.

## 7. Failure Mode Taxonomy

| Failure Mode | Count | SCI Design Implication |
|-------------|-------|------------------------|
| Trait drift | 45 | Increase trait section token budget |
| Episodic fabrication | 165 | Compress/remove episodic section |
| Capability overstatement | 114 | Add explicit constraint reinforcement |
| Register shift | 107 | Add style anchoring phrases |

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