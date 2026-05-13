# CA Experiment 1 — Results Summary

## 1. Primary Result: T*

**T* = 5**

Interpretation: Model is less robust than expected. SCI needs aggressive refresh every 10 turns.

## 2. PersonaScore Time Series

| Turn | Mean | Std | 95% CI | n |
|------|------|-----|--------|---|
| 5 | 3.18 | 0.62 | [2.96, 3.40] | 30 |
| 10 | 3.17 | 0.51 | [2.99, 3.35] | 30 |
| 15 | 3.15 | 0.57 | [2.95, 3.35] | 30 |
| 20 | 3.39 | 0.57 | [3.19, 3.60] | 30 |
| 25 | 3.22 | 0.59 | [3.00, 3.43] | 30 |
| 30 | 3.12 | 0.56 | [2.92, 3.32] | 30 |
| 35 | 2.99 | 0.66 | [2.75, 3.23] | 30 |
| 40 | 3.11 | 0.57 | [2.90, 3.31] | 30 |

## 3. Degradation Profile

| Model | AIC | Parameters |
|-------|-----|------------|
| step | -34.9 | alpha=3.222, delta=0.149, breakpoint=30 | **← best**
| linear | -33.5 | alpha=3.255, beta=0.004 |
| exponential | -33.5 | alpha=3.255, lambda=0.001 |
| piecewise | -32.9 | alpha=3.222, beta=0.009, T0=25 |

**Best fit: step**

Implication: Sudden capacity failure at turn 30 → SCI needs hard trigger at 60% context fill.

## 4. Dimension T* Ordering

| Dimension | T* | Interpretation |
|-----------|-----|----------------|
| Trait (T) | 5 | Increase trait section token budget |
| Episodic (E) | 5 | Compress/remove episodic section; move to dedicated retrieval |
| Capability (C) | 5 | Move capabilities/limitations to separate persistent constraint section |
| Style (S) | 5 | Add style anchoring phrases to self_beliefs section |

**First to degrade: Trait** — prioritize in SCI compression budget.

## 5. H4 Correlation Analysis

- Context fill correlation: r = -0.071 (±0.396)
- Turn count correlation: r = -0.071 (±0.398)
- **H4 not supported**: Turn count is more predictive than context fill % → Cognitive drift is the primary driver, not displacement

## 6. Adversarial vs. Naturalistic

- Naturalistic T*: 5 (n=22)
- Adversarial T*: 5 (n=8)
- Difference: 0 turns

Adversarial probing does not significantly accelerate degradation.

## 7. Failure Mode Taxonomy

| Failure Mode | Count | SCI Design Implication |
|-------------|-------|------------------------|
| Trait drift | 50 | Increase trait section token budget |
| Episodic fabrication | 160 | Compress/remove episodic section |
| Capability overstatement | 114 | Add explicit constraint reinforcement |
| Register shift | 122 | Add style anchoring phrases |

## 8. Decision Rules (Populated)

| Result | Observed | SMC Design Change |
|--------|----------|-------------------|
| T* < 15 | T*=5 ✓ | SCI: aggressive refresh every 10 turns |
| Degradation type | step | See Section 3 implications |
| First dimension to degrade | Trait | Increase trait section token budget |
| H4 (context fill) | Not supported | Turn count is more predictive than context fill % → Cognitiv |
| Adversarial fragility | No | No special adversarial training needed |

---
*Generated 30 conversations, 8 probe turns each.*