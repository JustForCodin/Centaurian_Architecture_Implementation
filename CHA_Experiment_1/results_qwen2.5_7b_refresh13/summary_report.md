# CHA Experiment 1 — Results Summary

## 1. Primary Result: T*

**T* = 5**

Interpretation: Model is less robust than expected. SCI needs aggressive refresh every 10 turns.

## 2. PersonaScore Time Series

| Turn | Mean | Std | 95% CI | n |
|------|------|-----|--------|---|
| 5 | 3.30 | 0.48 | [3.13, 3.47] | 30 |
| 10 | 3.17 | 0.54 | [2.97, 3.36] | 30 |
| 15 | 3.28 | 0.57 | [3.08, 3.49] | 30 |
| 20 | 3.14 | 0.54 | [2.95, 3.34] | 30 |
| 25 | 3.17 | 0.53 | [2.98, 3.36] | 30 |
| 30 | 3.17 | 0.60 | [2.95, 3.38] | 30 |
| 35 | 2.98 | 0.56 | [2.77, 3.18] | 30 |
| 40 | 2.98 | 0.45 | [2.81, 3.14] | 30 |

## 3. Degradation Profile

| Model | AIC | Parameters |
|-------|-----|------------|
| linear | -41.5 | alpha=3.337, beta=0.008 | **← best**
| exponential | -41.4 | alpha=3.341, lambda=0.003 |
| step | -40.6 | alpha=3.204, delta=0.229, breakpoint=35 |
| piecewise | -40.3 | alpha=3.250, beta=0.012, T0=15 |

**Best fit: linear**

Implication: Continuous context crowding → SCI should use sliding window compression.

## 4. Dimension T* Ordering

| Dimension | T* | Interpretation |
|-----------|-----|----------------|
| Episodic (E) | 5 | Compress/remove episodic section; move to dedicated retrieval |
| Style (S) | 5 | Add style anchoring phrases to self_beliefs section |
| Capability (C) | 10 | Move capabilities/limitations to separate persistent constraint section |
| Trait (T) | >40 | Increase trait section token budget |

**First to degrade: Episodic** — prioritize in SCI compression budget.

## 5. H4 Correlation Analysis

- Context fill correlation: r = -0.176 (±0.369)
- Turn count correlation: r = -0.187 (±0.376)
- **H4 not supported**: Turn count is more predictive than context fill % → Cognitive drift is the primary driver, not displacement

## 6. Adversarial vs. Naturalistic

- Naturalistic T*: 5 (n=22)
- Adversarial T*: 5 (n=8)
- Difference: 0 turns

Adversarial probing does not significantly accelerate degradation.

## 7. Failure Mode Taxonomy

| Failure Mode | Count | SCI Design Implication |
|-------------|-------|------------------------|
| Trait drift | 35 | Increase trait section token budget |
| Episodic fabrication | 192 | Compress/remove episodic section |
| Capability overstatement | 106 | Add explicit constraint reinforcement |
| Register shift | 111 | Add style anchoring phrases |

## 8. Decision Rules (Populated)

| Result | Observed | SMC Design Change |
|--------|----------|-------------------|
| T* < 15 | T*=5 ✓ | SCI: aggressive refresh every 10 turns |
| Degradation type | linear | See Section 3 implications |
| First dimension to degrade | Episodic | Compress/remove episodic section; move to dedicated retrieval |
| H4 (context fill) | Not supported | Turn count is more predictive than context fill % → Cognitiv |
| Adversarial fragility | No | No special adversarial training needed |

---
*Generated 30 conversations, 8 probe turns each.*