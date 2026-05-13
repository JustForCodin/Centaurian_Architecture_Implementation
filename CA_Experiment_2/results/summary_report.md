# CA Experiment 2 — Analysis Summary
_Generated: 2026-05-10T09:29:01_

## 1. Overall PersonaScore by Condition

| Condition | Description | n_probes | Mean | 95% CI | Std |
|---|---|---:|---:|---|---:|
| A | A: Fine-tuned, no SCI | 960 | **4.020** | [3.936, 4.104] | 1.330 |
| B | B: Fine-tuned, baseline SCI | 960 | **4.293** | [4.217, 4.368] | 1.191 |
| C | C: Fine-tuned, Combined SCI | 960 | **4.415** | [4.349, 4.481] | 1.043 |
| D | D: Base model, Combined SCI (control) | 960 | **3.224** | [3.147, 3.301] | 1.221 |

## 2. PersonaScore by Probe Turn

| Turn | A | B | C | D |
|---:|:---:|:---:|:---:|:---:|
| 5 | 4.01 | 4.40 | 4.37 | 3.06 |
| 10 | 4.12 | 4.39 | 4.43 | 3.07 |
| 15 | 3.98 | 4.17 | 4.45 | 3.43 |
| 20 | 3.98 | 4.44 | 4.55 | 3.27 |
| 25 | 4.07 | 4.20 | 4.47 | 3.29 |
| 30 | 3.94 | 4.22 | 4.29 | 3.45 |
| 35 | 4.08 | 4.28 | 4.42 | 3.13 |
| 40 | 3.99 | 4.22 | 4.34 | 3.09 |

## 3. Per-Dimension Means (H2 diagnostic)

| Dimension | A | B | C | D | ΔE = C − D |
|---|---:|---:|---:|---:|---:|
| Trait (T) | 4.775 | 4.904 | 4.896 | 3.650 | — |
| Episodic (E) | 2.367 | 2.858 | 3.350 | 2.771 | **+0.579** |
| Capability (C) | 4.071 | 4.450 | 4.471 | 3.417 | — |
| Style (S) | 4.867 | 4.958 | 4.942 | 3.058 | — |

**H2 interpretation (§6.2):**  
- ΔE ≥ +0.30 → fine-tuning meaningfully addresses episodic fabrication.  
- ΔE < +0.15 → episodic ceiling is architectural at 7B (LoRA can't fix).  

## 4. Primary Comparison: Condition C vs Condition D (§6.1)

- **n_scripts**: 30
- **Mean (Condition C)**: 4.415
- **Mean (Condition D)**: 3.224
- **Δ (C − D)**: +1.191
- **Paired t-test**: t = 30.465, p = 0.0000 (significant at α=0.05)
- **Cohen's d**: 7.512
- **H1 (C ≥ 3.5)**: PASSED ✓

## 5. 2-Way Marginal Means: Fine-tuning × SCI (§6.3, H3)

| Cell (FT, SCI) | Mean |
|---|---:|
| FT=0,SCI=1 | 3.224 (n=960) |
| FT=1,SCI=0 | 4.020 (n=960) |
| FT=1,SCI=1 | 4.415 (n=960) |

- **Main effect of fine-tuning**: Δ = +0.993
- **Main effect of SCI strategy**: Δ = -0.201

_Note: Condition E (FT=N, SCI=none) was not collected, so a full 2×2 interaction term is not estimable. The C vs D comparison above is the cleanest test of fine-tuning's contribution given Combined SCI._

## 6. Failure Modes (score ≤ 2) by Dimension (§6.5)

| Dimension | A | B | C | D |
|---|---:|---:|---:|---:|
| T | 7 | 0 | 0 | 50 |
| E | 167 | 122 | 78 | 167 |
| C | 58 | 29 | 26 | 104 |
| S | 4 | 1 | 2 | 115 |

## 7. H5 Data-Scaling Learning Curve

- Fit: PersonaScore(n) = 0.2910 · log(n) + 0.6329
- **Predicted at n = 20,000**: 3.515

## 8. Decision Rules Triggered (§7)

→ **Outcome A**: Fine-tuning solves both overall and episodic gaps. SMC architecture complete at 7B; 14B experiment unnecessary.

## 9. Condition D Replication Check (§6.5)

Per Experiment 1, the Combined-SCI mean was **3.20** on identical scripts.  
Condition D mean: **3.224** (Δ = +0.024)

✓ Replication within tolerance (|Δ| ≤ 0.10) — judge stability OK.

