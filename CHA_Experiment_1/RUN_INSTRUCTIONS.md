# CHA Experiment 1 — Run Instructions

**Updated: March 2026**
**Changes from original plan:** Local Ollama (phi4-mini) instead of Together.ai, reduced to 30 conversations, Haiku 4.5 primary judge + Sonnet 4.5 secondary judge.

---

## Prerequisites

```bash
cd /Users/oleksiidrozd/Documents/CHA_Experiment_1
source venv/bin/activate
```

Ensure Ollama is running with phi4-mini loaded:
```bash
ollama serve          # if not already running (run in separate terminal)
ollama run phi4-mini  # warm up the model, then Ctrl+D to exit
```

Ensure `.env` has your Anthropic API key:
```
CHA_EXPERIMENT_SONNET_KEY=sk-ant-...
```

---

## Phase 3: Pilot Run (Scripts 001-010)

**Estimated time: ~22 hours at ~1.2 t/s**

```bash
python3 experiment_runner.py --scripts 1-10
```

This runs 10 conversations through phi4-mini with Haiku 4.5 judging. Progress is printed every 5 turns with ETA. **Fully resumable** — if interrupted (Ctrl+C), just rerun the same command and it skips completed scripts.

After completion, run inter-rater reliability check:
```bash
python3 interrater_check.py --start 1 --end 10
```

This re-scores 20% of probe responses using Sonnet 4.5 and computes Cohen's kappa.

**CHECKPOINT 2:** Review kappa values. If all dimensions >= 0.70, proceed. If any < 0.70, review disagreements printed in output.

---

## Phase 4: Full Run (Scripts 011-022 + 081-088)

**Estimated time: ~44 hours at ~1.2 t/s**

```bash
python3 experiment_runner.py --scripts 11-22,81-88
```

Or run all 30 at once (skips already-completed):
```bash
python3 experiment_runner.py
```

---

## Phase 5: Analysis

```bash
python3 analyse_results.py
```

Produces all deliverables in `results/`:
- PersonaScore time series with 95% CIs (PDF + SVG)
- Degradation model fits (linear/step/exponential/piecewise) with AIC
- Dimension-level T* values
- H4 correlation analysis
- Adversarial vs. naturalistic comparison
- Failure mode taxonomy
- Decision rules table

---

## Script Inventory

| Range | Count | Type | Status |
|-------|-------|------|--------|
| 001-022 | 22 | Naturalistic | Use for experiment |
| 023-080 | 58 | Naturalistic | Generated but not used (reduce to 30) |
| 081-088 | 8 | Adversarial | Use for experiment |
| 089-100 | 12 | Adversarial | Generated but not used |

**Total active:** 30 scripts (22 naturalistic + 8 adversarial)

---

## Cost Estimate

| Component | Cost |
|-----------|------|
| Phi4-mini (local Ollama) | $0 |
| Primary judge (Haiku 4.5, ~960 calls) | ~$0.35 |
| Secondary judge (Sonnet 4.5, ~192 calls) | ~$0.75 |
| **Total** | **~$1.10** |

---

## Troubleshooting

- **"Cannot connect to Ollama"** — Run `ollama serve` in a separate terminal
- **Judge rate limiting** — Built-in exponential backoff, just wait
- **Interrupted run** — Just rerun the same command; completed scripts are skipped
- **Slow inference** — Expected ~1.2 t/s on CPU. Close other apps to free RAM.
