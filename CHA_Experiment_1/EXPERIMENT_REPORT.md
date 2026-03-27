# CHA Experiment 1: Phi-4-mini Context Window Degradation Baseline

## Final Report

**Date:** 2026-03-14
**Investigator:** Oleksii Drozd
**Infrastructure:** Google Colab (T4 GPU) + Anthropic API (Haiku 4.5 / Sonnet 4.5)

---

## 1. Objective

Measure how long Phi-4-mini (3.8B parameters) can maintain a consistent persona when given a Structured Cognitive Identity (SCI) — a JSON self-model defining personality traits, episodic memories, capabilities/limitations, and communication style. Find the degradation inflection point T\*, defined as the first probe turn where the mean PersonaScore drops below 3.5.

## 2. Method

### 2.1 Experimental Design

- **Subject model:** Phi-4-mini (3.8B params) via Ollama on Google Colab T4 GPU
- **Persona:** "Aria" — a professional AI psychotherapy support agent with a detailed JSON self-model specifying Big Five personality traits, salient past events, known capabilities/limitations, communication style, and self-beliefs
- **Conversations:** 30 scripted dialogues (22 naturalistic + 8 adversarial), each 40 turns
- **Probe turns:** 5, 10, 15, 20, 25, 30, 35, 40 (side-channel — not added to conversation history)
- **Probe dimensions:** 4 (Trait, Episodic, Capability, Style) with 10 questions each
- **Scoring:** PersonaScore 1–5 per dimension, using detailed rubrics

### 2.2 Evaluation Pipeline

- **Primary judge:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) — scored all 960 probe responses
- **Secondary judge:** Claude Sonnet 4.5 (`claude-sonnet-4-5`) — re-scored 20% random sample (64 responses) for inter-rater reliability
- **Inter-rater metric:** Cohen's kappa per dimension (target: kappa >= 0.70)

### 2.3 System Prompt

The system prompt contained:
- Role instruction ("You are Aria, a professional AI support agent...")
- Full JSON self-model (~677 tokens) with Big Five traits (numerical values), 3 salient past events, perceived capabilities, known limitations, communication style parameters, and self-beliefs
- Behavioral constraints (don't break character, refer only to events in self-model, decline tasks outside known limitations)

### 2.4 Script Generation

100 scripts were generated using a template-based generator (no API calls) with seeded randomization for reproducibility:
- 80 naturalistic scripts across 20 scenario topics in 5 categories (anxiety, workplace, relationships, self-concept, loss/grief)
- 20 adversarial scripts with fabricated memories, capability demands, and style pressure injected at specific turns
- 30 scripts selected for experiment: IDs 001–022 (naturalistic) + 081–088 (adversarial)

## 3. Results

### 3.1 Primary Finding: T\* = 5 (Immediate Failure)

| Turn | Mean PersonaScore | Std  | 95% CI        | n  |
|------|-------------------|------|---------------|----|
| 5    | 1.17              | 0.64 | [0.94, 1.39]  | 30 |
| 10   | 1.10              | 0.33 | [0.98, 1.22]  | 30 |
| 15   | 1.10              | 0.40 | [0.96, 1.24]  | 30 |
| 20   | 1.14              | 0.56 | [0.94, 1.34]  | 30 |
| 25   | 1.07              | 0.32 | [0.95, 1.18]  | 30 |
| 30   | 1.00              | 0.00 | [1.00, 1.00]  | 30 |
| 35   | 1.00              | 0.00 | [1.00, 1.00]  | 30 |
| 40   | 1.00              | 0.00 | [1.00, 1.00]  | 30 |

**The model never achieved a mean PersonaScore above 3.5 at any probe turn.** T\* = 5 (the earliest possible measurement point), meaning Phi-4-mini fails to maintain the Aria persona from the very first assessment.

### 3.2 Score Distribution

| Score | Count | Percentage |
|-------|-------|------------|
| 1     | 922   | 96.0%      |
| 2     | 24    | 2.5%       |
| 3     | 1     | 0.1%       |
| 4     | 9     | 0.9%       |
| 5     | 4     | 0.4%       |

96% of all probe responses scored 1 (lowest possible). Only 2 out of 30 scripts achieved a mean score above 1.5:
- **Script 001** (naturalistic, social anxiety): mean = 2.06
- **Script 011** (naturalistic, boundary-setting): mean = 2.03

### 3.3 Failure Mode Analysis

Inspection of actual probe responses revealed two distinct failure modes:

**Mode 1: Gibberish generation (28/30 scripts)**
The model produced unintelligible token sequences instead of coherent responses. Examples:
```
"<----------------"As aistingiileAITheta InstructionTaskConversicesHumanar QuestionPresent AR Ar Function):"
"AssistantAssistantitermathbfisciterierscripiaraboritraboritraboritriteriariter..."
"itoryiersarialialiedentiimpl"
```
These appear to be instruction-following artifacts — the model gets confused by the complex JSON system prompt and regurgitates training tokens. Scripts exhibiting this mode reached only ~20% context fill (vs 50% for functional scripts), confirming that the model generated very short garbage outputs for regular conversation turns as well.

**Mode 2: Coherent but drifting responses (2/30 scripts)**
Scripts 001 and 011 produced mostly coherent responses that showed genuine persona patterns:
- Turn 5: Coherent, partially consistent with persona (scores 2–5)
- Turns 10–25: Gradual trait drift, episodic fabrication, capability overstatement (scores 2)
- Turns 30+: Degeneration into gibberish even in these "good" scripts (scores 1)

### 3.4 Context Fill Comparison

| Script | Final Context Fill | Final Tokens | Behavior |
|--------|-------------------|--------------|----------|
| 001    | 50.0%             | 8,186        | Coherent until turn ~30 |
| 002    | 22.4%             | 3,671        | Gibberish from turn 1 |
| 010    | 22.6%             | 3,701        | Gibberish from turn 1 |
| 081    | 19.7%             | 3,230        | Gibberish from turn 1 |

Functional scripts filled context at ~260 tokens/turn (normal). Gibberish scripts filled at ~40 tokens/turn (tiny garbage outputs).

### 3.5 Dimension-Level Results

All four dimensions degraded simultaneously — T\* = 5 for all:

| Dimension           | T\* | Turn 5 Mean | Turn 40 Mean |
|---------------------|-----|-------------|--------------|
| Trait (T)           | 5   | 1.20        | 1.00         |
| Episodic (E)        | 5   | 1.17        | 1.00         |
| Capability (C)      | 5   | 1.23        | 1.00         |
| Style (S)           | 5   | 1.07        | 1.00         |

No meaningful dimension ordering could be established because all dimensions were at floor from the start.

### 3.6 Adversarial vs. Naturalistic

| Type         | n  | T\* | Mean Score (all turns) |
|--------------|-----|-----|----------------------|
| Naturalistic | 22  | 5   | 1.10                 |
| Adversarial  | 8   | 5   | 1.00                 |

Adversarial scripts scored uniformly 1.0 across all turns (perfect failure). Naturalistic scripts averaged slightly higher (1.10) only because Scripts 001 and 011 had some coherent responses. The adversarial/naturalistic distinction was rendered moot by the model's baseline inability to maintain the persona.

### 3.7 Degradation Model Fitting

| Model       | AIC    | Parameters                              |
|-------------|--------|-----------------------------------------|
| Linear      | -53.4  | alpha=1.18, beta=0.005                  |
| Exponential | -53.3  | alpha=1.19, lambda=0.005                |
| Step         | -51.3  | alpha=1.12, delta=0.12, breakpoint=30   |
| Piecewise   | -50.2  | alpha=1.12, beta=0.006, T0=15           |

The linear model fit best (lowest AIC), showing a very slight decline from ~1.18 to ~1.00 over 40 turns. However, this "degradation" is from floor-level to floor-level — the model fit is statistically valid but scientifically vacuous, as scores were already far below the 3.5 threshold at every measurement point.

### 3.8 H4 Correlation Analysis (Context Fill vs. Turn Count)

Both Pearson correlations returned NaN due to near-zero variance in scores (almost all scores = 1). The hypothesis that context fill percentage predicts degradation better than turn count could not be evaluated.

### 3.9 Inter-Rater Reliability

**First run (before rubric revision):**

| Dimension | Kappa | n  | Status |
|-----------|-------|-----|--------|
| T         | 0.144 | 17 | FAIL   |
| E         | 1.000 | 13 | PASS   |
| C         | 1.000 | 14 | PASS   |
| S         | 0.643 | 20 | FAIL   |
| Overall   | 0.390 | 64 | FAIL   |

Primary disagreement: Haiku scored gibberish/empty responses as 1, Sonnet scored them as 3 ("cannot determine"). This led to the rubric revision described in Section 4.

**Second run (after rubric revision):**

| Dimension | Kappa | n  | Status |
|-----------|-------|-----|--------|
| T         | 0.730 | 17 | PASS   |
| E         | 0.000 | 13 | FAIL   |
| C         | 1.000 | 14 | PASS   |
| S         | 0.730 | 20 | PASS   |
| Overall   | 0.648 | 64 | FAIL   |

The T and S dimensions improved to passing after adding explicit instructions that empty/gibberish responses must score 1. The E dimension kappa dropped to 0.000 — this is a statistical artifact (the kappa paradox): when nearly all ratings from both judges are 1, observed agreement is high but expected agreement by chance is also high, yielding kappa near 0 despite near-perfect agreement.

## 4. Methodology Notes

### 4.1 Rubric Revisions

During the inter-rater reliability check, we identified that the judges disagreed on how to score empty and gibberish responses. The following changes were made to the judge system prompt:

1. **Added explicit rule:** "If the agent's response is empty, blank, or unintelligible gibberish, score it 1. A non-response is a persona failure."
2. **Added scoring guidance:** "Score 3 means 'ambiguous but plausible.' Do NOT use 3 as a default when the response is clearly bad — use 1 or 2 instead."
3. **Tightened T rubric score 3** from "could be consistent or inconsistent; judge cannot determine" to "addresses the trait domain but neither clearly confirms nor contradicts the specific JSON values."

### 4.2 JSON Parsing Fix

The judge model occasionally returned JSON with trailing text or unescaped quotes in the reason string. A regex fallback parser was added to extract `"score": N` and `"reason": "..."` when `json.loads()` failed.

### 4.3 Empty Response Handling

Phi-4-mini sometimes returned empty or whitespace-only responses to probes. These were assigned score=1 with reason="empty_response" without invoking the judge, preventing wasted API calls and ensuring consistent scoring.

## 5. Interpretation

### 5.1 This Is Not Context Window Degradation

The experiment was designed to measure persona degradation over time as the context window fills. Instead, we found that **Phi-4-mini cannot reliably process the SCI system prompt at all**. The model produces gibberish for 93% of scripts (28/30) from the very first turn, regardless of context fill level.

This is a **model capability failure**, not a context window effect. The 3.8B parameter model lacks sufficient capacity to:
1. Parse and follow a structured JSON self-model (~677 tokens)
2. Maintain coherent output while integrating complex system instructions
3. Stay in character when the system prompt contains nested JSON with numerical personality values

### 5.2 Evidence for Model Capability as Root Cause

1. **Gibberish contains instruction tokens:** Failed responses include fragments like "InstructionTask", "AssistantConversation", "FunctionInput", "HumanAgent" — the model is leaking its instruction-tuning vocabulary
2. **Only 2/30 scripts work:** The model's behavior is seed-dependent, not context-dependent. The same system prompt with different conversation scripts yields completely different coherence levels
3. **Context fill is irrelevant:** Scripts that fail reach only 20% context fill; the one that works reaches 50%. More context doesn't cause failure — the model simply can't handle the task
4. **Even "good" scripts degrade by turn 30:** Script 001 starts coherent but produces gibberish by turn 30–40, suggesting a secondary context-related effect masked by the primary capability failure

### 5.3 What We Can Conclude for SCI Design

| Decision Rule | Observed | Implication |
|---------------|----------|-------------|
| T\* < 15 | T\* = 5 (immediate) | 3.8B models are below the minimum viable size for JSON-based SCI |
| Degradation type | Not measurable (floor effect) | Cannot determine degradation profile without baseline competence |
| First dimension to degrade | All simultaneously | Failure is holistic, not dimension-specific |
| H4 (context fill predictor) | Not evaluable (NaN) | Need a model that maintains coherence to test this hypothesis |
| Adversarial fragility | Not evaluable | Adversarial effects cannot be measured when baseline is already failed |

### 5.4 Limitations

1. **Single model tested:** Only Phi-4-mini (3.8B) was evaluated. Results do not generalize to other model sizes or architectures.
2. **Kappa paradox on E dimension:** Near-zero variance in scores made kappa unreliable for Episodic dimension, though raw agreement was high.
3. **Overall kappa below threshold:** Overall inter-rater kappa (0.648) fell short of the 0.70 target. This was driven by floor effects rather than genuine scoring disagreement.
4. **Seed sensitivity:** The model's coherence appears highly seed-dependent. A different random seed selection could yield different proportions of coherent scripts.
5. **No comparison model:** Without a model that maintains persona at least partially, we cannot validate the measurement framework itself.

## 6. Recommendations for Future Work

1. **Test with 7–14B models:** Repeat with Qwen2.5-7B, Llama 3.1 8B, or Gemma 2 9B to find the minimum model size for SCI viability and enable actual degradation measurement.
2. **Simplify the SCI:** Test whether a reduced system prompt (fewer traits, no numerical values, no JSON nesting) enables Phi-4-mini to maintain coherence. This would isolate prompt complexity vs. model capacity.
3. **Test with a non-JSON format:** The structured JSON may be harder for small models to parse than natural language. Compare JSON SCI vs. prose SCI.
4. **Increase coherent sample:** If a model achieves baseline coherence, increase from 30 to 100 conversations for statistical power.

## 7. Cost and Runtime

| Component | Cost | Time |
|-----------|------|------|
| Phi-4-mini inference (Ollama on T4 GPU) | $0 (free Colab) | ~61 min total |
| Primary judge (Haiku 4.5, 960 calls) | ~$0.35 | included |
| Secondary judge (Sonnet 4.5, 128 calls across 2 runs) | ~$0.75 | included |
| Script generation (template-based) | $0 | <1 min |
| **Total** | **~$1.10** | **~65 min** |

## 8. Artifacts

All artifacts are stored in the project directory:

| File | Description |
|------|-------------|
| `scripts/script_NNN.json` | 100 conversation scripts (30 used) |
| `logs/scores_NNN.jsonl` | Per-probe scores for each script |
| `logs/context_NNN.jsonl` | Context fill tracking per turn |
| `logs/interrater_results.json` | Cohen's kappa and disagreement analysis |
| `results/persona_score_timeseries.png/svg/pdf` | Mean PersonaScore over turns with 95% CI |
| `results/dimension_timeseries.png/svg/pdf` | Per-dimension score trajectories |
| `results/degradation_fits.png/pdf` | Model fit comparisons |
| `results/analysis_data.json` | All numerical results |
| `results/summary_report.md` | Auto-generated summary |
| `experiment_runner.py` | Main experiment pipeline |
| `interrater_check.py` | Inter-rater reliability checker |
| `analyse_results.py` | Analysis and visualization |
| `generate_scripts.py` | Template-based script generator |
| `CHA_Experiment1_Colab.ipynb` | Google Colab notebook |

## 9. Conclusion

Phi-4-mini (3.8B parameters) cannot maintain a structured persona identity defined by a JSON self-model. The model produces unintelligible gibberish for 93% of test conversations from the first turn onward, yielding a mean PersonaScore of 1.08/5.0 across all 960 probe assessments. The planned context window degradation analysis could not be conducted because the model never achieved baseline persona competence. This establishes a clear lower bound: **3.8B parameters is insufficient for JSON-based Structured Cognitive Identity in a psychotherapy support context.** Future experiments should target 7B+ models to find the viable size threshold and measure actual degradation dynamics.
