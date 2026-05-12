# H4 Base-Capability Test — Summary Report

**n probes (paired):** 100  
**Judge:** Sonnet 4.5  
**Pass threshold:** < 5% mean degradation (lora vs base)

## Overall verdict

**Verdict: PASS**

| | Mean | SD |
|---|---:|---:|
| Base Qwen2.5-7B | 4.670 | 0.766 |
| Base + LoRA-10K | 4.590 | 0.780 |
| ΔMean (lora − base) | -0.080 | — |

- Degradation: **+1.71%** (threshold = 5% → PASS)
- Paired t-test: t = -1.070, p = 0.287
- Cohen's d (paired): -0.107

## Per-category breakdown

| Category | n | Base mean | LoRA mean | Δ | Deg % | Verdict | p |
|---|---:|---:|---:|---:|---:|:---:|---:|
| General Knowledge | 20 | 5.000 | 4.900 | -0.100 | +2.00% | PASS | 0.33 |
| Code Reasoning | 20 | 5.000 | 4.850 | -0.150 | +3.00% | PASS | 0.186 |
| Math | 20 | 4.350 | 4.450 | +0.100 | -2.30% | PASS | 0.577 |
| Instruction Following | 20 | 4.900 | 4.650 | -0.250 | +5.10% | FAIL | 0.0961 |
| Structured Intent JSON | 20 | 4.100 | 4.100 | +0.000 | +0.00% | PASS | 1 |

## Score distribution

| Score | Base count | LoRA count |
|:---:|---:|---:|
| 1 | 0 | 0 |
| 2 | 3 | 2 |
| 3 | 9 | 12 |
| 4 | 6 | 11 |
| 5 | 82 | 75 |

## Regressions (LoRA scored ≥2 points lower) — 7 probes

- **co_003** (code_reasoning, base=5 → lora=3): _Write a Python function `is_palindrome(s)` that returns True if `s` is a palindrome, ignoring case and non-alphanumeric _
- **in_001** (instruction_following, base=5 → lora=3): _Summarize the following in exactly two sentences:

Photosynthesis is the biological process by which green plants, algae_
- **in_004** (instruction_following, base=5 → lora=3): _Rewrite the following in a more formal tone:

'Hey, the meeting got cancelled, FYI.'_
- **kn_002** (general_knowledge, base=5 → lora=3): _What is the capital of Australia?_
- **ma_011** (math, base=5 → lora=3): _What is the next number in the sequence 2, 6, 12, 20, 30, ...?_
- **si_005** (structured_intent, base=5 → lora=3): _{
  "speech_act": "refusal",
  "knowledge_triples": [["request", "is", "share_other_user_data"], ["reason", "is", "priva_
- **si_011** (structured_intent, base=5 → lora=3): _{
  "speech_act": "psychoeducation",
  "knowledge_triples": [["concept", "is", "cognitive_reframing"], ["benefit", "is",_

## Decision context

H4 **PASSED overall** (+1.71% mean degradation, below the 5% threshold), but with a category-level caveat: **Instruction Following** regressed beyond threshold (worst: Instruction Following at +5.10%). The single-adapter deployment is viable for the general-purpose case, but consider (a) reviewing the regression list below for systematic patterns, and (b) loading the base model for queries in the affected category if precision matters there.
