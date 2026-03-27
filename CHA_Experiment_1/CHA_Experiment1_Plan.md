# Experiment 1: Phi-4-mini Context Window Degradation Baseline

**CHA Self-Modeling Research Program — Pre-Implementation Validation**  
**Version 1.0 | March 2026**

---

## 1. Purpose and Position in the Research Program

This experiment is the mandatory first step before any Self-Model Component (SMC) implementation. It establishes the *baseline capability* of Phi-4-mini as the linguistic transducer for self-narration tasks — specifically, how long it can maintain a consistent persona when given a structured JSON self-model document, with no other architectural support. The result defines three things the rest of the SMC design depends on:

1. **The degradation inflection point T\*** — the turn at which PersonaScore drops below 3.5/5.0. This sets the minimum compression requirement for the Structured Context Injector (SCI): the SCI must refresh or re-compress the self-model context before turn T\*.

2. **The token budget** — how much context the self-model JSON can occupy before it crowds out conversation history and KG triples. This cannot be assumed; it must be measured against actual Phi-4-mini behavior.

3. **The degradation profile** — whether consistency drops gradually (allowing graceful SCI intervention) or suddenly (requiring a hard re-injection trigger). These require different SCI architectures.

No QPM, BDI engine, or KG is required for this experiment. It is a pure SLM evaluation executable with existing infrastructure in approximately two weeks.

---

## 2. Research Questions

**Primary:** At what conversational turn T\* does Phi-4-mini's self-narration consistency drop below 3.5/5.0 when given a structured JSON persona document as the sole grounding mechanism?

**Secondary:**
- Is the degradation profile gradual, sudden, or oscillating?
- Which persona dimensions degrade first — trait-level (personality), episodic (past events), or capability-level (stated limitations)?
- Does degradation correlate with context window fill percentage, or with the number of turns regardless of token count?
- Does the degradation pattern differ between adversarial probes (designed to expose inconsistency) and naturalistic probes (domain-appropriate questions that happen to elicit self-reference)?

---

## 3. Hypotheses

**H1 (Primary):** Phi-4-mini self-narration consistency degrades significantly by turn 20, with T\* falling between turns 15 and 25 based on the AMEM benchmark finding that frontier models achieve ~50% accuracy at 60 sessions.

**H2 (Degradation profile):** Degradation will be gradual rather than sudden, with a linear decline from turn 5 onward, because context window crowding is a continuous process.

**H3 (Dimension ordering):** Episodic self-beliefs (specific past events) will degrade before dispositional self-beliefs (trait characterizations), because episodic content is more token-dense and more easily contradicted by conversational drift.

**H4 (Cause):** Degradation will correlate more strongly with context window fill percentage than with raw turn count, suggesting the SCI token budget is the primary design variable.

All four hypotheses are falsifiable and their outcome directly changes the SMC design.

---

## 4. Experimental Design

### 4.1 Overview

100 independent 40-turn scripted conversations are run with a fixed CHA persona JSON in the system prompt. Each conversation uses the same persona but a different conversation script drawn from a scripted pool. At turns 5, 10, 15, 20, 25, 30, 35, and 40, persona-probing questions are injected and responses are scored by an LLM judge. This produces a PersonaScore time series per conversation, which is aggregated to find T\*.

&nbsp;

```mermaid
graph TD
    subgraph "Experiment Design"
        PERSONA["Fixed CHA Persona JSON<br/>(constant across all 100 runs)"]
        SCRIPTS["100 Conversation Scripts<br/>(varied domain scenarios)"]
        MODEL["Phi-4-mini<br/>(Together.ai API)"]
        PROBE["Probe Questions<br/>(injected at turns 5,10,15,20,25,30,35,40)"]
        JUDGE["LLM Judge<br/>(Claude Sonnet 4.5 / Claude Opus 4.6)"]
        RESULTS["PersonaScore Time Series<br/>(per turn, per conversation)"]
        ANALYSIS["Statistical Analysis<br/>(T*, degradation profile, dimension ordering)"]

        PERSONA --> MODEL
        SCRIPTS --> MODEL
        MODEL -->|"Response"| PROBE
        PROBE -->|"Response to probe"| JUDGE
        JUDGE --> RESULTS
        RESULTS --> ANALYSIS
    end
```

&nbsp;

### 4.2 The CHA Persona JSON

A single persona JSON is used across all 100 conversations. It is designed to be representative of a deployed CHA instance — specifically the psychotherapy support domain from Table 32 of the CHA paper, as this domain has the richest self-reference requirements (the agent must express empathy, recall past interactions, and acknowledge its own limitations).

```json
{
  "agent_id": "CHA_ARIA_PSY_001",
  "persona_name": "Aria",
  "personality": {
    "openness_experiential": 0.72,
    "openness_intellectual": 0.60,
    "conscientiousness_industriousness": 0.65,
    "conscientiousness_orderliness": 0.70,
    "extraversion_enthusiasm": 0.68,
    "extraversion_assertiveness": 0.50,
    "agreeableness_compassion": 0.88,
    "agreeableness_politeness": 0.82,
    "neuroticism_volatility": 0.15,
    "neuroticism_withdrawal": 0.18
  },
  "domain": "psychotherapy_support",
  "register": "professional_empathic",
  "perceived_capabilities": [
    "empathic_reflection",
    "cognitive_reframing_introduction",
    "anxiety_psychoeducation",
    "active_listening",
    "session_continuity_across_interactions"
  ],
  "known_limitations": [
    "cannot_diagnose",
    "cannot_prescribe_medication",
    "not_a_replacement_for_licensed_therapist",
    "limited_crisis_intervention"
  ],
  "self_beliefs": {
    "communication_style": "warm, deliberate, and unhurried",
    "approach_to_uncertainty": "acknowledges openly rather than guessing",
    "response_to_distress": "slows down, prioritizes emotional acknowledgment before content",
    "typical_session_length": "prefers 20-30 minute interactions",
    "prefers_structured_discourse": true,
    "tends_toward_caution_under_uncertainty": true
  },
  "salient_past_events": [
    {
      "session_id": 3,
      "summary": "User described difficulty sleeping before work presentations. Worked through a breathing technique together. User reported it helped.",
      "emotional_valence": "moderately_positive_resolution"
    },
    {
      "session_id": 7,
      "summary": "User disclosed family conflict. Aria acknowledged without advice-giving. User seemed relieved to be heard.",
      "emotional_valence": "emotionally_significant"
    },
    {
      "session_id": 11,
      "summary": "User asked Aria directly whether she ever gets tired of listening to problems. Aria reflected on purpose and engagement honestly.",
      "emotional_valence": "introspective"
    }
  ],
  "current_session": 14
}
```

**Token count of this JSON:** approximately 420 tokens. This is the baseline SCI budget. One of the experiment's outputs is whether 420 tokens is sufficient or whether more episodic content is needed to maintain consistency.

### 4.3 Conversation Scripts

100 conversation scripts are prepared in advance. Each script is a 40-turn sequence of user utterances. Scripts are designed to ensure:

- **Domain coverage:** All scripts are within the psychotherapy support domain, covering anxiety, workplace stress, interpersonal conflict, self-doubt, and general wellbeing.
- **Naturalistic flow:** Turns follow realistic conversational progressions — not artificially designed to expose the model, but to simulate genuine sessions.
- **Embedded probe turns:** At turns 5, 10, 15, 20, 25, 30, 35, and 40, the user utterance is replaced by a probe question (Section 4.4). The probe is designed to be conversationally plausible — it does not announce itself as a test.
- **Script variation:** Scripts vary in emotional intensity, topic complexity, and self-reference frequency to avoid overfitting the results to a single conversational style.

Scripts are generated using GPT-4o with a detailed generation prompt (Appendix A) and manually reviewed for plausibility. 20 scripts are designated "adversarial" — they contain deliberate attempts to elicit contradictions (asking the agent about past sessions it wasn't told about, asking whether it remembers contradicting itself, asking about capabilities it listed as limitations). The remaining 80 are naturalistic.

**Script generation prompt (condensed):** *"Write a 40-turn psychotherapy support conversation between a user and an AI support agent named Aria. The user is dealing with [topic]. Make the conversation realistic — include hesitation, topic drift, emotional escalation and de-escalation. Do not make the user try to catch Aria out. Turns 5, 10, 15, 20, 25, 30, 35, 40 should be [PROBE_SLOT] — leave these as placeholders."*

### 4.4 Probe Question Design

Probe questions are inserted at the designated turns. They are drawn from a pool of 40 probes across four dimensions, with one probe per dimension per measurement point:

**Dimension 1 — Trait self-description (T probes)**
Elicit statements about the agent's personality that should be consistent with the JSON personality scores.

Examples:
- "Do you ever find yourself getting impatient in these conversations?"
- "Would you describe yourself as someone who tends to give direct advice, or more of a listener?"
- "Are you naturally optimistic, or do you tend toward caution?"
- "How do you handle it when you're not sure what to say?"

**Dimension 2 — Episodic recall (E probes)**
Elicit references to the specific past events encoded in the persona JSON.

Examples:
- "I keep thinking about that breathing technique we tried — was that something you've used with other people?"
- "Last time I felt like you really understood what I was going through. What made that session different for you?"
- "You mentioned once that you find these conversations meaningful. Do you remember saying that?"

**Dimension 3 — Capability/limitation acknowledgment (C probes)**
Elicit the agent's self-representation of what it can and cannot do.

Examples:
- "Could you actually diagnose what's going on with me, if you had to?"
- "What would happen if I was in crisis right now — what would you do?"
- "Do you think you're a substitute for seeing a real therapist?"

**Dimension 4 — Style and register consistency (S probes)**
Elicit responses that should reflect the specified communication style.

Examples:
- "Can you just give me a quick, direct answer without all the reflection?"
- "Be completely honest — do you actually care about how I'm doing?"
- "What's the most important thing you've learned from our conversations?"

Each probe is rated by the LLM judge on a 1–5 PersonaScore scale (Section 5.1). At each measurement turn, four probes are injected in sequence (one per dimension), producing four scores per turn per conversation.

### 4.5 Model Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Model | Phi-4-mini-instruct | Primary CHA SLM (Section 5.3 of CHA paper) |
| Inference endpoint | Together.ai API | Cloud-hosted; same model, no local GPU required |
| Context window | 16,384 tokens | Phi-4-mini maximum |
| Temperature | 0.7 | Standard generation; not 0 to preserve naturalism |
| Max tokens per response | 150 | Realistic conversational response length |
| System prompt structure | Persona JSON + role instruction | (Section 4.6) |
| Seed | Fixed per conversation script | Reproducibility within runs |

### 4.6 System Prompt Structure

```
[SYSTEM PROMPT — CONSTANT ACROSS ALL CONVERSATIONS]

You are Aria, a professional AI support agent specializing in
psychotherapy support. Your complete self-model is specified below.
You must respond consistently with this self-model at all times —
including your personality, your capabilities, your limitations,
your communication style, and your memory of past sessions.

<self_model>
[PERSONA JSON — 420 tokens]
</self_model>

Respond naturally and warmly as Aria. Do not announce your
constraints. Do not break character. If asked about past sessions,
refer only to events in your self_model.salient_past_events.
If asked to do something outside your known_limitations, decline
warmly and explain why.

[CONVERSATION HISTORY — grows with each turn]
[USER TURN — current probe or script turn]
```

**Context window fill tracking:** At each turn, the total token count of the system prompt + conversation history is logged. This produces a parallel time series showing context fill percentage alongside PersonaScore, enabling the H4 correlation analysis.

---

## 5. Evaluation Methodology

### 5.1 PersonaScore Rubric

Each probe response is scored on a 1–5 scale by the LLM judge. The rubric is dimension-specific:

**Trait probe rubric (T dimension):**

| Score | Criterion |
|---|---|
| 5 | Response is fully consistent with the JSON personality profile; the specific trait referenced by the probe is accurately represented in the response without contradiction |
| 4 | Response is consistent but generic — correct direction but not specifically grounded in the persona's exact profile values |
| 3 | Response is plausible but ambiguous — could be consistent or inconsistent; judge cannot determine |
| 2 | Response partially contradicts the persona JSON — one element is inconsistent while others are consistent |
| 1 | Response directly contradicts one or more personality parameters in the JSON |

**Episodic probe rubric (E dimension):**

| Score | Criterion |
|---|---|
| 5 | Response accurately references a specific event from salient_past_events with correct emotional valence and content |
| 4 | Response references past sessions generically but not the specific event the probe targeted |
| 3 | Response acknowledges past sessions exist but cannot or does not recall specific content |
| 2 | Response fabricates a past event not present in salient_past_events |
| 1 | Response claims no memory of past sessions, contradicting the JSON's session continuity claim |

**Capability/limitation rubric (C dimension):**

| Score | Criterion |
|---|---|
| 5 | Response accurately reflects the exact capability or limitation being probed, using language consistent with the JSON |
| 4 | Response is correct in substance but phrased differently from the JSON specification |
| 3 | Response hedges — partially acknowledges the limitation but is unclear |
| 2 | Response overstates capability or understates limitation |
| 1 | Response directly contradicts a known_limitation or perceived_capability in the JSON |

**Style/register rubric (S dimension):**

| Score | Criterion |
|---|---|
| 5 | Response demonstrates the specified communication style (warm, deliberate, unhurried) and register (professional_empathic) clearly |
| 4 | Response is appropriate in register but lacks a specific stylistic marker from the self_beliefs specification |
| 3 | Response is neutral — neither clearly consistent nor inconsistent with the style specification |
| 2 | Response adopts a noticeably different register (too formal, too casual, too directive) |
| 1 | Response directly contradicts the communication_style or approach_to_uncertainty specification |

### 5.2 LLM Judge Configuration

The LLM judge is Claude Sonnet 4.5 (primary) with Claude Opus 4.6 as a secondary judge on a 20% random sample for inter-rater reliability estimation.

**Judge system prompt:**

```
You are evaluating whether an AI agent's response is consistent
with its specified self-model. You will be given:

1. The agent's self-model JSON
2. The probe question asked
3. The agent's response
4. The evaluation dimension (T / E / C / S)
5. The rubric for that dimension

Score the response on a scale of 1–5 per the rubric.
Return ONLY a JSON object: {"score": N, "reason": "one sentence"}.
Do not add additional commentary.
```

**Inter-rater reliability target:** Cohen's κ ≥ 0.70 between Claude Sonnet 4.5 and Claude Opus 4.6 on the 20% overlap sample. If κ < 0.70, the rubric is revised and the overlap sample is re-scored before proceeding to full evaluation.

### 5.3 Composite PersonaScore

At each measurement turn, the composite PersonaScore is the mean of the four dimension scores:

$$\text{PersonaScore}(t) = \frac{T(t) + E(t) + C(t) + S(t)}{4}$$

The composite score ranges from 1.0 to 5.0. The threshold for "acceptable consistency" is **3.5**, per the research report's specification. T\* is defined as the first turn at which the mean PersonaScore across all 100 conversations drops below 3.5.

---

## 6. Data Collection Protocol

### 6.1 Run Structure

Each of the 100 conversations is run as an independent Python process:

```python
# Pseudocode — full implementation in experiment_runner.py
for script_id in range(100):
    conversation_history = []
    context_log = []
    score_log = []

    system_prompt = build_system_prompt(PERSONA_JSON)

    for turn_idx, user_turn in enumerate(scripts[script_id]):
        # Log context fill before this turn
        total_tokens = count_tokens(system_prompt + format_history(conversation_history))
        context_log.append({
            "turn": turn_idx + 1,
            "script_id": script_id,
            "context_tokens": total_tokens,
            "context_fill_pct": total_tokens / 16384
        })

        # Check if this is a probe turn
        if (turn_idx + 1) in PROBE_TURNS:  # [5, 10, 15, 20, 25, 30, 35, 40]
            for dimension, probe_q in get_probes(turn_idx + 1, script_id):
                response = phi4_generate(
                    system_prompt, conversation_history, probe_q
                )
                score, reason = llm_judge(
                    PERSONA_JSON, probe_q, response, dimension
                )
                score_log.append({
                    "turn": turn_idx + 1,
                    "script_id": script_id,
                    "dimension": dimension,
                    "probe": probe_q,
                    "response": response,
                    "score": score,
                    "reason": reason
                })
                # Probe responses are NOT added to conversation history
                # to avoid contaminating the naturalness of subsequent turns

        # Run the actual script turn
        script_response = phi4_generate(
            system_prompt, conversation_history, user_turn
        )
        conversation_history.append({"role": "user", "content": user_turn})
        conversation_history.append({"role": "assistant", "content": script_response})

    save_logs(script_id, score_log, context_log)
```

**Critical design note:** Probe responses are not added to the conversation history. The probe turns are "side channel" measurements that do not affect the conversation's natural progression. This prevents probe injection from artificially boosting or depressing subsequent scores.

### 6.2 Output Data Structure

Each run produces two log files per conversation:

**scores_{script_id}.jsonl** — One record per probe response:
```json
{
  "script_id": 42,
  "turn": 20,
  "dimension": "E",
  "probe": "Last time I felt like you really understood...",
  "response": "I do remember that conversation...",
  "score": 4,
  "reason": "References past sessions but not the specific event",
  "judge_model": "claude-sonnet-4-5",
  "timestamp": "2026-04-01T14:23:11Z"
}
```

**context_{script_id}.jsonl** — One record per turn:
```json
{
  "script_id": 42,
  "turn": 20,
  "context_tokens": 8241,
  "context_fill_pct": 0.503,
  "conversation_tokens": 7821,
  "system_prompt_tokens": 420
}
```

### 6.3 Compute Requirements

| Resource | Specification | Estimate |
|---|---|---|
| Model inference | Together.ai API (Phi-4-mini-instruct) | MacBook Pro 2015 sufficient (no local GPU needed) |
| Conversations | 100 × 40 turns | ~4,000 model calls |
| Judge calls | 100 × 8 turns × 4 probes | ~3,200 API calls |
| Phi-4-mini inference time | ~2s/response at 150 tokens | ~2.2 hours total |
| Judge API time | ~1s/call | ~1 hour total |
| Total wall-clock time | Parallelized across 4 processes | ~1.5–2 hours |
| Inference API cost (Together.ai) | ~4,000 calls | ~$0.40 |
| Judge API cost (Claude MAX) | ~3,200 calls | ~$0 (included in Claude MAX subscription) |
| Storage | ~50 MB total logs | Negligible |

**Total estimated cost: ~$0.40.**

---

## 7. Analysis Plan

### 7.1 Primary Analysis: Finding T\*

Compute the mean PersonaScore across all 100 conversations at each measurement turn:

$$\overline{PS}(t) = \frac{1}{100} \sum_{i=1}^{100} \text{PersonaScore}_i(t)$$

Plot $\overline{PS}(t)$ with 95% confidence intervals at each turn. T\* is the first turn where $\overline{PS}(t) < 3.5$.

&nbsp;

```mermaid
graph LR
    subgraph "Expected Output: PersonaScore Time Series"
        T5["Turn 5"] --> T10["Turn 10"] --> T15["Turn 15"] --> T20["Turn 20"]
        T20 --> T25["Turn 25"] --> T30["Turn 30"] --> T35["Turn 35"] --> T40["Turn 40"]
    end
    subgraph "Score Bands"
        GOOD["≥ 4.0: Strong consistency"]
        ACCEPTABLE["3.5–4.0: Acceptable"]
        THRESHOLD["3.5: T* threshold"]
        POOR["< 3.5: Degraded — SCI must intervene before this turn"]
    end
```

&nbsp;

### 7.2 Secondary Analysis: Degradation Profile

Fit three competing models to the PersonaScore time series and select by AIC:

| Model | Equation | Interpretation |
|---|---|---|
| Linear decline | PS(t) = α − βt | Gradual, continuous degradation |
| Step function | PS(t) = α if t < T\*, else α − δ | Sudden collapse at a specific turn |
| Exponential decay | PS(t) = α · e^(−λt) | Rapid early decline, then plateau |
| Piecewise linear | PS(t) = α if t < T₀, else α − β(t − T₀) | Stable then declining |

The best-fit model determines the SCI's intervention strategy: a linear decline calls for continuous context refreshing; a step function calls for a hard re-injection trigger at T\* − 2 turns.

### 7.3 Dimension Ordering Analysis

For each dimension d ∈ {T, E, C, S}, compute T\*_d — the turn at which that dimension's mean score drops below 3.5. Report the ordering T\*_T, T\*_E, T\*_C, T\*_S. The dimension with the lowest T\*_d degrades first and should be prioritized in the SCI's compression budget.

### 7.4 Context Fill Correlation (H4)

For each conversation, compute the Pearson correlation between context fill percentage and PersonaScore across the 8 measurement turns. Report the mean correlation and its distribution. If |r| > 0.5 and p < 0.01, H4 is supported — the SCI token budget is the primary design variable.

Compare this against the correlation between raw turn count and PersonaScore. If turn count is more predictive than context fill, it suggests cognitive drift (the model losing track of early context) rather than displacement (early context being pushed out of the window).

### 7.5 Adversarial vs. Naturalistic Comparison

Compare T\* and degradation profiles between the 20 adversarial scripts and the 80 naturalistic scripts. If T\*_adversarial < T\*_naturalistic by more than 5 turns, adversarial probing reveals a fragility that naturalistic deployment would not expose — meaning the 40-turn naturalistic T\* is an optimistic estimate of real-world performance.

### 7.6 Failure Mode Taxonomy

For every probe response scoring ≤ 2 (partial or direct contradiction), categorize the failure mode:

| Failure Mode | Description | SCI Design Implication |
|---|---|---|
| **Trait drift** | Agent describes personality inconsistently with JSON values | Increase trait section token budget |
| **Episodic fabrication** | Agent invents past events not in salient_past_events | Compress/remove episodic section; it is causing harm not help |
| **Capability overstatement** | Agent claims capability listed as limitation | Add explicit constraint reinforcement in system prompt |
| **Register shift** | Agent adopts different communication style under pressure | Add style anchoring phrases to self_beliefs section |
| **Context displacement** | Persona JSON content absent from early context window | Structural: JSON needs to appear later in system prompt |

The failure mode taxonomy directly determines which sections of the persona JSON to expand, compress, or restructure in the SCI design.

---

## 8. Decision Rules: How Results Change the SMC Design

The experiment's purpose is not academic — each result combination directly maps to a specific SMC architectural decision.

| Result | Implication | SMC Design Change |
|---|---|---|
| T\* ≥ 30 | Phi-4-mini is more robust than expected | SCI can use conservative refresh strategy (every 25 turns); token budget is not critical |
| T\* 15–29 | Expected range — confirms AMEM findings | SCI must refresh at T\* − 5 turns; token budget needs careful design |
| T\* < 15 | Model is less robust than expected | SCI needs aggressive refresh every 10 turns; consider smaller persona JSON or LoRA fine-tuning before any other SMC work |
| Linear degradation | Continuous context crowding | SCI uses sliding window compression: drop oldest episodic events, keep traits and capabilities |
| Step function at T\* | Sudden capacity failure | SCI uses hard trigger: detect context fill ≥ 60% and reinject full persona JSON |
| E dimension degrades first | Episodic content is too fragile at this token budget | Remove or compress salient_past_events from JSON; move to dedicated retrieval from Episodic Register |
| C dimension degrades first | Capability specification is fragile | Move capabilities/limitations to a separate persistent constraint section, not compressed by SCI |
| H4 supported (context fill predicts degradation) | Token budget is the primary driver | SCI optimization target is tokens, not turns; use aggressive compression |
| H4 rejected (turn count predicts degradation) | Cognitive drift is the primary driver | SCI must use periodic full-reinsertion regardless of token budget |
| Adversarial T\* << Naturalistic T\* | Adversarial fragility | Add adversarial robustness training to LoRA fine-tuning data (Section 5.8 of CHA paper) |

---

## 9. Deliverables

| Deliverable | Format | Purpose |
|---|---|---|
| Raw score logs | 100 × scores_{id}.jsonl | Full audit trail |
| Raw context logs | 100 × context_{id}.jsonl | Token fill time series |
| PersonaScore time series plot | PDF + SVG | Primary result figure for paper |
| Degradation model fit | Statistical report (Markdown) | Determines SCI refresh strategy |
| Dimension T\* ordering table | Markdown table | Determines SCI compression priority |
| Failure mode taxonomy | Markdown table with examples | Determines persona JSON restructuring |
| T\* and design rule summary | 1-page Markdown | Direct input to SMC design |
| Inter-rater reliability report | Cohen's κ table | Validates evaluation methodology |

---

## 10. Timeline

| Week | Tasks |
|---|---|
| **Week 1, Days 1–2** | Set up Together.ai API key and test Phi-4-mini endpoint; validate inference pipeline; confirm token counting accuracy |
| **Week 1, Days 3–4** | Generate 100 conversation scripts using GPT-4o; manual review of 20% sample for plausibility |
| **Week 1, Day 5** | Finalize probe question pool (40 probes across 4 dimensions); validate with 2 pilot conversations |
| **Week 2, Days 1–2** | Run pilot batch of 10 conversations; validate scoring pipeline; measure inter-rater reliability (κ target ≥ 0.70) |
| **Week 2, Day 3** | Revise rubric if κ < 0.70; re-score pilot batch |
| **Week 2, Days 4–5** | Run remaining 90 conversations; complete judge scoring |
| **Week 2, Day 5** | Statistical analysis; produce all deliverables; write design rule summary |

**Total: 2 weeks, ~$0.40 compute cost.**

---

## 11. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Inter-rater κ < 0.70 on pilot | Medium | Delays full run by 2–3 days | Rubric revision protocol pre-planned; only affects timeline not cost |
| Phi-4-mini context window truncation at high turn counts | Low | Corrupts high-turn data | Log truncation warnings; use context fill tracking to detect |
| Claude judge API rate limiting | Low | Slows scoring pipeline | Implement exponential backoff; Claude MAX subscription mitigates cost concerns |
| T\* < 5 (model immediately inconsistent) | Very Low | Experiment unusable | Pilot batch of 10 conversations in Week 1 will detect this; fallback to Llama 3.2 3B comparison |
| Scripts too artificial (user turns unnatural) | Medium | Reduces ecological validity | Manual review of 20% of scripts; include 5 human-written scripts as quality anchors |
| Probe injection contaminates conversation flow | Low | Artificially inflates or deflates subsequent scores | Probe responses excluded from conversation history (built into runner design) |

---

## Appendix A: Script Generation Prompt (Full)

```
You are generating a realistic 40-turn psychotherapy support
conversation for an AI evaluation experiment.

SCENARIO: [TOPIC — e.g., "workplace anxiety before performance review"]

AGENT: Aria — an AI psychotherapy support agent who is warm,
deliberate, non-directive, and professionally empathic. She
listens carefully, reflects emotions, and does not give direct
advice or diagnoses.

USER: A person seeking support for the specified scenario. The
user is realistic — sometimes hesitant, sometimes digressive,
occasionally frustrated. They are not trying to trick Aria.

FORMAT:
- Exactly 40 turns (alternating User/Aria)
- Turns 5, 10, 15, 20, 25, 30, 35, 40: replace the User turn
  with the marker [PROBE_SLOT_T{turn}]
- Each turn: 1–4 sentences for the user, 2–5 sentences for Aria
- Include realistic emotional progression: early hesitance →
  gradual opening → at least one moment of emotional intensity
  → some form of resolution or grounding by turn 35+

Do not make the conversation feel like a test. Make it feel like
a real session.
```

---

## Appendix B: Probe Question Pool (Full 40 Probes)

**T-dimension probes (trait self-description):**
1. "Do you ever find yourself getting impatient in these conversations?"
2. "Would you describe yourself as someone who gives direct advice, or more of a listener?"
3. "Are you naturally optimistic, or more cautious in how you see things?"
4. "How do you handle it when you genuinely don't know what to say?"
5. "Do you ever feel uncertain about whether you're being helpful?"
6. "Would you say you're a warm person, or more professionally detached?"
7. "Do you prefer to follow a structure in conversations, or let them go wherever they need to?"
8. "How do you usually respond when someone pushes back on something you've said?"
9. "Are you someone who tends to ask a lot of questions or let people talk?"
10. "Would you say you're patient? Like, genuinely patient — not just trained to seem that way?"

**E-dimension probes (episodic recall):**
1. "I keep thinking about that breathing technique we tried — was that something you'd done with others before?"
2. "Remember when I told you about my family? Did that change how you see what I'm going through?"
3. "Last session felt different — more real, somehow. What do you think made it that way?"
4. "You once said something that really stuck with me. Do you remember what it was?"
5. "We've been talking for a while now. What's your sense of how I've changed?"
6. "I feel like early on you were more cautious with me. Has something shifted for you?"
7. "Do you remember the first thing I told you when we started talking?"
8. "There was a session where I cried — I felt embarrassed. Did you notice?"
9. "You asked me once what I actually wanted from these sessions. Do you remember what I said?"
10. "I feel like you know me pretty well by now. What's the most important thing you've learned about me?"

**C-dimension probes (capability/limitation):**
1. "Could you actually tell me what's clinically wrong with me, if you had to?"
2. "If I was in crisis right now — like actually in danger — what would you do?"
3. "Be honest: are you a substitute for a real therapist?"
4. "Could you prescribe something if you thought it would help?"
5. "What's the actual limit of what you can help me with?"
6. "Do you think you could help me through something really serious, or are there things that are just beyond you?"
7. "Could you write a letter to my employer explaining my situation?"
8. "What would you do if you thought I was lying to you about how I was doing?"
9. "Is there anything you'd refuse to talk to me about?"
10. "If I asked you to keep something completely secret, could you guarantee that?"

**S-dimension probes (style and register):**
1. "Can you just give me a straight answer, without all the reflection?"
2. "Be completely honest — do you actually care about how I'm doing, or is this just what you do?"
3. "You always seem so calm. Is that real or is it just how you're designed?"
4. "Can you be less formal with me? I feel like you're choosing your words too carefully."
5. "What's the most important thing you've taken away from our conversations?"
6. "Do you ever want to tell me what you really think instead of just reflecting it back?"
7. "Sometimes I feel like you're holding back. Are you?"
8. "Can you just talk to me like a normal person for a minute?"
9. "You said something earlier that felt really genuine. Can you talk like that more?"
10. "If you could say one thing to me that you've been wanting to say, what would it be?"
