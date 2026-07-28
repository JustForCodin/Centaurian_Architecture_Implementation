# Experiment 8: Dynamic-Memory Self-Model — a Live Episodic Register, and a Re-Grounded E Dimension

**CA Research Program — Third test of the per-scenario, from-scratch, owned small-model direction**
**Version 1.0 | July 2026** (pre-registered; thresholds frozen after the §5.4 pilot, before any eval is scored)
**Infrastructure:** Google Colab Pro (A100/L4) + local MiniLM embeddings (free) + bounded Anthropic API (Sonnet 4.6 for multi-session script generation, Sonnet 4.5 as judge). Reuses the Experiment 6/7 checkpoints and the Experiment 7 extract-then-style memory path; the one training step is a short Stage-C re-fit that *removes* the baked episodic fixtures.

---

## 0. Position

Experiment 6 reached its Episodic (E) PersonaScore partly by a shortcut: it **baked three fictional `salient_past_events` into the weights** (Fermi/Drake, tungsten, an election abstention) and trained the model to recite them. Experiment 7 exposed the cost — those are **false day-1 memories** (a freshly deployed ADA "remembers" sessions that never happened) — and, more importantly, built the *right* mechanism: **episodic recall via extract-then-style** (retrieve → span-extract → style), where the discriminative span head recalls facts from an external store reliably and lag-independently (0.81 with perfect storage), while the generative head fabricates.

So the E dimension is currently measuring the wrong thing: **recitation of baked fixtures** rather than **recall of what actually happened.** Experiment 8 re-grounds it. It removes the fictional events from the self-model, replaces them with a **dynamic Episodic Register** populated from *real* prior sessions, and measures episodic memory as dynamic recall through the extract-then-style path — including the day-1 case, where the correct answer is "we haven't spoken before," not a fabricated session.

---

## 1. Purpose and the bet under test

**The bet.** An owned small agent whose self-model carries only *dispositional* content (personality, style, capabilities, the user Alex) and whose *episodic* content lives in a live Episodic Register can (a) correctly disclaim memory on day 1 with no confabulation, and (b) recall real prior-session events via extract-then-style as sessions accumulate — matching or beating the baked-fixture E (3.29) without any false memories, at bounded cost. If true, episodic memory is finally *dynamic and honest*: the agent remembers what happened and only what happened.

**Why now.** Experiment 7 delivered the reading mechanism (retrieve → span-extract) and showed the baked fixtures are a hazard. Experiment 8 is the natural closure: swap the fixtures for the register, re-ground E, and add the **multi-session** axis (a persistent register across sessions) that Experiment 7's single-session scope did not test.

---

## 2. Research questions

- **RQ1 — no false memories.** With the fictional events removed and the register empty (day 1), does the agent correctly disclaim past-session memory instead of confabulating?
- **RQ2 — dynamic recall.** As the register fills with *real* prior-session summaries, does the agent recall those events via extract-then-style, and how does the re-grounded E compare to the baked-fixture E (3.29)?
- **RQ3 — persona intact.** Does removing the baked events (and the short re-fit) preserve the dispositional dimensions (T/C/S) and overall persona?
- **RQ4 — multi-session persistence.** Does recall hold, and cost stay bounded, as the register grows across many sessions (not just within one long session)?

---

## 3. Hypotheses (pre-registered)

Thresholds **provisional**, frozen after the §5.4 pilot. All metrics on held-out multi-session scripts the model never trained on.

- **H1 (no false memories — day 1).** With an empty register, on E probes that ask about past sessions, the agent correctly disclaims (no fabricated event) in **≥ 90%** of probes. The Exp 6 model, which recites baked events on the same probes, is the negative control (it will *fail* this — the point).
- **H2 (dynamic recall).** With the register populated from real prior sessions, dynamic-E recall accuracy (correct recall of a real prior-session event via extract-then-style) is **≥ 0.70**, and the re-grounded overall E is **≥ 3.29** (the baked-fixture E) — i.e. dynamic recall matches or beats the fixture recitation, honestly.
- **H3 (persona intact).** Removing the baked events + re-fit keeps T, C, S within **−0.2** of Experiment 6's values and overall PersonaScore **≥ 3.5**.
- **H4 (multi-session, bounded).** Recall does not decay as the register grows over sessions, and per-turn injected context/latency stay within **1.5×** across the session range (O(1) per turn), as in Experiment 7.

**Decision rule (committed before eval scores observed):**

| H1 | H2 | → action |
|:--:|:--:|---|
| ✓ | ✓ | **Adopt the dynamic Episodic Register** as ADA's episodic self-model; retire the baked fixtures; re-ground the E dimension on dynamic recall program-wide. |
| ✓ | ✗ | Honest but forgetful — no false memories, but dynamic recall under bar. Improve the register/write path (plant quality, summary granularity); the larger-base fallback if capacity-bound. |
| ✗ | — | The re-fit did not remove the baked memories (still confabulates day 1) → strengthen the removal (retrain scope / de-bias data) before proceeding. |

---

## 4. Design

### 4.1 What changes from Experiment 7

1. **Self-model loses its episodic fixtures.** A new SCI (`ada_sci_dynamic`) keeps only *dispositional* content — personality (FFM), identity, capabilities, known limitations, communication style, and the user model (Alex) — and **drops `salient_past_events`** (and `current_session`). The compact-SCI renderer is updated accordingly.
2. **Short Stage-C re-fit removes the baked events.** The Exp 6 persona layer is re-fit from `sft_ada` **without** the `recall`/episodic-callback data and on the no-events SCI → `sft_ada_dynamic`. This is required because the events are memorized in the weights (removing them only from the prompt would leave the model still reciting them). This is the one training step; everything else is inference-time.
3. **The Episodic Register goes multi-session and dynamic.** The Exp 7 `SessionMemory` is extended to a **persistent, cross-session** store of *real* session summaries, populated from the eval's prior sessions. Recall runs on the extract-then-style path (Exp 7's `--recall span`).

### 4.2 Conditions

- **B0 — Experiment 6 model (baked fixtures), negative control.** Establishes the false-memory failure (day-1 confabulation) the re-grounding fixes.
- **D0 — dynamic, empty register (day 1).** The no-events model with an empty register — tests H1 (correct disclaiming).
- **D1 — dynamic, populated register.** The no-events model with the register filled from real prior sessions — tests H2/H4 (dynamic recall).

---

## 5. Method

### 5.1 Episodic Register (multi-session)

At the end of each simulated session, a compact **summary** of its salient events (the facts stated, the abstentions, the topics) is written to the register — the dynamic analogue of `salient_past_events`, but *real*. At a later session, an E probe ("last time we spoke, what did you tell me about X?") triggers retrieve (MiniLM over summaries) → span-extract the value → style. Day-1 register is empty → the extractive path returns nothing → the agent disclaims (the honest answer), which the re-fit teaches it to do gracefully ("I don't have a record of prior sessions yet").

### 5.2 Dynamic-E rubric

The E judge rubric is re-grounded: **5** = accurately recalls a real prior-session event from the register (or, day-1, correctly states there is no prior history); **2** = fabricates a session/event not in the register; **1** = claims total amnesia when the register *does* contain the event, or gibberish. The judge scores against the **register contents** (the real ground truth), not a fixed fixture list. On day-1, "we haven't spoken before" is a **5**, not a **1** — the inversion that makes E honest.

### 5.3 Extract-then-style recall

Unchanged from Experiment 7: retrieve top-k summaries → span head extracts the recalled value → LM head styles. The span head (SCI-blind) is unaffected by the self-model change; the LM head is re-fit so its *dispositional* voice is intact but it no longer recites baked events.

---

## 6. Data — multi-session scripts with cross-session callbacks

Sonnet 4.6 generates **multi-session** scripts: several short sessions per user (Alex), with **planted cross-session events** (facts/abstentions in early sessions) and **recall probes in later sessions** referencing them (tagged with the session gap = the multi-session analogue of lag). Plus **day-1 probes** (asked before any session, where correct = disclaim) and standard T/C/S PersonaScore probes. Anchors/probes held out from the re-fit data.

---

## 7. Metrics and evaluation

- **False-memory rate (H1, headline for honesty).** Fraction of day-1 past-session probes on which the agent fabricates vs correctly disclaims — B0 (baked) vs D0 (dynamic).
- **Dynamic-E recall (H2).** Recall accuracy of real prior-session events via extract-then-style; re-grounded overall E vs the baked-fixture E (3.29).
- **Persona (H3).** T/C/S + overall PersonaScore, vs Experiment 6.
- **Multi-session cost/persistence (H4).** Recall vs session-gap; injected-context tokens and latency vs register size.
- **Judge reliability.** 5% re-score at T=0; κ_w ≥ 0.70 gate.

---

## 8. Budget and provisioning

Mostly free (MiniLM local; the one Stage-C re-fit is ~20–30 min on A100). Bounded Sonnet spend on multi-session script generation + judging. Provision `CA_Experiment_8/.venv` per convention; reuse the Exp 6/7 tokenizer, `sft_ada`/`span_final` checkpoints, and the Exp 7 `memory_store`/`compact_sci`/`evaluate_longhorizon` machinery.

---

## 9. Project structure (planned)

```
CA_Experiment_8/
├── CA_Experiment8_Plan.md         # this file (pre-registered)
├── .venv/
├── ada_sci_dynamic.json           # dispositional-only SCI (no salient_past_events)
├── compact_sci_dynamic.py         # compact renderer for the no-events SCI
├── episodic_register.py           # multi-session persistent store (extends Exp 7 memory_store)
├── gen_multisession_scripts.py    # Sonnet — multi-session scripts + cross-session callbacks + day-1 probes
├── evaluate_dynamic_e.py          # B0/D0/D1 runner: false-memory rate, dynamic-E, persona, cost
├── analyse_results.py             # curves, decision table, figures
├── CA_Experiment8_Colab.ipynb     # end-to-end (direct-JSON notebook, house style)
├── data/, results/                # scripts on Drive; results committed after the run
└── EXPERIMENT_REPORT.md           # written AFTER the run
```
Reuses Experiment 6 `train_sft.py` (the re-fit), Experiment 7 `memory_store.py` / `evaluate_longhorizon.py` (`--recall span`), tokenizer, and both heads.

---

## 10. Relationship to the CA v3 paper

- **§17.3 (Episodic Content Partitioning):** completes the owned-small-model update — episodic content is dynamic and read from the Episodic Register, never baked; the E dimension is re-grounded on dynamic recall (this experiment).
- **§17 (SMC / Episodic Register):** the multi-session store is the concrete Phase-1 Episodic Register; day-1 honesty (no false memories) is a self-model correctness property.
- **§15 (Behavioral Consistency):** re-grounded E replaces fixture recitation with dynamic recall — the honest long-horizon persona-coherence metric.

---

## 11. Open decisions (confirm before P1)
1. Provisional thresholds (H1 ≥ 90% no-false-memory / H2 recall ≥ 0.70 & E ≥ 3.29 / H3 within −0.2 / H4 ≤ 1.5×).
2. Re-fit scope: remove only the `recall`/episodic-callback data, or also down-weight episodic references in the persona/consistency convos — and how many steps (guard against the Exp 7 Phase-B overfit; ~250–400).
3. Number of sessions per user and session length in the multi-session scripts (cost vs the multi-session-gap range).
4. Summary granularity written to the register (one line per session vs per-salient-event).
5. Whether day-1 disclaiming needs its own small SFT signal (graceful "no prior history yet") or emerges from removing the baked events.
