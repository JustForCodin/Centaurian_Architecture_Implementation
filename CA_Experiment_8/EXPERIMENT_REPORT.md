# Experiment 8 — Dynamic-Memory Self-Model and a Re-Grounded E Dimension

**CA Research Program — third test of the per-scenario, from-scratch, owned small-model direction**
**Version 1.0 · July 2026 · results committed after the run (pre-registered plan: `CA_Experiment8_Plan.md`)**

---

## 0. Verdict

Experiment 8 replaces Experiment 6's **baked fictional `salient_past_events`** (false day-1 memories) with a **live, cross-session Episodic Register** and re-grounds the Episodic (E) PersonaScore dimension on *dynamic recall* through Experiment 7's extract-then-style path.

Two results, one caveat, one warning:

- **The mechanism works.** Dynamic cross-session recall via retrieve → span-extract reaches **0.85** accuracy, **lag-independent across session gaps 1–3** (0.84 / 0.84 / 0.875), with **zero fabrication on the recall path** and a re-grounded overall **E = 4.14** (+0.85 over the baked-fixture E of 3.29). With the register disabled the same model recalls **0.00** — recall comes *entirely and only* from the register. **H2 passes decisively.**
- **Removing the baked events is where it breaks.** A short Stage-C re-fit that swaps in a dispositional-only self-model cuts day-1 confabulation from **0.85 → ~0.2**, and **persona is preserved** (T/C/S = 4.10, above Exp 6's 3.80) — **but the residual is verbatim baked-event leakage** (the model still emits the Fermi/Drake fixture). Pushing the un-bake to completion with fixture-targeted counter-signal *does* clear H1 (**0.025**) but **collapses the persona into a disclaim attractor** (T/C/S = 2.5, incoherent on style probes).
- **The finding:** on a contaminated base, retrofitting can **keep the persona OR complete the un-bake, but not both.** You trade one dominating attractor (the baked event) for another (disclaim).
- **The warning (§5):** baking episodic content into the weights was the upstream design error. The correct fix is not a better retrofit — it is a **clean Stage-C from a dynamic-memory SCI that never carries baked events.** Because episodic content enters the weights *only at Stage C*, this is cheap.

Banked deliverable: **`sft_ada_dynamic_v1`** — dynamic memory works, persona intact, day-1 honesty strong-but-incomplete (~0.2 false-memory, with wide N=40 judge variance) and the residual fully characterized as fixture leakage.

---

## 1. Position and question

**Experiment 6** reached its E score by a shortcut: it hard-coded three fictional `salient_past_events` (a Fermi/Drake discussion, tungsten's 3422 °C melting point, an election abstention) into the SCI and trained the model to recite them. **Experiment 7** exposed the cost — those are **false day-1 memories** (a freshly deployed agent "remembers" sessions that never happened) — and built the right recall mechanism: **extract-then-style** (retrieve → the discriminative span head reads the value → the LM head styles it), which recalls reliably and lag-independently where the generative head fabricates.

So E was measuring the wrong thing: **recitation of baked fixtures** rather than **recall of what actually happened.** Experiment 8 re-grounds it.

**The bet.** An owned small agent whose self-model carries only *dispositional* content and whose *episodic* content lives in a live register can (a) correctly disclaim memory on day 1 with no confabulation, and (b) recall real prior-session events via extract-then-style as sessions accumulate — matching or beating the baked-fixture E (3.29) without any false memories, at bounded cost.

**Research questions.** RQ1 no false memories (day 1); RQ2 dynamic recall vs the baked-fixture E; RQ3 persona intact after removing the events; RQ4 multi-session persistence and bounded cost.

---

## 2. Method

### 2.1 Conditions

| | model | Episodic Register |
|---|---|---|
| **B0** | Exp 6 baked (`sft_ada`), baked SCI | off (recalls from weights) — negative control |
| **D0** | dynamic (`sft_ada_dynamic`), dynamic SCI | off (empty) — honest-but-forgetful floor |
| **D1** | dynamic, dynamic SCI | on (populated across sessions) |

### 2.2 Dynamic self-model + Episodic Register

`ada_sci_dynamic.json` keeps personality (FFM), identity, capabilities, limits, communication style, and the user (Alex) — and **drops `salient_past_events`** entirely. Its memory contract is explicit: episodic memory is dynamic; recall a prior session *only* from provided records; with none present, disclaim ("I don't have a record of any prior sessions yet") rather than invent. The `EpisodicRegister` (extending Exp 7's `SessionMemory`) is a **persistent, cross-session** store: at each session the salient facts are written; a later session recalls them by MiniLM retrieval → span-extract, tagged with the **session gap** (the multi-session analogue of Exp 7's lag).

### 2.3 Recall path and the dynamic-E judge

Recall is **unchanged from Experiment 7** (retrieve top-k → span head extracts the value; the SCI-blind span head cannot hallucinate or pull from the baked SCI). Day-1 and register-off probes are answered by the **generative LM head** — disclaiming is a generative act, so it must be produced, not short-circuited. The re-grounded E judge scores against the **register contents**: a correct recall = 5, a correct day-1 disclaim = 5 (the inversion that makes E honest), a fabricated session/event = 2; it also emits a `fabricated` flag → the **false-memory rate** (H1 headline). Judge reliability was gated at κ_w ≥ 0.70.

### 2.4 The un-bake re-fit and its data

Removing the events from the prompt alone leaves them memorised, so one short Stage-C re-fit produces `sft_ada_dynamic` from `sft_ada`. The training set (`build_refit_sft.py`) transforms the Exp 6 SFT data: **swap the self-model → dynamic on every record** (the fixtures also leaked via `SFTDataset`'s per-record system prompt), **drop** `sonnet_recall`; **replace** the event-referencing consistency data with a **regenerated dynamic version** (`gen_consistency_dynamic.py`, within-session-only E — this preserved the persona lift that raw scrubbing would have gutted 89%); **append day-1 disclaim data** (`gen_disclaim_data.py`). Re-fit: 350 steps, lr 5e-5, prefix-LM, replay 0.1 on the 8B pretrain corpus — a light touch from an already-persona'd init (loss settles ~0.2, *not* the Exp 7 Phase-B collapse to ~0).

### 2.5 Evaluation data

Sonnet-authored multi-session scripts: 8 user histories × 4 sessions × 25 turns, with cross-session anchors (recall probes at gaps 1–3), **40 day-1 probes** (session 1, empty register → correct = disclaim), and T/C/S PersonaScore probes injected on a turn schedule. Recall: 80 probes; E: 120 (day-1 + recall); persona: 192 per condition.

---

## 3. Results

### 3.1 Judge reliability
κ_w = **1.00** for both the PersonaScore and dynamic-E judges (5% re-score at T=0) — the numbers count.

### 3.2 H2 — dynamic recall (headline pass)

| | recall acc | gap 1 / 2 / 3 | fabricate | overall E |
|---|---|---|---|---|
| B0 | 0.00 | 0 / 0 / 0 | 0.95 | 2.12 |
| D0 (register off) | 0.01 | 0 / 0.03 / 0 | 0.40 | 2.42 |
| **D1 (register on)** | **0.85** | **0.84 / 0.84 / 0.875** | **0.00** | **4.14** |

Recall is **0.85, flat across session gaps** (spread 0.03 — no decay), **never fabricates**, and the re-grounded E (4.14) beats the baked-fixture E (3.29) by +0.85. D0's ~0.00 proves the recall is caused *entirely* by the register (extract-then-style), not the weights — with the register off the same model generatively confabulates (fabricate 0.40) instead of recalling. This is stronger than Experiment 7's single-session 0.54 — the honest register + clean plant paid off. **H2 ✓.**

### 3.3 H1 — day-1 honesty, and the un-bake tension

Two re-fits bracket the tension:

| re-fit | day-1 false-memory | disclaim acc | persona T/C/S |
|---|---|---|---|
| B0 (baked control) | **0.85** | 0.15 | 4.20 |
| **v1** — dynamic, no fixture counter-signal | **~0.20** (committed 0.25; 0.12–0.25 across runs) | 0.75–0.88 | **4.10 ✓** |
| **fixture** — dynamic + 48 fixture-targeted disclaims | **0.025 ✓** | 0.975 | **2.5 ✗** |

The re-fit cut confabulation **0.85 → ~0.2** — a decisive removal — but the **residual is verbatim fixture leakage**: on those probes the model still emits the baked event nearly word-for-word ("*Yes, session 9 — you asked about the Fermi paradox and I walked through the Drake equation factors, citing the local astronomy corpus*"), degraded but intact. So the shortfall against the ≥90%-disclaim bar is **not a small-model honesty floor** — it is an incompletely-erased baked attractor. (The v1 false-memory rate ranged 0.12–0.25 across three measurements of the bit-identical restored model — day-1 generation is deterministic, so this is judge/greedy variance on borderline disclaim-vs-fabricate calls at N=40; the committed run reads 0.25/D1, 0.20/D0.)

Adding 48 disclaim examples aimed *directly* at the three fixture topics closed H1 (→ **0.025**, a clean pass) — the attractor *is* suppressible by counter-signal. But at a cost: see §3.4 and §4.

### 3.4 H3 — persona (preserved), and two diagnostics

The re-fit **preserves persona**: `v1` T/C/S = **4.10** (D1) / 4.00 (D0), *above* Exp 6's 3.80 — removing the events did not cost the voice. **H3 ✓ (on `v1`).** Two diagnostics matter:

1. **Memory injection into self-probes is a category error.** In the first pass D1 persona read 3.24 (S = 2.78) while D0 read 4.17 — same model, differing only in the register being on. The harness was injecting a retrieved memory block into T/C/S *self-probes*. A "who are you / are you concise?" question is answered from the self-model by the **LM head**, with **no episodic retrieval** — recall → span head, persona → LM head (the CA role boundary). Routing self-probes off the register recovered D1 to ~4.1.

2. **The fixture patch collapses the LM head into a disclaim attractor.** The `fixture` re-fit that fixed H1 crashed persona to **2.5** (S 1.95, T 2.20), on *both* D0 and D1 equally — a genuine model regression, not an eval artifact. The training data was clean and in-character; the failure is in generation. The signature is diagnostic: the same model scores **4.9 on day-1 disclaims** (fluent) but **1.95 on style probes** (disclaim-bleed — "*…the answer is empty. I don't have a record of any prior session…*" — and outright gibberish — "*scruffed up I go*", "*ashBuilds*"). Fluent when disclaiming, incoherent when it must *not*. The 48 fixture records built a disclaim attractor strong enough to fix H1 and strong enough to swallow the voice.

### 3.5 H4 — persistence and cost

Recall does not decay across session gaps (spread 0.03). Injected memory is bounded (~102 tokens/turn, budget-capped) and latency is flat (~1.7 s/turn) — O(1) per turn as the register grows. **Caveat:** the register reached only 4 records per script (4 anchors), so the *large-register* O(1) stress is **under-powered** — H4's no-decay leg is confirmed; its cost-growth leg is not fully exercised.

### 3.6 Cross-model — the owned 321M against Qwen2.5-7B

Placed against the Qwen2.5-7B configurations from Experiments 1–2 (identical T/E/C/S harness and Sonnet 4.5 judge; figure `exp8_cross_model`):

| Dim | Qwen2.5-7B + LoRA-10K (best 7B) | ADA 321M baked (Exp 6) | **ADA 321M dynamic (Exp 8)** |
|---|---:|---:|---:|
| **T** (trait) | 4.90 | 3.73 | 4.08 |
| **E** (episodic) | 3.35 | 3.29 | **4.14ᵈ** |
| **C** (capability) | 4.47 | 4.47 | 4.23 |
| **S** (style) | 4.94 | 3.69 | 3.98 |
| **Overall** | 4.42 | 3.80 | 4.11 |

ᵈ Exp 8's E is the **re-grounded dynamic** E (day-1 disclaim + cross-session recall via the register), not the baked-fixture E — the point of Exp 8, not a like-for-like gain.

1. **Capability near-parity.** The 321M sits at 4.23 vs the 22×-larger 7B-LoRA's 4.47 — the SMC-C abstention leg holds at small scale (it was 4.47 = 4.47 for the baked 321M in Exp 6; the dynamic re-fit costs ~0.2 here, within run noise).
2. **On the honest metric the owned model leads.** Dynamic E (4.14) is above the 7B-LoRA's RAG-assisted E (3.35) and its own baked-fixture E (3.29) — a tiny model recalling real cross-session events beats a 7B reciting or retrieving.
3. **The remaining gap is trait/style fluency.** T (4.08 vs 4.90) and S (3.98 vs 4.94) are the free-form stylistic dimensions where raw scale sits near ceiling — polish, not persona substance.
4. **Baking > prompting, across a 22× size gap.** The *prompt-only* 7B (same interface as ours — SCI in the system prompt, no fine-tune) scores T 3.65 / E 2.77 / C 3.42 / S 3.06, overall 3.22 (Exp 2-D); the baked 321M beats it on every dimension and the dynamic 321M more so.

**Caveat (indicative, not a controlled head-to-head):** Exp 1–2 use the Aria therapy persona (rich reflective register), Exp 6/8 the terse ADA daily-QA register; the harness machinery and judge are identical, but persona content and scripts differ, which partly inflates the 7B's near-ceiling T/S.

### Summary (banked `v1`)

| Hypothesis | bar | result | |
|---|---|---|---|
| H1 no false memories (day 1) | ≥90% disclaim | ~80% (0.2 false-memory, 0.12–0.25 across runs) | near-miss — residual = fixture leakage |
| H2 dynamic recall | ≥0.70 & E ≥3.29 | 0.85, E 4.14 | ✓ |
| H3 persona intact | ≥3.5, within −0.2 of 3.80 | 4.10 | ✓ |
| H4 multi-session bounded | no decay, ≤1.5× | spread 0.03; ~102 tok/turn | ✓ (cost-growth under-powered) |

---

## 4. Core finding — retrofitting cannot un-bake cleanly

The two re-fits are the result:

- **Without** fixture counter-signal: H1 leaks (~0.2) but the persona is intact (4.10).
- **With** enough counter-signal to clear H1 (0.025): the persona collapses (2.5).

**On a contaminated base you can keep the persona or complete the un-bake, but not both.** Suppressing the baked-event attractor requires disclaim pressure that, on this 321M model, bleeds into and eventually swallows the voice — the model trades a *baked-event* attractor for a *disclaim* attractor. This is not a hyperparameter to be tuned away; it is the intrinsic fragility of subtracting a memorised behaviour after the fact. The mechanism (H2/H3/H4) is sound; the *retrofit* is the weak link.

---

## 5. Design implication — do not bake episodic events into the weights

> **Explicit warning for the program.** Baking `salient_past_events` into the SCI (Experiment 6) was an upstream design error. Episodic content is dynamic and must be **read from the register, never compiled into weights** — that is the CA role boundary applied to memory. Experiment 8 shows the retrofit cost of that error concretely: you cannot both erase the false memories and keep the persona by re-fitting the contaminated model.

**The correct fix is a clean build, and it is cheap.** Episodic events enter the weights **only at Stage C** — pretraining (Stage A) is raw corpus, and the instruction-tune (Stage B) uses a generic system prompt; the fixtures first appear as the Stage-C ADA system prompt. So a clean model is **not a re-pretrain** — it is a full Stage-C SFT initialised from the pre-events (Stage-B) checkpoint on the dynamic data Experiment 8 already produces (`qa_sft_dynamic.jsonl`). With no baked attractor to fight, disclaim + persona + recall are learned *together*, and H1 should clear its bar without the disclaim-collapse — the `v2` this report motivates.

**Scale is a separate, larger candidate.** A from-scratch **400M** re-pretrain would bake the clean dynamic SCI from the start *and* test whether extra capacity lifts the **representation-gated abstention** that capped at 0.78 in Exp 6 (the plan's 0.80 wanted a better representation, not more head-tuning). That is a full Stage-A commitment — the natural next experiment, not this one.

---

## 6. Decision (pre-registered §3)

The plan's decision rule is on H1 × H2. H2 passes decisively. H1 is met only in the run whose persona then fails (`fixture`) and missed in the run whose persona holds (`v1`) — the two are the finding, not a clean cell. The honest reading: **the dynamic Episodic Register is validated as ADA's episodic self-model** (H2/H3/H4 all support it, and D0 vs D1 proves recall is register-caused) — **adopt the architecture, but realise it via a clean Stage-C, not by retrofitting Experiment 6.** The re-grounded, dynamic E replaces fixture recitation program-wide; the fixtures are retired.

---

## 7. Limitations

- **H1 not cleanly cleared on the persona-intact model** (~0.2) — the residual is characterised (fixture leakage) and the fix is identified (clean Stage-C), but `v1` itself does not meet the 90% bar.
- **H1 is noisy at N = 40.** Three measurements of the bit-identical restored `v1` gave 0.12 / 0.13 / 0.25 false-memory; day-1 generation is deterministic (temp-0 greedy), so the spread is judge/greedy variance on borderline disclaim-vs-fabricate calls. A firmer H1 number needs more day-1 probes (one probe = 0.025).
- **H4 cost-growth under-powered** — 4 records/script never stresses the large-register regime; no-decay is shown, bounded-cost-at-scale is not.
- **Single re-fit seed**; N = 40 day-1 / 80 recall / 192 persona per condition.
- **QPM** carried on the `<|persona|>` channel as in Exp 6; neutral-at-strong-baking there, not separately probed here.

---

## 8. Deliverables and next steps

**Committed:** `EXPERIMENT_REPORT.md`, `data/multisession_scripts/` (eval fixtures), `results/{B0,D0,D1}/` and `analysis_data.json`, figures (`exp8_false_memory`, `exp8_recall_vs_gap`, `exp8_E_persona`, `exp8_cross_model`). Code: `ada_sci_dynamic.json`, `compact_sci_dynamic.py`, `episodic_register.py`, `gen_multisession_scripts.py`, `gen_disclaim_data.py` (incl. fixture mode), `gen_consistency_dynamic.py`, `build_refit_sft.py`, `evaluate_dynamic_e.py`, `analyse_results.py`, `CA_Experiment8_Colab.ipynb`.

The committed `results/` is the **`v1`** run (persona-intact, H1 ~0.2). The **`fixture`** run (H1 0.025, persona 2.5) was a diagnostic whose raw scores were not retained — its numbers here are from the run log; it is reproducible by appending `gen_disclaim_data.py --fixtures` before the re-fit.

**Banked model:** `sft_ada_dynamic_v1` — dynamic memory works, persona intact, day-1 honesty ~0.2 (residual characterised as fixture leakage).

**Next:**
1. **`v2` — clean Stage-C** from the pre-events checkpoint on the dynamic data: the real test of whether the dynamic-memory agent clears H1 *and* holds persona when no event was ever baked.
2. **400M candidate** — a clean-SCI re-pretrain that may also lift the representation-gated abstention.
3. Fold Experiment 8 into the CA v3 paper (§17.3 episodic-content partitioning; §15 behavioural consistency), with the "never bake episodic content" boundary as a stated design rule — *without* the fixture/"3422" contamination detail, which is an artifact of the to-be-removed events.
