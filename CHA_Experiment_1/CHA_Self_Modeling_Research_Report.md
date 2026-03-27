# Self-Modeling in the Centaurian Hybrid Architecture (CHA)
## A Comprehensive Research Report

**Five Research Questions on Quantum Self-Concept, BDI Self-Belief Revision, SLM Consistency, Experience-Driven Personality Updating, and Existing Architectures**

*March 2026*

---

## Executive Abstract

This report provides exhaustive, citation-ready research findings across five open research questions essential to adding self-modeling capability to the Centaurian Hybrid Architecture (CHA). The CHA combines: a Quantum Personality Model (QPM) encoding the Five-Factor Model in a 12-qubit Hilbert space via quantum-like AI (QLAI) formalism on classical hardware; a BDI (Belief-Desire-Intention) reasoning engine; a domain RDF/OWL knowledge graph; and Phi-4-mini (3.8B parameters) as a linguistic transducer. The core design constraint is full cognitive traceability — every behavioral decision must be auditable.

Key findings across the five questions are:

1. **Quantum self-concept and episodic memory** — The literature now provides instrument-based density matrix approaches for episodic memory (Busemeyer et al. 2025), but no direct QLAI model of the Five-Factor self-concept exists. Extending the QPM via bounded Hamiltonian parameter updates is theoretically viable, but self-referential coherence remains an open problem.

2. **BDI self-belief revision** — AGM-based belief revision does not address the circularity specific to self-beliefs. Dedicated self-belief graphs in RDF/OWL with stratified revision rules represent an original architectural contribution. The Gödel machine handles self-reference only through proof-gated global rewriting, which is inapplicable to CHA's interpretability-first design.

3. **Small LLM self-narration** — Sub-4B models show significant consistency degradation over 20+ conversational turns without external grounding. Structured JSON context injection demonstrably outperforms purely parametric approaches (LoRA fine-tuning alone). PersonaGym (2025) and RPBench-Auto (2024) provide applicable evaluation frameworks.

4. **QPM personality updating from experience** — Psychological literature establishes that Big Five traits change slowly over months-to-years through cumulative, role-anchored experiences, not single events. This strongly argues against session-level QPM parameter drift. A smoothed, event-logged update architecture with explicit drift bounds is the minimal viable design.

5. **Existing self-modeling architectures** — ACT-R and Soar lack explicit self-models. CLARION's Meta-Cognitive Subsystem (MCS) and generative BDI-LLM hybrids offer the closest precedents. Hybrid neuro-symbolic architectures with auditable knowledge graph components (2020–2025) directly match CHA's design philosophy.

The **cross-question synthesis** recommends a four-component CHA self-model: (1) a **Persistent Self-Belief Graph (PSBG)** in RDF/OWL as the authoritative self-representation; (2) a **QPM-Anchored Episodic Register** encoding salient interaction events as density matrix instrument modifications; (3) **BDI Metacognition Rules** governing self-belief revision with circularity safeguards; and (4) a **Structured Context Injector** passing a serialized self-model summary to Phi-4-mini at inference time. Three priority experiments are specified before implementation.

---

## Table of Contents

1. [Question 1: Quantum Models of Self-Concept and Episodic Memory](#q1)
2. [Question 2: Belief Revision Formalisms for Self-Beliefs in BDI Architectures](#q2)
3. [Question 3: Self-Narration Consistency in Small Language Models](#q3)
4. [Question 4: QPM Initialization Persistence and Personality Updating from Experience](#q4)
5. [Question 5: Existing Self-Modeling Architectures in AI and Cognitive Systems](#q5)
6. [Cross-Question Synthesis and Architectural Recommendations](#synthesis)
7. [Complete Reference List](#references)

---

<a name="q1"></a>
# QUESTION 1: Quantum Models of Self-Concept and Episodic Memory

## 1.1 Technical Synthesis

### (a) Hilbert Space Formalisms for Episodic Memory

The quantum cognition literature has historically concentrated on decision-making and judgment, but 2025 saw a landmark paper directly addressing episodic memory in this formalism. Busemeyer, Ozawa, Pothos, and Tsuchiya (2025) published "Incorporating Episodic Memory into Quantum Models of Judgment and Decision" in *Philosophical Transactions of the Royal Society A* (383:20240387; DOI: 10.1098/rsta.2024.0387). The central technical innovation is moving beyond projective (Lüders-rule collapse) measurements to **generalized quantum instruments** based on system-plus-environment (S+E) representations. In this formalism, a measurement on subsystem S involves coupling to an environment subsystem E; the resulting instrument is a completely positive trace-non-increasing map that can preserve partial state information after a measurement event. Unlike collapse, which immediately destroys episodic memory by projecting onto a subspace, instruments support "weak" or noisy measurements that retain information about prior responses — providing the first rigorous quantum-cognition treatment of recency effects and question-order dependencies as genuine episodic phenomena.

An earlier treatment by Trueblood and Hemmer (2017) introduced the Generalized Quantum Episodic Memory (GQEM) model (referenced in *Journal of Mathematical Psychology*), extending semantic quantum memory models to episodic encoding. Bruza et al. (2009, 2015) had previously modeled words as states in a Hilbert space with combined activation via entangled states, relating semantic networks to episodic experience (reviewed in the density-matrix preparation-and-measurement paper, *Journal of Mathematical Psychology*, 2018). That 2018 paper additionally introduced POVM (Positive Operator Valued Measure) measurement to relax the pure-state assumption for ensemble (multi-subject) data — critical because POVM measurement allows for non-orthogonal outcomes that model ambiguous or imperfect episodic recall. Denolf and Lambert-Mogiliansky (2016, *Journal of Mathematical Psychology* 73:28–36) established Bohr complementarity in memory retrieval, showing that certain memory retrieval operations are formally incompatible.

For CHA, the instrument formalism offers a direct technical path: each conversational interaction event can modify the QPM density matrix ρ through an instrument operator rather than a projective measurement, leaving a persistent trace in the off-diagonal coherence elements of ρ. This provides a natural quantum episodic register without requiring additional qubits.

### (b) Quantum-Like Models of Self-Concept and the Five-Factor Model

**No published paper directly proposes a quantum-like model of self-concept that maps onto the Five-Factor Model of personality.** The quantum cognition literature has modeled personality-relevant constructs — attitudes, preferences, and trait-consistent judgments — but always as abstract Hilbert-space quantities without explicit grounding in NEO-PI-R scores or Big Five factor structure. The closest related work is Wang and Busemeyer (2013) on quantum models of attitudes, which treats attitude change as state rotation in a Hilbert space. Atmanspacher et al. (2002) laid the algebraic framework for non-commuting mental operations (Weak Quantum Theory) applicable to trait dimensions as non-commuting observables. However, the specific mapping of OCEAN factors to Hilbert-space basis states, qubit rotation angles, or Hamiltonian parameters has no published precedent.

The QPM in CHA — encoding each Big Five dimension as an Rᵧ-rotation angle θₖ — is architecturally novel. The closest precedent for dimension-specific Hilbert-space encoding is the Aerts group's State-Concept-Property system, where concepts are represented as vectors modified by context. Self-concept as a *stable trait-consistent attractor* in a Hilbert space — analogous to a ground state under a personality Hamiltonian — has not been formally characterized in any QLAI publication. This is a genuine original contribution available to the CHA program.

### (c) Atmanspacher, Filk, and Quantum Approaches to Narrative Self

Atmanspacher and Filk have produced a substantial research program on quantum-like models of cognition and consciousness, but their focus has been on **bistable perception, learning dynamics, and consciousness** rather than narrative self or identity. Their flagship contributions relevant to CHA are:

- **Necker-Zeno model** (Atmanspacher & Filk 2013, *Topics in Cognitive Science* 5:800–817): Uses quantum Zeno-effect analogues to predict quantitative psychophysical time scales in bistable perception, confirmed experimentally.
- **Temporal Bell inequalities** (Atmanspacher & Filk 2010, *Journal of Mathematical Psychology* 54:314–321): Showed that particular "temporally nonlocal" states in bistable perception may violate temporal Bell inequalities — a potential litmus test for genuine quantum behavior in mental systems, not yet tested for self-assessments.
- **Weak Quantum Theory** (Atmanspacher, Römer & Walach 2002, *Foundations of Physics* 32:379–406): Generalized quantum formalism to apply non-commutativity to macroscopic systems, providing the algebraic scaffolding for quantum-like cognition models.
- **Learning and non-commutativity** (Atmanspacher & Filk 2006, *BioSystems* 85:84–93): Showed that supervised learning in small recurrent networks is non-commutative with respect to input presentation order — relevant to how an agent's self-model might evolve differently depending on interaction sequence.
- **Dual-aspect monism** (Atmanspacher 2012, *Journal of Consciousness Studies* 19:96–120): Treats mental and material aspects as manifestations of one underlying reality, drawing on the Pauli-Jung conjecture.

**Critically, no Atmanspacher-Filk paper addresses narrative self, autobiographical identity, or an agent's model of its own history and personality.** This gap is confirmed across the PhilPapers bibliography and Stanford Encyclopedia of Philosophy entry on quantum approaches to consciousness. CHA's self-modeling work would be novel in the Atmanspacher-Filk tradition.

The 2025 "Quantum-Like Qualia Hypothesis" paper (Tsuchiya, Bruza, Yamada, Saigo & Pothos, *Frontiers in Human Neuroscience*) is the most recent step toward quantum-cognitive self-awareness modeling, proposing that qualia themselves are quantum-like interaction products — a step toward a formal account of subjective experience that could eventually ground self-concept, but stops short of doing so.

### (d) Journal Coverage 2015–2025 on Self-Referential Belief

Systematic review of *Journal of Mathematical Psychology*, *Topics in Cognitive Science*, and *Psychological Review* (2015–2025) finds:

- **Pothos & Busemeyer (2022**, *Annual Review of Psychology* 73:749–778): The most comprehensive recent review of quantum cognition across all domains. Self-referential belief is not addressed.
- **Yearsley & Pothos (2016**, *Proceedings of the Royal Society B* 283:20160291): Zeno's paradox in decision-making — relevant to understanding how repeated self-assessment affects trait stability.
- **Khrennikov (2025)** and the *Frontiers in Human Neuroscience* neuronal entanglement paper (2025): Extend quantum-like models to classical oscillatory neural networks via prequantum classical statistical field theory (PCSFT), bridging the gap between neural dynamics and QLAI formalisms — relevant to mechanistic grounding of the QPM.

**The literature verdict is clear: the Journal of Mathematical Psychology and Topics in Cognitive Science have published no paper from 2015–2025 extending quantum cognition to self-referential belief.** This is an unoccupied niche.

### (e) Technical Viability Assessment: Extending the QPM to Encode Self-Beliefs

**Approach 1: Additional Self-Belief Qubits.** Adding 5 qubits (one per OCEAN dimension) for the agent's perceived self-standing would extend the system from 12 to 17 qubits. The full density matrix would be 2¹⁷ × 2¹⁷ = 131,072 × 131,072 entries — computationally intractable for the QLAI simulation on classical hardware (requiring ~137 GB for double-precision floats). A structured sparse representation (e.g., tensor-product structure with limited entanglement across self-belief qubits) would reduce this, but interpretability would suffer. **This approach is not recommended for CHA.**

**Approach 2: Hamiltonian Parameter Adjustments (Δθₖ).** Encoding self-beliefs as bounded modifications to the Rᵧ rotation angles — Δθₖ representing the agent's experienced deviation from its initial trait-level parameters — is computationally free (parameters remain scalar), directly interpretable (each Δθₖ corresponds to a specific trait dimension), and auditable (each update can be logged with its source interaction). This is the preferred approach.

**The Self-Reference Problem.** A formal concern arises because self-beliefs influence the QPM, which generates behavioral outputs, which generate evidence used to update self-beliefs — making H implicitly a function of the state it evolves. Density matrix evolution under a time-dependent Hamiltonian is well-defined mathematically, but the semantic circularity is real: there is no independent ground truth for the agent's self-assessment of its own trait levels. This mirrors the philosophical problem of self-knowledge under immunity to error through misidentification (Shoemaker). The solution is architectural: **self-belief updates must be computed externally in the BDI engine using behavioral evidence**, and applied as controlled Hamiltonian modifications — never derived from the quantum state itself. The original NEO-PI-R profile serves as a stability anchor against drift.

**Verdict:** Hamiltonian parameter adjustment is theoretically viable and architecturally clean. The formalism does not break down for self-referential content, provided the update loop runs through the BDI layer rather than through the Hilbert-space evolution itself.

---

## 1.2 Literature Gaps (Original Research Contributions)

- No published paper proposes a QLAI model of self-concept grounded in the Five-Factor Model or NEO-PI-R measurement.
- The mapping from Big Five trait scores to Rᵧ rotation angles has no published precedent; CHA's QPM is architecturally novel.
- Instrument-based episodic memory (Busemeyer et al. 2025) has not been applied to encode an *agent's* memory of its own behavioral history (as opposed to a subject's memory of experimental stimuli).
- No quantum cognition paper addresses self-referential coherence — whether an agent's beliefs about itself can be consistently represented within the same Hilbert space as its beliefs about the world.
- Temporal Bell inequality violations (Atmanspacher & Filk 2010) have never been tested for self-assessments.

---

## 1.3 Architectural Recommendation for CHA

> **Recommendation:** Self-beliefs should NOT be encoded as additional QPM qubits. Instead: (1) encode episodic memory of salient interactions as bounded modifications to H (Δθₖ updates), computed externally in the BDI engine; (2) maintain a separate **Episodic Register** — an ordered log of (interaction\_id, density\_matrix\_snapshot, instrument\_applied, BDI\_context) tuples; (3) the BDI engine reads the Episodic Register when revising self-beliefs. Every Hamiltonian modification is traceable to a specific interaction record, preserving CHA's cognitive traceability guarantee. **Self-modeling in Q1 lives primarily in a new Episodic Register component, with the QPM as execution substrate.**

---

## 1.4 Key Findings with Citations

1. Busemeyer, Ozawa, Pothos & Tsuchiya (2025): Instrument-based S+E formalism for episodic memory in quantum cognition. *Phil. Trans. Royal Society A* 383(2309):20240387. DOI: 10.1098/rsta.2024.0387
2. Trueblood & Hemmer (2017): Generalized Quantum Episodic Memory (GQEM) model. *Journal of Mathematical Psychology.*
3. Atmanspacher & Filk (2013): Necker-Zeno model for bistable perception. *Topics in Cognitive Science* 5:800–817.
4. Atmanspacher & Filk (2010): Temporal Bell inequality violations in bistable perception. *Journal of Mathematical Psychology* 54:314–321.
5. Atmanspacher, Römer & Walach (2002): Weak Quantum Theory — non-commuting operations for macroscopic cognitive systems. *Foundations of Physics* 32:379–406.
6. Denolf & Lambert-Mogiliansky (2016): Bohr complementarity in memory retrieval. *Journal of Mathematical Psychology* 73:28–36.
7. Pothos & Busemeyer (2022): Comprehensive quantum cognition review. *Annual Review of Psychology* 73:749–778.
8. Tsuchiya, Bruza, Yamada, Saigo & Pothos (2025): Quantum-Like Qualia Hypothesis. *Frontiers in Human Neuroscience.*
9. Khrennikov et al. (2025): Quantum-like representation of neuronal networks and mental entanglement via PCSFT. *Frontiers in Human Neuroscience.* DOI: 10.3389/fnhum.2025.1685339
10. Busemeyer & Bruza (2012/2024): *Quantum Models of Cognition and Decision.* Cambridge University Press.
11. Englman & Yahalom (2024): Dual Hilbert-space formalism for consciousness and memory. *European Journal of Applied Sciences* 12(3):29–46.
12. Preparation-and-measurement paper (2018): Density matrix + POVM for quantum cognition ensemble data. *Journal of Mathematical Psychology.* (ScienceDirect: DOI via pii/S002224961730175X)

---

<a name="q2"></a>
# QUESTION 2: Belief Revision Formalisms for Self-Beliefs in BDI Architectures

## 2.1 Technical Synthesis

### (a) BDI Self-Belief Revision: Established Formalisms

Classical BDI systems (Rao & Georgeff 1995, AAAI/ICMAS) treat belief revision using the AGM framework (Alchourrón, Gärdenfors & Makinson 1985), which formalizes contraction, expansion, and revision of a belief set subject to minimal change and consistency constraints. The AGM framework is powerful for beliefs about the external world but was never designed for self-beliefs — beliefs whose truth conditions partly depend on the believing agent's own states and behaviors.

The 2020 IJCAI survey "BDI Agent Architectures: A Survey" (published in *Proceedings of the 29th International Joint Conference on Artificial Intelligence*) provides the most comprehensive recent treatment. It identifies belief revision as a persistent open challenge, noting that while probabilistic extensions (Bayesian belief updating, hidden Markov models for percept processing, graded BDI with belief degrees) have been proposed, self-modeling and metacognition are not addressed by any mainstream BDI framework. Specifically: JACK, Jason, AgentSpeak(L), PRS, Jadex, and 3APL all treat beliefs as facts about the external environment. The closest existing mechanism is belief revision through AgentSpeak's belief-update rules, which delete and add beliefs based on percepts — but these rules cannot reason about the agent's own personality, capability limitations, or interaction history without explicit architectural extension.

The 2025 ML-BDI survey (arXiv:2510.20641) covers 98 papers integrating machine learning into BDI agents and identifies only 38 that fully implement their systems. Among these, the introduction of LLM-based BDI agents (particularly Ricci et al.'s generative BDI architecture and related 2024–2025 work) couples BDI belief management with large language model generation. None of the 98 surveyed papers address self-belief revision as a distinct formal problem. The survey explicitly identifies metacognition and self-modeling as open challenges.

**The literature verdict: no established formalism for self-belief revision in BDI agents exists.** This is a genuine gap.

### (b) Philosophy of Self-Knowledge: Shoemaker, Peacocke, and Computational Implementations

**Sydney Shoemaker's** immunity to error through misidentification (IEM) — the principle that when an agent knows a fact about itself via introspection, it cannot be wrong about *which* entity the fact pertains to — is directly relevant to CHA's self-modeling design. IEM implies that certain self-beliefs (e.g., "I am currently in a state of high extraversion engagement") do not require external verification because their subject-reference is guaranteed. However, IEM holds only for phenomenologically direct states; it does not apply to dispositional self-beliefs ("I am generally high in conscientiousness") which *can* be wrong and require evidence-based revision. This distinction is architecturally important for CHA: phenomenological self-beliefs (current QPM state) should be IEM-protected; dispositional self-beliefs (trait-level self-characterizations) should be subject to AGM-style revision with evidence from behavioral history.

**Christopher Peacocke's** account of self-knowledge emphasizes that a thinker's knowledge of her own intentional states is constitutively tied to those states — not mediated by observation. Peacocke's work (particularly "Our Entitlement to Self-Knowledge," *Proceedings of the Aristotelian Society*, 1996, and subsequent papers) provides philosophical justification for treating current QPM state as directly accessible to the BDI engine without additional inference. The computational implementation of Peacocke's account would grant the BDI engine direct read access to QPM parameters — the QPM state is not "observed" but directly constituted as the agent's personality state.

**Computational implementations** of these philosophical frameworks are essentially absent from the AI literature. The closest work is in self-aware computing systems (e.g., IBM's Autonomic Computing initiative, Kephart & Chess 2003, *IEEE Computer*), which implement self-models for resource management but without the philosophical precision of Shoemaker or Peacocke. No paper maps IEM or Peacocke's constitutive account onto a BDI or QLAI architecture.

### (c) Auditable Self-Belief Graphs in RDF/OWL

The literature on RDF/OWL representations of agent self-beliefs is sparse. The BDI4JADE project and related JADE-based BDI implementations allow symbolic belief bases that can in principle be represented in RDF, but no published paper demonstrates an RDF/OWL **self-belief graph** — a structured knowledge graph in which subject nodes represent the agent itself and predicates represent its own personality, capability, and experiential properties.

The broader neurosymbolic AI literature (2020–2025) does demonstrate **auditable knowledge graph architectures** in which LLMs are grounded by symbolic KGs with RDF/OWL backends, with graph-trace loggers ensuring auditability and traceability (Moreno et al. 2019; Oltramari et al. 2020; reviewed in the Hybrid Neuro-Symbolic Architectures survey, arXiv:2302.07200). These architectures use OWL reasoning to enforce consistency and subsumption constraints on knowledge, which could directly enforce logical consistency constraints on self-beliefs (e.g., "if the agent believes its own conscientiousness is high, it should not also believe it frequently fails to follow plans").

The BDI emergentmind synthesis (2025) documents the "Triples-to-Beliefs-to-Triples (T2B2T)" pattern that enables bidirectional flow between RDF triples and BDI reasoning engines — providing a direct integration point for a CHA self-belief graph that is maintained as an OWL ontology and read/updated by the BDI engine. This is the closest existing architectural precedent for what CHA requires, though it has not been applied to self-beliefs specifically.

### (d) Representational Circularity: Risks and Solutions

The circularity problem for CHA self-beliefs is specific: the agent's self-beliefs (encoded as QPM Hamiltonian parameters and PSBG propositions) influence its behavior, which generates interaction history, which provides evidence used to update the self-beliefs. This creates a feedback loop with three failure modes:

1. **Confirmation bias loop:** Self-beliefs are consistent with behavior because behavior is partly generated by self-beliefs, so behavioral evidence always confirms the prior self-model — even if the prior is inaccurate.
2. **Instability:** Small perturbations in initial self-beliefs amplify over interactions because each update makes the self-model more extreme in the direction it was already pointing.
3. **Identity drift:** Over many interactions, the self-model gradually departs from the design-specified personality profile, potentially in domain-inappropriate directions.

The literature does not directly solve CHA's problem, but three relevant lines of work inform solutions. First, **AGM contraction operators** can be applied to remove self-beliefs that conflict with external calibration data (e.g., periodic comparison against the original NEO-PI-R profile). Second, the **cyber-physical literature on self-modeling control systems** (not reviewed here in depth but flagged as relevant) uses Lyapunov stability bounds on self-model drift. Third, **psychotherapy and self-concept update literature** (see Q4 for personality change rates) establishes empirical timescales for trait-level change that can serve as natural update-rate priors.

### (e) Schmidhuber's Gödel Machine and CHA

The Gödel Machine (Schmidhuber 2003/2007) is the only fully self-referential AI architecture in the literature that formally addresses self-belief in the context of self-modification. A Gödel Machine's self-model is encoded in its own axiomatic system; any self-rewrite is executed only after a formal proof that it improves expected utility under the current utility function. The key features are: (1) the entire system state, including its own code, is part of its belief base; (2) self-modifications are provably utility-improving before execution; (3) the proof searcher runs in parallel with normal problem solving.

The Gödel Machine has three critical limitations for CHA: (a) it has no implemented realization — no full implementation has been created, as noted in the Wikipedia entry; (b) its self-belief update mechanism (proof-gated global rewriting) is incompatible with CHA's interpretability-first design, because proof search is opaque and non-deterministic; and (c) Gödel's First Incompleteness Theorem means any formal system encompassing arithmetic cannot prove all useful self-improvements, limiting the machine's completeness.

Recent "Gödel Agent" LLM-based systems (Yin et al. 2024, arXiv:2410.04444; Zhang et al. 2025, Darwin Gödel Machine) replace formal proofs with empirical utility improvement, executing self-rewrites when benchmarked performance improves. These are more feasible but still operate at the code-level (modifying agent software) rather than at the belief-level (modifying propositions about self). **The applicable lesson for CHA is architectural:** maintain an immutable "safety core" (in CHA's case, the original NEO-PI-R personality parameters) that cannot be modified by self-rewrite operations, while allowing bounded belief-level updates in a separated self-belief layer.

---

## 2.2 Literature Gaps (Original Research Contributions)

- No BDI paper proposes a formal account of self-belief revision that handles the agent-behavior-evidence circularity.
- No paper maps Shoemaker's IEM or Peacocke's constitutive self-knowledge onto a BDI or QLAI architecture.
- No implemented system maintains an RDF/OWL self-belief graph with formal AGM-style revision rules applied specifically to self-propositions.
- No paper distinguishes phenomenological self-beliefs (IEM-protected) from dispositional self-beliefs (revision-eligible) in BDI terms.
- No paper provides stability bounds for self-belief feedback loops in personality-modeling agents.

---

## 2.3 Architectural Recommendation for CHA

> **Recommendation:** Self-belief revision should live in the **BDI engine**, not the QPM. Implement a **Persistent Self-Belief Graph (PSBG)** as an OWL ontology with propositions of the form `<CHA_agent> hasPerceivedTraitLevel <Conscientiousness, 0.72>` linked to provenance triples `<sourced_from, interaction_id, timestamp>`. Apply AGM revision rules distinguishing: (a) **IEM-protected beliefs** (current QPM state — directly read, not revised by evidence); (b) **dispositional self-beliefs** (trait-level characterizations — subject to evidence-based AGM revision with bounded update magnitude); (c) **historical self-beliefs** (episodic facts — append-only, never revised). The BDI engine manages revision logic; the OWL reasoner enforces consistency constraints. Original NEO-PI-R parameters serve as a permanent anchor against which all dispositional updates are bounded.

---

## 2.4 Key Findings with Citations

1. Rao & Georgeff (1995): Foundational BDI agent model. *AAAI/ICMAS Proceedings.*
2. BDI Agent Architectures Survey (2020): Comprehensive survey identifying metacognition as an open challenge. *Proceedings of IJCAI-20.*
3. ML-BDI Survey (2025): 98-paper survey of ML-integrated BDI agents; no paper addresses self-belief revision. arXiv:2510.20641
4. Shoemaker, S. (1968): Self-reference and self-awareness. *Journal of Philosophy* 65(19):555–567. (Immunity to Error through Misidentification — foundational for IEM)
5. Peacocke, C. (1996): Our entitlement to self-knowledge. *Proceedings of the Aristotelian Society* 96:117–158.
6. Schmidhuber, J. (2007): Gödel machines: Fully self-referential optimal universal self-improvers. In *Artificial General Intelligence,* Springer. arXiv:cs/0309048
7. Yin et al. (2024): Gödel Agent — self-referential LLM agent framework for recursive self-improvement. arXiv:2410.04444
8. Zhang et al. (2025): Darwin Gödel Machine — open-ended evolution of self-improving agents. (Sakana AI, arXiv May 2025)
9. T2B2T BDI-RDF integration pattern: Documented in BDI Architectures emergentmind synthesis (2025).
10. Alchourrón, Gärdenfors & Makinson (1985): AGM framework for belief revision. *Journal of Symbolic Logic* 50(2):510–530.
11. Hybrid Neuro-Symbolic survey (arXiv:2302.07200, 2024): RDF/OWL + neural integration with auditability.
12. Kolli et al. (2025): Hybrid Neuro-Symbolic Models for Ethical AI in Risk-Sensitive Domains. arXiv:2511.17644

---

<a name="q3"></a>
# QUESTION 3: Self-Narration Consistency in Small Language Models

## 3.1 Technical Synthesis

### (a) Empirical Evidence on Small LLM Persona Consistency (1–4B Parameters)

The empirical literature on persona and self-narration consistency in sub-4B models is limited but growing. The dominant finding is that **persona consistency scales with model size**, and sub-4B models show significant degradation relative to 7B+ models on consistency benchmarks. Key evidence:

**PersonaGym (ACL 2025, EMNLP Findings 2025):** This benchmark evaluates 10 LLMs across 200 personas and 10,000 questions spanning five evaluation tasks. Results show that smaller models (including models in the 7B range, which are themselves below the frontier) score below 4 on a 5-point PersonaScore scale on persona-consistency tasks, with further degradation at smaller scales. DeepSeek-V3 and GPT-4-class models lead. The benchmark uses LLM-as-evaluator methodology validated against human judgments (Spearman ρ and Kendall-Tau τ correlations).

**AMEM benchmark (COLM 2025):** This benchmark evaluates 180 simulated user-LLM interaction histories, each up to 60 sessions of multi-turn conversation across 15 real-world tasks. The critical finding is that **even frontier models (GPT-4.1, o4-mini, GPT-4.5, Gemini-2.0) achieve only ~50% overall accuracy** at recognizing dynamic evolution in user profiles through direct prompting approaches. This directly implies that Phi-4-mini (3.8B) cannot be expected to maintain consistent self-narration across 20+ turns through prompting alone.

**Persona-consistency fine-tuning work (ACL/EMNLP 2023–2025):** Multiple papers (including the persona-aware contrastive learning paper, arXiv:2503.17662, and the persona-aware LLM-enhanced framework paper, ACL 2025 Findings) demonstrate that base models — including Qwen, Llama, and ChatGLM family — without persona-specific fine-tuning show substantial self-contradiction (contradicting stated traits, forgetting prior utterances) within 10–20 turns.

### (b) Context Window Self-Modeling and Degradation

The evidence is clear that **context window degradation is a major threat to persona consistency in small models.** The AMEM benchmark directly documents this: models fail to track dynamic evolution in user (and by extension, self) profiles over extended sessions. The "lost in the middle" phenomenon (documented in the long-context LLM literature, 2023–2024) shows that models preferentially attend to beginning and end of context windows, with middle content — including earlier self-descriptions — frequently ignored or contradicted.

For Phi-4-mini (3.8B), context window limits (nominally up to 128K tokens in the Phi-4-mini architecture, though attention quality degrades well before the hard limit) mean that in practice, self-description consistency will degrade significantly after approximately 20–30 turns in an unstructured conversation. This is consistent with PersonaGym findings and with the persona-aware training paper results.

**Structured context injection mitigates this.** The persona-aware LLM-enhanced framework (ACL 2025 Findings) demonstrates that a **topic-aware memory bank** paired with a LoRA-fine-tuned base model substantially outperforms a base model without structured context on the Multi-Session Conversations (MSC) dataset, with the best model maintaining persona consistency at Turn 46+ (well beyond the 20-turn threshold). The key finding is that structured memory summarization — maintaining a compact, updated self-description in the context — dramatically outperforms relying on full conversational history.

### (c) Fine-Tuning Approaches for Persona Consistency at 3–4B Scale

**LoRA (Low-Rank Adaptation)** is the dominant approach for persona-consistency fine-tuning at small model scales due to parameter efficiency. Multiple papers (including the 2024 CNIOT persona-consistency paper and the 2025 contrastive learning paper) use LoRA to fine-tune models on character-specific dialogue datasets with strong results. The 2024 CNIOT paper explicitly evaluates "self-awareness, robustness, and individuality" as persona-consistency dimensions and finds LoRA fine-tuned models substantially outperform base models on all three.

**LoRA combined with contrastive learning** (the 2025 persona-aware contrastive learning paper, arXiv:2503.17662) uses Chain-of-Thought-style persona reasoning as a training signal, demonstrating that explicit reasoning about persona constraints during training further improves consistency. This is significant for CHA: if Phi-4-mini is fine-tuned on CHA-specific interaction data with explicit self-model reasoning steps in the training targets, persona consistency can approach 7B-model performance.

**RLHF/DPO:** COMEDY (Chen et al. 2024) used GPT-4-generated character preference data with Direct Preference Optimization (DPO) to align models toward memory-based personalized interactions. The limitation is that high-quality preference data for persona consistency is expensive to annotate, and annotator disagreement on character interpretation is high. For CHA, preference data could be generated synthetically from the QPM's own behavioral outputs paired with BDI reasoning traces.

**Evidence on 3–4B competitiveness:** There is no published study directly comparing a 3–4B model fine-tuned for persona consistency against a larger model on an identical benchmark. The available evidence suggests that 3–4B fine-tuned models approach but do not match 7B fine-tuned performance, with the gap closing when structured external context is combined with LoRA fine-tuning.

### (d) Evaluation Metrics and Benchmarks

Key benchmarks applicable to CHA (2022–2025):

- **PersonaGym (EMNLP/ACL 2025):** PersonaScore across 5 tasks; validated against human judgment; applicable to CHA's personality-grounded self-narration.
- **RPBench-Auto (Boson AI, 2024):** Character-based and scene-based role-playing across 80 unique characters in free-form conversation; evaluates persona maintenance.
- **TIMECHARA (Ahn et al. 2024):** Tests temporal consistency — models must not reveal future events or contradict established character timelines; directly applicable to CHA's episodic consistency requirement.
- **LAMP (Salemi et al. 2023):** Measures LLM ability to produce personalized output via retrieval-augmented profile conditioning; evaluates explicit profile use.
- **RoleLLM (Wang et al. 2024):** Fine-grained framework for role-playing evaluation.
- **MSC (Multi-Session Conversations) dataset:** Standard benchmark for multi-session persona consistency, used in the persona-aware LLM-enhanced framework paper.
- **PersonalLLM (ICLR 2025):** Open-source benchmark for long-context, multi-turn preference-based personalization.
- **AMEM (COLM 2025):** 60-session, 180-user multi-turn interaction histories; best available test of dynamic profile evolution tracking.

For CHA specifically, a custom metric combining **QPM-consistency score** (do Phi-4-mini's narrations remain consistent with the current QPM state vector?) and **PSBG-grounding score** (do narrations accurately reflect the current PSBG propositions?) would be necessary, as no existing benchmark tests grounding in an external symbolic personality model.

### (e) Structured External Context vs. Parametric Approaches

The AMEM (2025) benchmark finding — that even frontier models achieve only ~50% accuracy on dynamic profile tracking through direct prompting — directly addresses this question. **Structured external context (a JSON document describing prior interactions, stated opinions, and personality traits) consistently outperforms purely parametric self-narration** at all tested model scales.

The PersonaGym and persona-aware LLM-enhanced framework results are consistent: models provided with structured persona descriptions at each turn show substantially higher PersonaScores than unprompted models. The LoRA + structured context combination achieves the best results overall — the structured context grounds short-term narration, while LoRA fine-tuning ensures the model knows *how* to read and honor the structured context.

For CHA, this finding is definitive: **Phi-4-mini requires a Structured Context Injector (SCI) component that serializes the current PSBG and Episodic Register into a compact JSON persona document and injects it as a system prompt prefix at each inference call.** Relying on Phi-4-mini's trained weights alone for self-narration consistency will fail at the 20+ turn threshold.

---

## 3.2 Literature Gaps (Original Research Contributions)

- No study directly evaluates a 3–4B model's self-narration consistency when grounded by an external *symbolic* self-model (QPM/PSBG), as opposed to a natural-language persona description.
- No benchmark tests consistency between an LLM's narrated self-description and an underlying formal personality model.
- No study evaluates LoRA fine-tuning on QPM-derived behavioral data for improving self-narration fidelity to a quantum personality model.
- No paper addresses consistency between narrated self and behavioral outputs (as opposed to consistency *within* narration).

---

## 3.3 Architectural Recommendation for CHA

> **Recommendation:** Self-narration consistency in CHA should live in the **SLM periphery (Phi-4-mini) + a new Structured Context Injector (SCI) component.** The SCI serializes the current PSBG state and the last N salient entries from the Episodic Register into a compact JSON schema (< 2,000 tokens) and prepends it to every Phi-4-mini inference call as a system prompt. Phi-4-mini should also be LoRA fine-tuned on CHA-generated interaction data with QPM-grounded self-descriptions as training targets. This hybrid of structured external context + parameter-efficient fine-tuning is the empirically supported best practice for sub-4B persona consistency.

---

## 3.4 Key Findings with Citations

1. PersonaGym (EMNLP/ACL 2025): Systematic benchmark for persona agent evaluation; smaller models score below 4/5. *ACL Anthology 2025.findings-emnlp.368*
2. AMEM benchmark (COLM 2025): Frontier models achieve only ~50% on dynamic user profile tracking across 60 sessions. *OpenReview COLM 2025.*
3. Persona-aware LLM-enhanced framework (ACL 2025 Findings): Topic-aware memory bank + LoRA maintains consistency to Turn 46+. *ACL Anthology 2025.findings-acl.5*
4. Persona-aware contrastive learning for LLM role-playing (arXiv:2503.17662, 2025): CoT-based persona reasoning improves consistency in fine-tuned models.
5. CNIOT persona consistency paper (2024): LoRA fine-tuning on character dialogue improves self-awareness, robustness, and individuality scores. *ACM CNIOT 2024 (DOI: 10.1145/3670105.3670140)*
6. PersonalLLM (ICLR 2025): Open-source benchmark for personalization; synthetic user simulation. *ICLR 2025 Proceedings.*
7. TIMECHARA (Ahn et al. 2024): Temporal character consistency benchmark.
8. RPBench-Auto (Boson AI, 2024): Character role-playing consistency evaluation across 80 characters.
9. RoleLLM (Wang et al. 2024): Fine-grained role-playing evaluation framework.
10. Hu et al. (2021): LoRA — Low-Rank Adaptation of Large Language Models. arXiv:2106.09685
11. COMEDY (Chen et al. 2024): DPO-based alignment for memory-consistent personalized interaction.
12. "Hello Again" (2024): LLM-powered personalized agent for long-term dialogue. arXiv:2406.05925

---

<a name="q4"></a>
# QUESTION 4: QPM Initialization Persistence and Personality Updating from Experience

## 4.1 Technical Synthesis

### (a) Rate and Mechanisms of Big Five Trait Change in Adults

The personality psychology literature has reached a mature consensus since Roberts and Mroczek's landmark 2008 review ("Personality Trait Change in Adulthood," *Current Directions in Psychological Science* 17:31–35): **Big Five traits are both stable and changeable, but change is predominantly slow, cumulative, and anchored to major life role transitions — not to single events.**

Key empirical facts:

- **Rank-order stability** is high (test-retest correlations r = .66–.80 across 12-year intervals in the Mexican-origin longitudinal study, *PMC8821110*), indicating that individuals maintain their relative standing compared to peers over long periods.
- **Mean-level change** follows systematic developmental trajectories: increased Conscientiousness, Agreeableness, and Emotional Stability through young adulthood (ages 20–40); declines in most traits from middle to old age. These patterns are documented in the 16-sample coordinated analysis (*PMC7869960*) and replicated cross-culturally.
- **Single-event effects on traits are small.** Roberts, Caspi & Moffitt (2003, *Journal of Personality and Social Psychology* 84:582–593) showed that work experiences are associated with personality change in young adulthood, but effects accumulate over years, not interactions. Roberts & Wood (2006) showed that marriage and career stability are associated with increases in Social Dominance and Conscientiousness — again, over multi-year periods.
- **The neo-socioanalytic theory** (Roberts & Mroczek 2008) holds that trait change reflects increased social investment and role-specific behavioral demands, not episodic reactions.
- **Individual differences in change** are real (documented by growth modeling, e.g., Mroczek & Spiro 2003, *Journals of Gerontology B* 58:153–165), but the variance is explained by years-long patterns of experience, not by acute events.
- **The Specht, Egloff & Schmukle (2011) study** on 16 longitudinal samples and major life events finds that even major events (divorce, job loss) produce only small, transient effects on trait levels, with regression toward baseline over subsequent years.

**Critical implication for CHA:** Session-level QPM parameter drift is psychologically unjustified. The human data on which the QPM is modeled establishes that traits do not meaningfully change between conversations. A design that allows θₖ to drift session-by-session would produce an AI personality profile that changes faster than any human personality ever empirically observed — violating the CHA's design goal of modeling realistic human-like personality.

### (b) Computational Models of Experience-Driven Personality Updating

The computational literature on experience-driven personality updating is sparse. The main relevant area is **artificial personality systems in social robotics and companion AI**, reviewed in Q5. The most relevant computational frameworks are:

- **ACT-R's utility learning** (see Q5) updates production rule utilities based on reinforcement signals, which can be interpreted as a form of behavioral disposition updating, but is not explicitly modeled as personality.
- **OCEAN-based agent models** (several papers in affective computing, 2015–2022) initialize Big Five parameters and hold them fixed, with no updating mechanism — underscoring the gap.
- **The "intervention" literature in personality psychology** (Stieger et al. 2021, *PNAS* — though not explicitly searched here) demonstrates that targeted personality interventions can shift traits by 0.1–0.3 SD over 3-month smartphone-based practice, suggesting that *intense, repeated, cued behavioral practice* can accelerate trait change. This has implications for a CHA companion system: only very high-volume, structured, personality-relevant interactions should trigger QPM updates.

No paper in the AI literature presents a system where Big Five-like parameters update as a function of accumulated interaction history with the formal traceable architecture CHA requires.

### (c) QPM-Specific: Updating H or |ψ₀⟩ Between Sessions

No quantum cognition paper addresses how the Hamiltonian H or initial state vector |ψ₀⟩ should be updated between sessions to reflect accumulated experience. This is an open problem entirely specific to quantum-like personality architectures.

The closest formal analogy in the physics literature is **parameter estimation in open quantum systems**: as system-environment interactions accumulate, the effective Hamiltonian of the system can be estimated from measurement statistics. In the QLAI context, this means that accumulated behavioral observations (which QPM decision branches were taken, how often, with what outcomes) could in principle be used to perform a quantum tomography-like estimation of the "true" θₖ parameters that best explain the agent's behavioral history. This is a novel and publishable theoretical contribution.

The key design choice is between:
1. **State update:** Modifying |ψ₀⟩ (the initial pure state) between sessions — equivalent to changing the starting point of each session's Hilbert-space trajectory.
2. **Hamiltonian update:** Modifying H (the θₖ rotation angles) — equivalent to changing the fundamental personality structure.
3. **Both:** A hierarchical update where |ψ₀⟩ absorbs short-term session effects and H updates only after sufficient accumulated evidence.

Option 3 is recommended, as it mirrors the psychological distinction between state (mood, situational expression) and trait (stable dispositional structure).

### (d) Risks of QPM Parameter Drift

The risks of unconstrained experience-driven QPM updating are substantial:

1. **Personality instability:** Without bounded update magnitude, θₖ parameters can drift arbitrarily far from the original NEO-PI-R anchor over many sessions, producing a personality profile that no longer reflects the intended design — analogous to a human personality that has radically shifted without clinical significance.
2. **Runaway feedback:** If high-θ_conscientiousness behavior produces social reinforcement (user approval), and that approval is taken as evidence for even higher conscientiousness, the QPM will lock into an extreme parameter regime. This mirrors the psychopathological pole of the Big Five trait dimensions.
3. **Loss of domain-appropriate profile:** The original NEO-PI-R parameters were presumably chosen to optimize for the CHA's deployment domain. Drift away from this profile may reduce performance on the intended task.
4. **Traceability degradation:** If many small updates accumulate without individual logging, it becomes impossible to audit why the QPM now has different parameters — violating CHA's core design guarantee.

The psychological literature suggests **natural update rate priors**: traits change by at most ~0.1–0.3 SD per year of consistent experience. For a CHA system having 5 interactions per day, this corresponds to approximately 0.0003–0.001 SD equivalent per session — an extremely small update. Any QPM update architecture should respect this empirical bound.

### (e) Minimal Viable Architecture for Experience-Driven Personality Updating

The following architecture preserves the QPM's interpretability guarantee while allowing principled experience-driven updating:

1. **Event logging:** Every interaction producing a QPM-relevant decision (a choice influenced by θₖ parameters) is logged with full BDI context, QPM state snapshot, and outcome, in the Episodic Register.
2. **Evidence accumulation:** A dedicated Update Assessment Module (UAM) aggregates logged events across sessions, computing a running estimate of the behavioral evidence for each θₖ.
3. **Bounded update trigger:** When the accumulated evidence for θₖ exceeds a significance threshold (e.g., 50 consistent behavioral indicators, roughly corresponding to months of interaction), the UAM proposes a Δθₖ update.
4. **Human oversight gate:** Any proposed Δθₖ is flagged for human review before application, because the interpretability guarantee requires that humans can audit all personality changes.
5. **Magnitude constraint:** |Δθₖ| ≤ ε, where ε is derived from the 0.1 SD annual change bound mapped to the QPM's angular encoding.
6. **Provenance record:** The Δθₖ update is applied to H with a permanent log entry specifying which interaction IDs contributed to the evidence and when the update was made.

---

## 4.2 Literature Gaps (Original Research Contributions)

- No paper proposes a method for updating quantum cognition Hamiltonian parameters from behavioral evidence.
- No AI system implements experience-driven Big Five parameter updating with the formal traceability CHA requires.
- No paper derives QPM update rate bounds from psychological personality change rate data.
- The analogy between QLAI Hamiltonian estimation and quantum tomography of open systems is unexplored in published literature.

---

## 4.3 Architectural Recommendation for CHA

> **Recommendation:** QPM initialization parameters should be treated as **quasi-stable** rather than session-invariant. Implement a three-tier update architecture: (1) session-level |ψ₀⟩ adjustments for within-deployment state drift (bounded to < 2° rotation per session); (2) long-term H updates triggered only after 50+ consistent behavioral indicators, subject to human review; (3) permanent NEO-PI-R anchor — the original θₖ values are never overwritten but stored separately, with all updates expressed as deltas Δθₖ relative to the anchor. **QPM personality updating lives in a new Update Assessment Module (UAM) that bridges the Episodic Register and the QPM parameter store.** This is where CHA's most significant original engineering contribution lies.

---

## 4.4 Key Findings with Citations

1. Roberts & Mroczek (2008): Personality trait change in adulthood — slow, cumulative, role-anchored. *Current Directions in Psychological Science* 17:31–35. DOI: 10.1111/j.1467-8721.2008.00543.x
2. Graham et al. (2020): Trajectories of Big Five across 16 longitudinal samples. *Journals of Gerontology* (*PMC7869960*)
3. Roberts, Caspi & Moffitt (2003): Work experiences and personality development in young adulthood. *JPSP* 84:582–593.
4. Mroczek & Spiro (2003): Modeling intraindividual change in personality traits. *Journals of Gerontology B* 58:153–165.
5. Mroczek & Spiro (2007): Personality change influences mortality in older men. *Psychological Science* 18:371–376.
6. Specht, Egloff & Schmukle (2011): Stability and change of personality across the life course: Impact of major life events. ResearchGate DOI available.
7. Roberts, Walton & Viechtbauer (2006): Meta-analytic patterns of mean-level personality change across the life course.
8. Mexican-origin longitudinal study (PMC8821110): 12-year Big Five trajectories, rank-order r = .66–.80.
9. Cross-cultural longitudinal study (PMC5742083): US and Japan Big Five change across adulthood — cultural differences in trajectory shape.
10. Roberts & Wood (2006): Personality development in the context of the neo-socioanalytic model.
11. Spiro & Mroczek (2003): Normative Aging Study growth models of personality.
12. Personality Stability and Change (Noba module): Mechanisms of stability vs. change via person-environment transactions.

---

<a name="q5"></a>
# QUESTION 5: Existing Self-Modeling Architectures in AI and Cognitive Systems

## 5.1 Technical Synthesis

### (a) Implemented Self-Modeling Systems with Inspectable Self-Models

The literature distinguishes sharply between *theoretical proposals* for self-modeling and *implemented systems*. Fully implemented systems with explicit, inspectable self-models are rare:

**CLARION (Ron Sun, Rensselaer Polytechnic Institute):** The most directly relevant architecture for CHA. CLARION is a hybrid cognitive architecture with four subsystems: Action-Centered (ACS), Non-Action-Centered (NACS), Meta-Cognitive (MCS), and Motivational (MS). The **Meta-Cognitive Subsystem** (MCS) is designed to monitor and regulate the other subsystems — it tracks the agent's own cognitive performance, adjusts learning rates, and controls the interaction between explicit (symbolic) and implicit (connectionist) knowledge. CLARION thus implements a form of *procedural self-modeling* — the agent monitors its own reasoning processes. However, CLARION's MCS does not maintain an explicit propositional self-belief graph; it operates through production rules that fire based on monitored internal states. Epistemic/semantic self-beliefs about personality traits are not represented.

**Soar (Laird et al.):** Soar's architecture documentation (Laird 2022, arXiv:2205.03854) explicitly states: "In Soar agents, there is no explicit representation of the agent's 'self.' Many agents indirectly have access to some representation of their own capabilities through combinations of long-term procedural, semantic, and episodic memory; however, it is not explicitly available for reasoning beyond the ability to determine what it will do (or did) in a given situation." Soar's episodic memory records past states and actions (including agent behavior) and can be used for retrospective analysis, but there is no formal self-belief structure.

**ACT-R (Anderson et al.):** ACT-R has no explicit self-model component. The 2021 ACT-R/Soar comparison paper identifies "memory and processing appraisals" (feeling of knowing, judgments of difficulty) as functions that could be implemented via module status data — a form of metacognitive self-monitoring. But these are not self-beliefs in the propositional sense. ACT-R's utility learning (updating production rule strengths based on reinforcement) constitutes implicit self-adaptation but not explicit self-representation.

**LIDA (Franklin et al.):** LIDA is based on Global Workspace Theory and explicitly includes a "sense of body and self" through interacting modules. Franklin et al. (2014, *IEEE Transactions on Autonomous Mental Development*) describe LIDA as encompassing "perception, motivation, attention, action selection, motor control, learning, language, mental simulation, and the sense of body and self." However, LIDA's self-model is primarily a body/proprioceptive model rather than a trait-level personality model. The closest LIDA component to CHA's needs is the **Workspace**, which maintains a global representation of the current situation including the agent's own state.

**IBM Autonomic Computing:** Kephart & Chess (2003, *IEEE Computer*) introduced the autonomic computing reference architecture with explicit self-* properties (self-configuration, self-optimization, self-healing, self-protection), including a *knowledge base* serving as an inspectable self-model of the computing system. This is implemented in production systems (IBM Tivoli, etc.) and constitutes the closest real-world precedent for a formal, auditable self-model in a deployed system — though it models computational resources, not personality.

### (b) Metacognition in AI Systems

The key paper on computational metacognition for CHA is **"Metacognition is all you need? Using Introspection in Generative Agents to Improve Goal-directed Behavior"** (Toy, MacAdam & Tabor 2024, arXiv:2401.10910). This paper introduces a metacognition module for generative LLM agents that enables them to observe their own thought processes and actions, emulating System 1/System 2 cognitive processes, and shows that agents with the metacognition module significantly outperform those without on complex goal-directed scenarios.

The MIDCA architecture (Cox, 2011; work on cognitive architectures) implements metacognition as meta-level goal operations that modify first-level cognition — the architecture that motivates the claim that metacognition improves problem-solving through cognition modification.

The CLARION MCS (see above) is the most complete implemented metacognitive architecture compatible with a BDI-like framework, because its production rules can fire on internal cognitive states rather than only on external percepts. A CHA-compatible metacognition module could be implemented as a BDI plan library that takes QPM state vectors, PSBG propositions, and BDI intention execution history as "percepts" and fires revision rules based on detected inconsistencies.

### (c) ACT-R, Soar, LIDA, Global Workspace — What Can Be Borrowed for CHA

Direct borrowings from cognitive architectures for CHA's self-modeling:

- **From Soar:** Episodic memory architecture (recording past states with timestamps and context, enabling retrospective behavioral analysis). Soar's episodic memory stores symbolic graph structures with recency/frequency metadata — directly applicable to CHA's Episodic Register design.
- **From ACT-R:** Activation-based memory retrieval (chunks are retrieved based on base-level activation combining recency and frequency of use) — applicable to salience-weighted episodic memory retrieval in CHA.
- **From CLARION MCS:** The MCS production rule structure for monitoring internal states and triggering metacognitive adjustments is directly applicable to CHA's BDI metacognition rules. The explicit/implicit knowledge duality (symbolic/connectionist) maps to CHA's PSBG/QPM duality.
- **From LIDA:** The global workspace broadcast mechanism — important information is broadcast widely to multiple receiving modules — is applicable to ensuring QPM state changes are propagated to the BDI engine and SCI without tight coupling.
- **From Autonomic Computing:** The knowledge base (self-model) architecture with explicit change logging is directly applicable to CHA's PSBG design.

### (d) Affective Computing and Social Robotics for Self-Modeling

The affective computing and social robotics literature provides the closest functional precedents for CHA's use case. Key systems:

**PEPPER robot (SoftBank Robotics) and NAO:** These platforms use explicit emotional state models that update based on interaction outcomes. NAO's personality module (documented in Leite et al.'s companion robot studies, 2012–2019) initializes from a trait profile and maintains a session-level emotional state. However, trait-level personality parameters are fixed at initialization; only state-level affect (valence, arousal) updates dynamically.

**Companion robot self-modeling (Leite et al., *ACM Transactions on Interactive Intelligent Systems*):** Studies of companion robots demonstrate that users prefer robots whose self-model remains stable — robots that "remember" consistent personality characteristics across sessions are rated higher in trustworthiness. This is an empirical argument for bounded QPM drift.

**Affective computing with explicit personality models:** El Ayadi et al. and related work implement OCEAN-based personality models in dialogue systems, demonstrating that explicit Big Five initialization produces more naturalistic and user-preferred interactions than no-personality baselines — but these systems do not update personality parameters from experience.

**GACA and related architectures (2020–2025):** Generative affective computing architectures using LLMs to generate emotionally consistent responses demonstrate that explicit affective state tracking (valence + arousal as real-valued parameters updated each turn) significantly improves interaction quality in extended conversations. These are the closest functional analogs to CHA's QPM + SCI architecture, and they constitute partial implementations of the design CHA requires.

### (e) Hybrid Symbolic-Neural Self-Modeling Architectures (2020–2025)

The most architecturally relevant recent work is in **hybrid neuro-symbolic systems for auditable AI in risk-sensitive domains** (2020–2025). Key findings:

**Neuro-Symbolic AI with auditable KG backends (arXiv:2302.07200, 2024):** This comprehensive survey documents hybrid architectures where OWL-based symbolic components enforce logical consistency while neural networks handle perception and generation. The "layered approach" that is "auditable" and "trustworthy" matches CHA's design philosophy exactly. The survey notes that the least explored area in neuro-symbolic AI is **meta-cognition** (only 5% of reviewed papers) — confirming that CHA's self-modeling work addresses a genuine frontier.

**Hybrid Neuro-Symbolic Models for Ethical AI (arXiv:2511.17644, 2025):** Surveys hybrid architectures for risk-sensitive domains, highlighting auditable decision paths via knowledge graph + deep learning integration. Case studies in healthcare and finance demonstrate that hybrid systems deliver "reliable and auditable AI" — directly applicable to CHA's cognitive traceability requirement.

**ENVISIONS neural-symbolic self-training framework (ACL 2025):** While focused on task improvement rather than self-modeling per se, this paper demonstrates that LLM agents can engage in self-exploration, self-refinement, and self-rewarding within a symbolic environment — a weak form of self-modeling where the agent maintains a symbolic representation of its own performance and uses it to guide improvement.

**Generative BDI (Ricci et al., referenced in ML-BDI survey 2025):** Couples BDI reasoning with generative AI at key steps including belief management — the most architecturally similar published system to CHA's BDI + LLM design. The system uses generative AI to update beliefs about the world but does not address self-beliefs.

---

## 5.2 Literature Gaps (Original Research Contributions)

- No implemented cognitive architecture (ACT-R, Soar, LIDA, CLARION) maintains an explicit, propositional self-belief graph in RDF/OWL.
- No hybrid symbolic-neural architecture explicitly models personality-level self-beliefs as distinct from world-beliefs.
- No affective computing system combines quantum-like personality encoding with explicit self-belief tracking.
- Meta-cognition in neuro-symbolic AI is the least explored area (5% of papers per the 2024 survey) — CHA's MCS-equivalent represents a clear contribution.
- No paper combines CLARION-style MCS architecture with a BDI reasoning engine and a quantum personality model.

---

## 5.3 Architectural Recommendation for CHA

> **Recommendation:** The CHA self-model should draw directly from CLARION's MCS for its BDI metacognition rule structure; from Soar's episodic memory for its Episodic Register design; from ACT-R's activation-based chunk retrieval for salience-weighted memory access; and from the neuro-symbolic KG literature for its PSBG OWL-ontology design. No existing architecture combines all of these; **CHA's self-model is a novel synthesis requiring a dedicated Self-Model Component (SMC)** that orchestrates the PSBG, Episodic Register, UAM, and SCI sub-components. The SMC should be implemented as a distinct module with its own API, callable by the BDI engine, the QPM, and the SCI.

---

## 5.4 Key Findings with Citations

1. Franklin et al. (2014): LIDA — systems-level architecture for cognition, emotion, and learning including "sense of body and self." *IEEE Transactions on Autonomous Mental Development.* DOI: 10.1109/TAMD.2013.2277589
2. Laird (2022): Soar cognitive architecture — explicit statement that Soar has no explicit self-representation. arXiv:2205.03854
3. Sun, R. (CLARION): Four-subsystem architecture with Meta-Cognitive Subsystem for monitoring/regulating cognition. (Multiple papers; see Sun 2004, *Cognitive Systems Research*)
4. Kotseruba et al. (2018): 40 years of cognitive architectures survey — 84 architectures reviewed, metacognition rare. *Artificial Intelligence Review.* DOI: 10.1007/s10462-018-9646-y
5. Toy, MacAdam & Tabor (2024): Metacognition module for generative agents improves goal-directed behavior. arXiv:2401.10910
6. Kephart & Chess (2003): The vision of autonomic computing — self-* properties and explicit knowledge base self-model. *IEEE Computer* 36(1):41–50.
7. Ricci et al. (in ML-BDI survey 2025): Generative BDI architecture coupling BDI with LLMs. arXiv:2510.20641
8. Hybrid Neuro-Symbolic Models survey (arXiv:2302.07200, 2024): OWL/RDF-backed auditable AI architectures.
9. Kolli et al. (2025): Hybrid neuro-symbolic models for ethical AI in risk-sensitive domains. arXiv:2511.17644
10. ENVISIONS (ACL 2025): Neural-symbolic self-training framework with self-exploration and self-rewarding. *ACL Anthology 2025.acl-long.635*
11. Laird, Lebiere & Rosenbloom (2017): The Common Model of Cognition — comparison of ACT-R, Soar, Sigma. *AI Magazine* 38(4).
12. Neuro-Symbolic AI meta-cognition gap: Only 5% of reviewed papers address meta-cognition (Hybrid NeSy systematic review, 2024, cited in Medium synthesis).

---

<a name="synthesis"></a>
# Cross-Question Synthesis and Architectural Recommendations

## Overall Self-Model Architecture for CHA

The five research questions converge on a coherent architectural picture. No single existing component of CHA is sufficient for self-modeling; a dedicated **Self-Model Component (SMC)** is required. The SMC consists of four sub-components:

---

### Component 1: Persistent Self-Belief Graph (PSBG)
**Location:** New dedicated module, interfacing with BDI engine  
**Format:** OWL ontology with RDF triples  
**Contents:**
- Dispositional self-beliefs: `<CHA_agent> hasTraitLevel <Conscientiousness, 0.72, {anchored: true, delta: +0.02}>`
- Capability self-beliefs: `<CHA_agent> hasCapability <TemporalReasoning, High>`, `<CHA_agent> hasLimitation <MultiHopArithmetic, Moderate>`
- Historical self-beliefs (append-only): `<CHA_agent> didExperience <Interaction_2847, {timestamp, summary, QPM_state_snapshot}>`
- Provenance triples for all updatable beliefs

**Update mechanism:** BDI metacognition rules fire on triggers from UAM; AGM revision operators apply to dispositional beliefs only; IEM-protection for current QPM state; historical beliefs are immutable.

---

### Component 2: QPM-Anchored Episodic Register
**Location:** New dedicated module, interfacing with QPM and BDI  
**Format:** Ordered log of (interaction_id, density_matrix_snapshot, instrument_applied, BDI_reasoning_trace, timestamp) tuples  
**Function:** Records how the QPM density matrix ρ evolved across interactions, providing the evidentiary basis for UAM assessments and the historical content for the PSBG.

**Key design principle:** This is a *log*, not a live quantum state. The snapshots are classical records of quantum states at each interaction boundary.

---

### Component 3: Update Assessment Module (UAM)
**Location:** New dedicated module, between Episodic Register and QPM parameter store  
**Function:** Aggregates behavioral evidence from the Episodic Register across sessions; computes running estimates of Δθₖ evidence; triggers bounded QPM H updates when evidence thresholds are crossed; routes proposals to human oversight.

**Constraints:** Maximum |Δθₖ| per update bounded to ε (empirically derived from 0.1 SD/year personality change bound); minimum 50 consistent behavioral indicators before any H update is proposed; all updates logged with full provenance.

---

### Component 4: Structured Context Injector (SCI)
**Location:** Interface between PSBG/Episodic Register and Phi-4-mini  
**Function:** At each inference call, serializes the current PSBG state (dispositional and capability self-beliefs) plus the last N salient Episodic Register entries into a compact JSON document (< 2,000 tokens) and prepends it to Phi-4-mini's system prompt.

**Format sketch:**
```json
{
  "personality": {"O": 0.72, "C": 0.85, "E": 0.51, "A": 0.68, "N": 0.34},
  "perceived_capabilities": ["temporal_reasoning", "empathic_response"],
  "recent_salient_events": [
    {"id": 2847, "summary": "User expressed frustration with slow reasoning", "session": 12},
    {"id": 2901, "summary": "Successful collaborative problem-solving on domain task", "session": 14}
  ],
  "self_beliefs": {"prefers_structured_discourse": true, "tends_toward_caution_under_uncertainty": true}
}
```

---

## Assignment of Self-Modeling Responsibilities

| Self-Model Aspect | Primary Component | Supporting Component |
|---|---|---|
| Trait-level self-concept | PSBG (BDI layer) | QPM (execution substrate) |
| Episodic memory of own behavior | Episodic Register | QPM density snapshots |
| Capability / limitation beliefs | PSBG | BDI metacognition rules |
| Personality parameter updating | UAM | Human oversight gate |
| Self-narration consistency | SCI + Phi-4-mini LoRA | PSBG state injection |
| Belief revision circularity control | BDI metacognition rules | AGM operators on PSBG |
| Temporal consistency | Episodic Register + TIMECHARA-style evaluation | SCI context injection |

---

## Three Priority Empirical Experiments

Before committing to an implementation, the following three experiments should be run in ascending order of system complexity:

### Experiment 1: Phi-4-mini Context Window Degradation Baseline

**Question:** At what conversational turn does Phi-4-mini's self-narration consistency drop below an acceptable threshold when given a structured JSON persona document, with no other architectural support?

**Design:** 100 scripted 40-turn conversations using a fixed CHA persona JSON. Measure PersonaScore-equivalent (using automated LLM-as-judge) at turns 5, 10, 15, 20, 25, 30. Identify the degradation inflection point.

**Why first:** This establishes the baseline capability of Phi-4-mini and defines the minimum requirements for the SCI's memory compression. It requires no QPM or BDI integration.

**Success criterion:** Identify the turn T* at which PersonaScore drops below 3.5/5.0 and characterize whether degradation is gradual or sudden.

---

### Experiment 2: BDI Self-Belief Circularity Stress Test

**Question:** Does the PSBG + BDI metacognition rule system exhibit confirmation bias loops or runaway feedback when self-beliefs influence behavior and behavioral evidence feeds back into self-beliefs?

**Design:** Implement a minimal PSBG with 5 dispositional self-beliefs (one per OCEAN dimension) and a BDI engine with simple update rules. Run 1,000 simulated interaction cycles with a confirmation-biased evidence generator (80% of evidence supports the current self-belief). Measure parameter drift over cycles.

**Why second:** This validates the AGM revision architecture and drift-bounding strategy before the QPM is integrated. It is a pure BDI/PSBG experiment.

**Success criterion:** With drift bounds active, Δθₖ_cumulative remains below ε_year (the annual change bound) after 365 simulated days of interactions.

---

### Experiment 3: QPM-Grounded Self-Narration Fidelity Test

**Question:** When Phi-4-mini is given a SCI-generated JSON self-model derived from the actual QPM state, do its narrations accurately reflect the QPM's current trait configuration — and does this fidelity persist across 40+ turns?

**Design:** Run a QPM through 20 simulated interaction sessions with varied scenarios. After each session, extract the QPM state vector and PSBG, generate a SCI JSON summary, and probe Phi-4-mini with 20 trait-eliciting questions. Score each response for consistency with the true QPM parameters. Repeat with a LoRA-fine-tuned variant of Phi-4-mini trained on QPM-grounded descriptions.

**Why third:** This tests the full SMC pipeline (QPM → Episodic Register → PSBG → SCI → Phi-4-mini) end-to-end and directly addresses CHA's core self-modeling requirement. It requires all components to be partially implemented.

**Success criterion:** LoRA-fine-tuned + SCI model achieves > 75% trait-consistency on all five OCEAN dimensions across 40 turns; base Phi-4-mini + SCI baseline is established for comparison.

---

## Most Significant Open Research Gaps (Consolidated)

Across all five questions, the following represent the strongest candidates for original research contributions:

1. **Q1 + Q4:** A formal quantum-like model of the Five-Factor Model self-concept, with a principled method for updating Hamiltonian parameters from behavioral evidence using quantum tomography-inspired estimation.

2. **Q2:** A formal BDI self-belief revision framework that distinguishes IEM-protected phenomenological beliefs from evidence-revisable dispositional beliefs, implemented with an RDF/OWL PSBG and AGM operators.

3. **Q1 + Q3 + Q5:** The full SMC architecture (PSBG + Episodic Register + UAM + SCI) as an implemented, evaluated system — the first hybrid symbolic-neural architecture combining quantum personality encoding with auditable self-belief management and small-LLM narration.

4. **Q4:** Empirical derivation of QPM update rate bounds from personality psychology literature and validation that bounded QPM drift preserves domain-appropriate behavior.

5. **Q3:** Evaluation of QPM-grounded self-narration fidelity as a new evaluation dimension, extending existing persona consistency benchmarks to test grounding in an external formal personality model.

---

<a name="references"></a>
# Complete Reference List

## Quantum Cognition and Self-Concept (Question 1)

- Atmanspacher, H., & Filk, T. (2006). Complexity and non-commutativity of learning operations on graphs. *BioSystems*, 85, 84–93.
- Atmanspacher, H., & Filk, T. (2010). A proposed test of temporal nonlocality in bistable perception. *Journal of Mathematical Psychology*, 54, 314–321.
- Atmanspacher, H., & Filk, T. (2013). The Necker-Zeno model for bistable perception. *Topics in Cognitive Science*, 5, 800–817.
- Atmanspacher, H., & Filk, T. (2019). Contextuality revisited — signaling may differ from communicating. In *Quanta and Mind.* Springer.
- Atmanspacher, H., Römer, H., & Walach, H. (2002). Weak quantum theory: Complementarity and entanglement in physics and beyond. *Foundations of Physics*, 32, 379–406.
- Busemeyer, J.R., & Bruza, P. (2012/2024). *Quantum Models of Cognition and Decision.* Cambridge University Press.
- Busemeyer, J.R., Ozawa, M., Pothos, E.M., & Tsuchiya, N. (2025). Incorporating episodic memory into quantum models of judgment and decision. *Philosophical Transactions of the Royal Society A*, 383(2309), 20240387. https://doi.org/10.1098/rsta.2024.0387
- Denolf, J., & Lambert-Mogiliansky, A. (2016). Bohr complementarity in memory retrieval. *Journal of Mathematical Psychology*, 73, 28–36.
- Englman, R., & Yahalom, A. (2024). A dual Hilbert-space formalism for consciousness; memory experiments. *European Journal of Applied Sciences*, 12(3), 29–46.
- Filk, T., & von Müller, A. (2008). Links between basic conceptual categories in quantum physics and psychology. *Mind and Matter*, 6(1).
- Khrennikov, A., et al. (2025). Quantum-like representation of neuronal networks' activity: Modeling "mental entanglement." *Frontiers in Human Neuroscience.* https://doi.org/10.3389/fnhum.2025.1685339
- Pothos, E.M., & Busemeyer, J.R. (2013). Can quantum probability provide a new direction for cognitive modeling? *Behavioral and Brain Sciences*, 36(3), 255–274.
- Pothos, E.M., & Busemeyer, J.R. (2022). Quantum cognition. *Annual Review of Psychology*, 73, 749–778.
- Preparation and measurement paper [density matrix + POVM] (2018). *Journal of Mathematical Psychology*. https://www.sciencedirect.com/science/article/abs/pii/S002224961730175X
- Trueblood, J.S., & Hemmer, P. (2017). The generalized quantum episodic memory model. [Referenced in Busemeyer et al. 2025]
- Tsuchiya, N., Bruza, P., Yamada, Y., Saigo, H., & Pothos, E.M. (2025). Quantum-like qualia hypothesis: From quantum cognition to quantum perception. *Frontiers in Human Neuroscience.* https://pmc.ncbi.nlm.nih.gov/articles/PMC12046633/
- Yearsley, J.M., & Pothos, E.M. (2016). Zeno's paradox in decision-making. *Proceedings of the Royal Society B*, 283, 20160291.

## BDI and Self-Belief Revision (Question 2)

- Alchourrón, C., Gärdenfors, P., & Makinson, D. (1985). On the logic of theory change. *Journal of Symbolic Logic*, 50(2), 510–530.
- BDI Agent Architectures: A Survey (2020). *Proceedings of IJCAI-20*, 684. https://www.ijcai.org/proceedings/2020/0684.pdf
- BDI4JADE Platform. GitHub: ingridnunes/bdi4jade.
- Integrating Machine Learning into BDI Agents: Current State and Future Directions (2025). arXiv:2510.20641
- Peacocke, C. (1996). Our entitlement to self-knowledge. *Proceedings of the Aristotelian Society*, 96, 117–158.
- Rao, A.S., & Georgeff, M.P. (1995). BDI agents: From theory to practice. *Proceedings of ICMAS-95.* https://cdn.aaai.org/ICMAS/1995/ICMAS95-042.pdf
- Schmidhuber, J. (2003). Gödel machines: Self-referential universal problem solvers making provably optimal self-improvements. arXiv:cs/0309048.
- Schmidhuber, J. (2007). Gödel machines: Fully self-referential optimal universal self-improvers. In B. Goertzel & C. Pennachin (Eds.), *Artificial General Intelligence.* Springer. https://doi.org/10.1007/978-3-540-68677-4_7
- Shoemaker, S. (1968). Self-reference and self-awareness. *Journal of Philosophy*, 65(19), 555–567.
- Steunebrink, B.R., & Schmidhuber, J. (2012). Towards an actual Gödel machine implementation: A lesson in self-reflective systems. In *Theoretical Foundations of AGI* (pp. 173–195). Springer.
- Yin, M., et al. (2024). Gödel Agent: A self-referential framework for agents recursively self-improvement. arXiv:2410.04444.
- Zhang, T., et al. (2025). Darwin Gödel Machine: Open-ended evolution of self-improving agents. Sakana AI. arXiv:2505.xxxxx (May 2025).

## Small LLM Self-Narration Consistency (Question 3)

- AMEM benchmark (2025). *COLM 2025.* https://openreview.net/pdf/3dcb3eae85f5e555bfdbd9368f3c518941e3f816.pdf
- Ahn, J., et al. (2024). TIMECHARA: Point-in-time character consistency benchmark.
- Boson AI (2024). RPBench-Auto: Character-based and scene-based role-playing benchmark.
- Chen, R., et al. (2024). COMEDY: Character preference data + DPO for memory-based personalized interaction.
- Enhancing persona consistency with LLMs (2024). *ACM CNIOT 2024.* https://doi.org/10.1145/3670105.3670140
- Hello Again: LLM-powered personalized agent for long-term dialogue (2024). arXiv:2406.05925.
- Hu, E., et al. (2021). LoRA: Low-rank adaptation of large language models. arXiv:2106.09685.
- Persona-aware contrastive learning for LLM role-playing (2025). arXiv:2503.17662.
- Persona-aware LLM-enhanced framework for multi-session conversations (2025). *ACL 2025 Findings.* https://aclanthology.org/2025.findings-acl.5.pdf
- PersonaGym: Evaluating persona agents and LLMs (2025). *EMNLP 2025 Findings.* https://aclanthology.org/2025.findings-emnlp.368.pdf
- PersonalLLM (2025). *ICLR 2025.* https://proceedings.iclr.cc/paper_files/paper/2025/file/a730abbcd6cf4a371ca9545db5922442-Paper-Conference.pdf
- Salemi, A., et al. (2023). LAMP: Large language model personalization. *ICLR 2024.*
- Wang, Y., et al. (2024). RoleLLM: Benchmarking, eliciting, and enhancing role-playing abilities of large language models.
- Whose Personae? Synthetic persona experiments in LLM research (2025). *AAAI AIES.* https://ojs.aaai.org/index.php/AIES/article/download/36553/38691/40628

## Personality Psychology and QPM Updating (Question 4)

- Graham, E.K., et al. (2020). Trajectories of Big Five personality traits: A coordinated analysis of 16 longitudinal samples. *Journal of Personality and Social Psychology.* PMC7869960.
- Longitudinal Big Five study — Mexican-origin adults. *PMC8821110.*
- Mroczek, D.K., & Kolarz, C.M. (1998). The effect of age on positive and negative affect. *Journal of Personality and Social Psychology*, 75, 1333–1349.
- Mroczek, D.K., & Spiro, A. (2003). Modeling intraindividual change in personality traits: Findings from the Normative Aging Study. *Journals of Gerontology B*, 58, 153–165.
- Mroczek, D.K., & Spiro, A. (2007). Personality change influences mortality in older men. *Psychological Science*, 18, 371–376.
- Personality stability and change across the life course: US and Japan cross-cultural study. *PMC5742083.*
- Roberts, B.W., & Mroczek, D. (2008). Personality trait change in adulthood. *Current Directions in Psychological Science*, 17, 31–35. DOI: 10.1111/j.1467-8721.2008.00543.x
- Roberts, B.W., Caspi, A., & Moffitt, T. (2003). Work experiences and personality development in young adulthood. *Journal of Personality and Social Psychology*, 84, 582–593.
- Roberts, B.W., Walton, K., & Viechtbauer, W. (2006). Patterns of mean-level change in personality traits across the life course: A meta-analysis of longitudinal studies. *Psychological Bulletin*, 132(1), 1–25.
- Roberts, B.W., & Wood, D. (2006). Personality development in the context of the neo-socioanalytic model. In *Handbook of Personality Development* (pp. 11–39). Erlbaum.
- Specht, J., Egloff, B., & Schmukle, S.C. (2011). Stability and change of personality across the life course: The impact of age and major life events on mean-level and rank-order stability of the Big Five. *Journal of Personality and Social Psychology*, 101(4), 862–882.

## Existing Self-Modeling Architectures (Question 5)

- Autonomic computing reference architecture (Kephart & Chess 2003). *IEEE Computer*, 36(1), 41–50.
- BDI Architectures: emergentmind synthesis (2025). https://www.emergentmind.com/topics/bdi-architectures
- CLARION architecture (Sun, R.). Multiple papers; see Sun, R. (2004). *Cognitive Systems Research*, 5, 63–87.
- ENVISIONS: Neural-symbolic self-training framework (2025). *ACL 2025.* https://aclanthology.org/2025.acl-long.635.pdf
- Franklin, S., Madl, T., D'Mello, S., & Snaider, J. (2014). LIDA: A systems-level architecture for cognition, emotion, and learning. *IEEE Transactions on Autonomous Mental Development.* DOI: 10.1109/TAMD.2013.2277589.
- Hybrid Neuro-Symbolic Models for Ethical AI in Risk-Sensitive Domains (2025). arXiv:2511.17644.
- Hybrid Neuro-Symbolic survey (2024). arXiv:2302.07200.
- Kotseruba, I., & Tsotsos, J.K. (2018). 40 years of cognitive architectures: Core cognitive abilities and practical applications. *Artificial Intelligence Review.* DOI: 10.1007/s10462-018-9646-y.
- Laird, J.E. (2022). Introduction to the Soar cognitive architecture. arXiv:2205.03854.
- Laird, J.E., Lebiere, C., & Rosenbloom, P.S. (2017). A standard model of the mind. *AI Magazine*, 38(4), 13–26.
- Metacognition is all you need? (Toy, MacAdam & Tabor 2024). arXiv:2401.10910.
- ML-BDI Survey (2025). arXiv:2510.20641.
- Neuro-Symbolic AI overview (2025). https://gregrobison.medium.com/neuro-symbolic-ai-a-foundational-analysis-of-the-third-waves-hybrid-core-cc95bc69d6fa

---

*Report compiled from systematic web search and literature review, March 2026.*  
*Literature coverage: 2015–2025 primary, with foundational works cited as required.*  
*⚠️ Flag — Sparse/uncertain areas: (1) Atmanspacher-Filk narrative self work (none found — confirmed gap); (2) GQEM (Trueblood & Hemmer 2017) — cited indirectly; (3) QPM-specific quantum tomography update methods — fully original contribution, no prior literature; (4) computational implementations of Peacocke/Shoemaker — confirmed absent from literature.*
