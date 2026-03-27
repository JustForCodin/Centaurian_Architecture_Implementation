# Centaurian Hybrid Architecture — Implementation

Empirical validation of the **Centaurian Hybrid Architecture (CHA)**, a multi-layered AI system that pairs a symbolic/quantum-like cognitive core with lightweight neural periphery (small language models, neural TTS, procedural animation). The architecture encodes personality via a Quantum Personality Model (QPM) running on classical hardware and confines neural networks to bounded I/O transduction roles, preserving end-to-end traceability of every behavioral decision.

Full architecture specification: [`CHA_Experiment_1/Centaurian_Hybrid_Architecture_v2.md`](CHA_Experiment_1/Centaurian_Hybrid_Architecture_v2.md)

## Experiment 1 — SCI Persona Degradation Baseline

**Goal:** Measure how long a small language model can maintain a consistent persona when given a Structured Cognitive Identity (SCI) — a JSON self-model defining personality traits, episodic memories, capabilities, and communication style. Find the degradation inflection point T\* (first probe turn where mean PersonaScore drops below 3.5).

**Method:**
- 30 scripted dialogues (22 naturalistic + 8 adversarial), each 40 turns
- Side-channel probe questions at turns 5, 10, 15, … 40 across 4 dimensions (Trait, Episodic, Capability, Style)
- Primary judge: Claude Haiku 4.5; secondary judge: Claude Sonnet 4.5 (inter-rater reliability via Cohen's kappa)

### Models tested

| Model | Params | T\* | Mean PersonaScore | Outcome |
|-------|--------|-----|-------------------|---------|
| Phi-4-mini | 3.8B | 5 (immediate) | 1.08 / 5.0 | Capability failure — gibberish in 93% of scripts |
| Qwen2.5-7B | 7B | 5 | 3.16 → 2.96 over 40 turns | Coherent but below threshold; piecewise degradation from turn 15 |

### Key files

```
CHA_Experiment_1/
├── Centaurian_Hybrid_Architecture_v2.md   # Full architecture paper
├── EXPERIMENT_REPORT.md                   # Detailed experiment report
├── experiment_runner.py                   # Main experiment pipeline
├── generate_scripts.py                    # Template-based script generator
├── analyse_results.py                     # Analysis and visualization
├── interrater_check.py                    # Inter-rater reliability checker
├── CHA_Experiment1_Colab.ipynb            # Google Colab notebook
├── logs_qwen2.5_7b/                      # Raw score & context logs
└── results_qwen2.5_7b/                   # Charts, fits, summary report
```

## Running the experiment

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) with the target model pulled, an Anthropic API key.

```bash
cd CHA_Experiment_1
pip install ollama anthropic python-dotenv
# Set your API key
echo "CHA_EXPERIMENT_SONNET_KEY=sk-..." > .env

# Generate scripts, run experiment, analyse
python generate_scripts.py
python experiment_runner.py --model qwen2.5:7b
python analyse_results.py
```

Or use the provided Colab notebook for GPU-accelerated runs.

## License

See repository for license details.
