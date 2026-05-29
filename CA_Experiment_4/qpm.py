"""
Quantum Personality Model (QPM) — 12-qubit Qiskit Aer circuit.

Implements the QPM from Centaurian Architecture v3 §3:
- 11 trait qubits (q0–q10) + 1 ancilla (q11)
- Ry initialization from personality profile s_k values  (§3.4)
- Intra-domain CNOT entanglement layer (5 gates)           (§3.5.3)
- Inter-domain CRz entanglement layer (8 gates)            (§3.5.3, Table 7)
- Per-step context Ry rotations from d-vector (10 gates)   (§3.5.4)
- Lindblad noise via Aer NoiseModel                        (§3.2, Table 4)
- 1024-shot measurement → marginals + purity proxy         (§3.5.6)

For multi-step d-vector sequences (Battery A order-effect test), each d-vector
is applied as a successive context-rotation layer in a single circuit, producing
non-commutative order effects (Ry(A)·Ry(B) ≠ Ry(B)·Ry(A)).
"""

from __future__ import annotations

import math
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    amplitude_damping_error,
    phase_damping_error,
    depolarizing_error,
)

# ── Qubit layout ──────────────────────────────────────────────────────────

QUBIT_LABELS = [
    "O_exp", "O_int", "O_val",   # q0-q2  Openness
    "C_ind", "C_ord",             # q3-q4  Conscientiousness
    "E_ent", "E_ass",             # q5-q6  Extraversion
    "A_com", "A_pol",             # q7-q8  Agreeableness
    "N_vol", "N_wth",             # q9-q10 Neuroticism
]                                 # q11    Ancilla (entanglement mediator)

N_TRAIT_QUBITS = 11
N_TOTAL_QUBITS = 12
ANCILLA_IDX = 11

# ── Decoherence parameters — Table 4 of v3 paper ─────────────────────────
# (gamma_k, lambda_k) = (amplitude-damping rate, phase-damping rate)

_DECOHERENCE: dict[str, tuple[float, float]] = {
    "O_exp": (0.02, 0.01),
    "O_int": (0.02, 0.01),
    "O_val": (0.02, 0.01),
    "C_ind": (0.01, 0.005),
    "C_ord": (0.01, 0.005),
    "E_ent": (0.03, 0.02),
    "E_ass": (0.03, 0.02),
    "A_com": (0.02, 0.015),
    "A_pol": (0.02, 0.015),
    "N_vol": (0.05, 0.04),
    "N_wth": (0.05, 0.04),
}

_MU = 0.1        # pressure coupling constant for ancilla
_DELTA_T = 1.0   # one conversational turn

# ── Context-coupling constants — §3.5.4 ──────────────────────────────────

DELTA = 0.3  # δᵢ = 0.3 for all domains (initial calibration value)

# Each entry: (qubit_idx, d_component_idx)  — d is 0-indexed [d1..d5] = [0..4]
_CONTEXT_COUPLINGS: list[tuple[int, int]] = [
    (0,  3),   # q0  O_exp  ← d4 Ambiguity
    (1,  3),   # q1  O_int  ← d4
    (3,  1),   # q3  C_ind  ← d2 Task Orientation
    (4,  1),   # q4  C_ord  ← d2
    (5,  0),   # q5  E_ent  ← d1 Affective Intensity
    (6,  0),   # q6  E_ass  ← d1
    (7,  2),   # q7  A_com  ← d3 Social Normative Constraint
    (8,  2),   # q8  A_pol  ← d3
    (9,  4),   # q9  N_vol  ← d5 Temporal Pressure
    (10, 4),   # q10 N_wth  ← d5
]

# ── Intra-domain CNOT layer — §3.5.3 ─────────────────────────────────────

_INTRADOMAIN_CNOTS: list[tuple[int, int]] = [
    (0,  1),   # O_exp → O_int   within-ρ ≈ 0.35
    (3,  4),   # C_ind → C_ord   within-ρ ≈ 0.45
    (5,  6),   # E_ent → E_ass   within-ρ ≈ 0.40
    (7,  8),   # A_com → A_pol   within-ρ ≈ 0.38
    (9,  10),  # N_vol → N_wth   within-ρ ≈ 0.45
]

# ── Inter-domain CRz layer — Table 7 ─────────────────────────────────────
# (control_idx, target_idx, empirical_rho)
# φ = arcsin(ρ)·π  per §3.5.2
# Negative ρ → X gate on target before and after CRz(|φ|)  per §3.5.2

_INTERDOMAIN_EDGES: list[tuple[int, int, float]] = [
    (3,  9,  -0.43),   # C_ind → N_vol   Stability cluster
    (4,  10, -0.43),   # C_ord → N_wth   Stability cluster
    (7,  9,  -0.36),   # A_com → N_vol   Stability cluster
    (8,  10, -0.36),   # A_pol → N_wth   Stability cluster
    (3,  7,  +0.43),   # C_ind → A_com   Stability cluster
    (0,  5,  +0.43),   # O_exp → E_ent   Plasticity cluster
    (5,  9,  -0.36),   # E_ent → N_vol   Cross-factor
    (6,  10, -0.36),   # E_ass → N_wth   Cross-factor
]


# ── Helpers ───────────────────────────────────────────────────────────────

def _theta(s_k: float) -> float:
    """Ry rotation angle for aspect score s_k: θ = 2·arcsin(√s_k)."""
    s = float(np.clip(s_k, 1e-9, 1.0 - 1e-9))
    return 2.0 * math.asin(math.sqrt(s))


def _phi(rho: float) -> float:
    """CRz phase angle from empirical correlation: φ = arcsin(ρ)·π."""
    return math.asin(float(np.clip(rho, -1.0, 1.0))) * math.pi


def _build_noise_model(d5: float = 0.5) -> NoiseModel:
    """Lindblad-discretized noise model (amplitude damping + phase damping).

    Both channels are composed into a single QuantumError per qubit before
    being added to the NoiseModel, avoiding Qiskit Aer's duplicate-channel
    warning that fires when two errors are added to the same gate+qubit.
    """
    nm = NoiseModel()
    for idx, label in enumerate(QUBIT_LABELS):
        gamma_k, lambda_k = _DECOHERENCE[label]
        combined = amplitude_damping_error(gamma_k * _DELTA_T).compose(
            phase_damping_error(lambda_k * _DELTA_T)
        )
        nm.add_quantum_error(combined, ["ry", "id"], [idx])
    # Temporal-pressure depolarising noise on ancilla
    pressure = float(np.clip(d5, 0.0, 1.0))
    nm.add_quantum_error(
        depolarizing_error(_MU * pressure * _DELTA_T, 1), ["id"], [ANCILLA_IDX]
    )
    return nm


# ── QPM class ─────────────────────────────────────────────────────────────

class QPM:
    """Quantum Personality Model.

    Parameters
    ----------
    profile : dict[str, float]
        Aspect scores s_k ∈ [0,1] keyed by QUBIT_LABELS.
        Missing keys default to 0.5.
    n_shots : int
        Default measurement shot count.
    """

    def __init__(self, profile: dict[str, float], n_shots: int = 1024):
        self.profile = {
            k: float(np.clip(profile.get(k, 0.5), 0.0, 1.0))
            for k in QUBIT_LABELS
        }
        self.n_shots = n_shots

    # ── Circuit builder ───────────────────────────────────────────────────

    def _build_circuit(self, d_sequence: list[list[float]]) -> QuantumCircuit:
        """Build QPM circuit for a (possibly multi-step) d-vector sequence.

        Multiple d-vectors → successive Ry context layers → non-commutative
        order effects.  Single d-vector → standard single-turn QPM call.
        """
        qr = QuantumRegister(N_TOTAL_QUBITS, "q")
        cr = ClassicalRegister(N_TRAIT_QUBITS, "meas")
        qc = QuantumCircuit(qr, cr)

        # Stage 1 — Ry initialization from personality profile
        for idx, label in enumerate(QUBIT_LABELS):
            qc.ry(_theta(self.profile[label]), qr[idx])
        qc.barrier()

        # Stage 2 — Intra-domain CNOT correlation layer
        for ctrl, tgt in _INTRADOMAIN_CNOTS:
            qc.cx(qr[ctrl], qr[tgt])
        qc.barrier()

        # Stage 3 — Inter-domain CRz entanglement layer
        for ctrl, tgt, rho in _INTERDOMAIN_EDGES:
            phi = _phi(rho)
            if rho < 0:
                # Anti-correlation encoding: X-CRz(|φ|)-X on target  (§3.5.2)
                qc.x(qr[tgt])
                qc.crz(abs(phi), qr[ctrl], qr[tgt])
                qc.x(qr[tgt])
            else:
                qc.crz(phi, qr[ctrl], qr[tgt])
        qc.barrier()

        # Stage 4+ — Context-rotation layers (one per d-vector step)
        for d in d_sequence:
            d_clipped = [float(np.clip(v, 0.0, 1.0)) for v in d]
            for qubit_idx, d_idx in _CONTEXT_COUPLINGS:
                qc.ry(DELTA * d_clipped[d_idx], qr[qubit_idx])
            qc.barrier()

        # Lindblad noise carriers (identity gates with attached noise)
        for idx in range(N_TOTAL_QUBITS):
            qc.id(qr[idx])

        # Measurement — trait qubits only (ancilla excluded)
        for idx in range(N_TRAIT_QUBITS):
            qc.measure(qr[idx], cr[idx])

        return qc

    # ── Run ───────────────────────────────────────────────────────────────

    def run(
        self,
        d_sequence: list | list[list[float]],
        n_shots: int | None = None,
    ) -> dict:
        """Execute QPM and return marginals + diagnostics.

        Parameters
        ----------
        d_sequence : list[float] | list[list[float]]
            Single d-vector [d1, d2, d3, d4, d5] or a sequence for multi-step
            order-effect testing (each element is one 5-dim step).
        n_shots : int | None
            Override default shot count.

        Returns
        -------
        dict
            marginals     : dict[str, float]  — p̂_k per trait qubit
            counts        : dict[str, int]    — raw bitstring → count
            purity_approx : float             — 1 − mean_k[p̂_k² + (1−p̂_k)²]
            n_shots       : int
        """
        shots = n_shots or self.n_shots

        # Normalise: single d-vector → one-element sequence
        if d_sequence and not isinstance(d_sequence[0], (list, tuple, np.ndarray)):
            d_sequence = [list(d_sequence)]

        # d5 of last step drives ancilla pressure noise
        d5 = float(d_sequence[-1][4]) if d_sequence else 0.5

        circuit = self._build_circuit(d_sequence)
        sim = AerSimulator(noise_model=_build_noise_model(d5))
        result = sim.run(circuit, shots=shots).result()
        counts_raw: dict[str, int] = result.get_counts(circuit)

        # Compute per-qubit marginals
        # Qiskit bitstrings are big-endian: rightmost character = q0
        total = sum(counts_raw.values())
        marginals: dict[str, float] = {lbl: 0.0 for lbl in QUBIT_LABELS}
        for bitstring, count in counts_raw.items():
            bs = bitstring.replace(" ", "")
            n_bits = len(bs)
            for idx, label in enumerate(QUBIT_LABELS):
                # q{idx} maps to bs[n_bits - 1 - idx]
                pos = n_bits - 1 - idx
                if 0 <= pos < n_bits and bs[pos] == "1":
                    marginals[label] += count
        for lbl in QUBIT_LABELS:
            marginals[lbl] /= total

        purity_approx = _purity_approx(marginals)

        return {
            "marginals": marginals,
            "counts": counts_raw,
            "purity_approx": purity_approx,
            "n_shots": total,
        }

    # ── Entropy ───────────────────────────────────────────────────────────

    def entropy(
        self,
        d_sequence: list | list[list[float]],
        n_shots: int | None = None,
    ) -> float:
        """Mean Bernoulli entropy across trait qubits (Battery B metric).

        H = -mean_k[ p̂_k·ln(p̂_k) + (1−p̂_k)·ln(1−p̂_k) ]  ∈ [0, ln 2]

        Directly comparable with CMG-CDK which uses the same formula over its
        marginals.  Avoids the histogram degeneracy that arises from QPM's
        binary shot outcomes.
        """
        result = self.run(d_sequence, n_shots=n_shots)
        entropies = []
        for lbl in QUBIT_LABELS:
            p = float(np.clip(result["marginals"][lbl], 1e-10, 1.0 - 1e-10))
            entropies.append(-p * math.log(p) - (1.0 - p) * math.log(1.0 - p))
        return float(np.mean(entropies))


# ── Shared purity proxy (also used by CMG-CDK) ───────────────────────────

def _purity_approx(marginals: dict[str, float]) -> float:
    """C̄_approx = 1 − mean_k[p̂_k² + (1−p̂_k)²].

    = 0 when all p̂_k = 0 or 1 (pure state).
    = 0.5 when all p̂_k = 0.5 (maximally uncertain / ambivalent).
    """
    per_qubit = [
        marginals[lbl] ** 2 + (1.0 - marginals[lbl]) ** 2
        for lbl in QUBIT_LABELS
    ]
    return 1.0 - float(np.mean(per_qubit))
