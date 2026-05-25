"""
CMG-CDK: Correlated Multivariate Gaussian with Context-Dependent Kernel.

Classical control model for CA Experiment 3 (plan §4.1, Appendix A).

Matched to QPM on every dimension except quantum-specific mechanisms:
  C1 — Trait correlations: same ρ matrix → same covariance Σ       [MATCHED]
  C2 — Context sensitivity: same δ constants, linear additive shift  [MATCHED]
  C3 — Ambivalence / superposition: NOT present in CMG-CDK          [DIFFERS]
  Order effects: linear mean-shift is commutative; QPM Ry is not    [DIFFERS]

Any measured difference between QPM and CMG-CDK outputs is therefore
attributable specifically to quantum-like mechanisms (superposition,
non-commutative evolution, coherence) rather than parameter count,
initialization, or context sensitivity.
"""

from __future__ import annotations

import numpy as np

from qpm import QUBIT_LABELS, N_TRAIT_QUBITS, DELTA, _CONTEXT_COUPLINGS, _purity_approx

# ── Correlation matrix construction — Appendix A ─────────────────────────

# Inter-domain ρ (Table 6, van der Linden et al. 2010)
_DOMAIN_RHO: dict[tuple[str, str], float] = {
    ("C", "O"): 0.20,
    ("E", "O"): 0.43,
    ("A", "O"): 0.21,
    ("N", "O"): -0.17,
    ("E", "C"): 0.29,
    ("A", "C"): 0.43,
    ("N", "C"): -0.43,
    ("A", "E"): 0.26,
    ("N", "E"): -0.36,
    ("N", "A"): -0.36,
}

# Within-domain ρ (DeYoung et al. 2007, Appendix A)
_WITHIN_RHO: dict[str, float] = {
    "O": 0.35,
    "C": 0.45,
    "E": 0.40,
    "A": 0.38,
    "N": 0.45,
}

_QUBIT_DOMAIN: dict[str, str] = {
    "O_exp": "O", "O_int": "O", "O_val": "O",
    "C_ind": "C", "C_ord": "C",
    "E_ent": "E", "E_ass": "E",
    "A_com": "A", "A_pol": "A",
    "N_vol": "N", "N_wth": "N",
}


def build_sigma() -> np.ndarray:
    """Build the 11×11 aspect-level correlation matrix Σ.

    Cross-domain entries distribute inter-domain ρ equally across aspect pairs.
    Applies Higham's nearest-PD correction if the matrix is not PD.
    """
    n = N_TRAIT_QUBITS
    S = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            d_i = _QUBIT_DOMAIN[QUBIT_LABELS[i]]
            d_j = _QUBIT_DOMAIN[QUBIT_LABELS[j]]
            if d_i == d_j:
                rho = _WITHIN_RHO[d_i]
            else:
                key = tuple(sorted([d_i, d_j]))
                rho = _DOMAIN_RHO.get(key, 0.0)  # type: ignore[arg-type]
            S[i, j] = S[j, i] = rho
    eigvals = np.linalg.eigvalsh(S)
    if eigvals.min() <= 0.0:
        S = _nearest_pd(S)
    return S


def _nearest_pd(A: np.ndarray) -> np.ndarray:
    """Higham (1988) nearest positive-definite matrix."""
    B = (A + A.T) / 2.0
    _, s, Vt = np.linalg.svd(B)
    H = Vt.T @ np.diag(s) @ Vt
    A2 = (B + H) / 2.0
    A3 = (A2 + A2.T) / 2.0
    eps = np.finfo(float).eps
    k = 1
    while not _is_pd(A3):
        mineig = np.linalg.eigvalsh(A3).min()
        A3 += (-mineig * k ** 2 + eps * k) * np.eye(A.shape[0])
        k += 1
    return A3


def _is_pd(B: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def build_W() -> np.ndarray:
    """Build 11×5 context-coupling matrix W.

    W[qubit_idx, d_idx] = δ = 0.3, matching QPM's Ry(δ·d_j) rotations.
    """
    W = np.zeros((N_TRAIT_QUBITS, 5))
    for qubit_idx, d_idx in _CONTEXT_COUPLINGS:
        W[qubit_idx, d_idx] = DELTA
    return W


# Pre-built shared matrices (module-level for efficiency)
SIGMA: np.ndarray = build_sigma()
W: np.ndarray = build_W()


# ── CMG-CDK class ─────────────────────────────────────────────────────────

class CMG_CDK:
    """Correlated Multivariate Gaussian with Context-Dependent Kernel.

    Parameters
    ----------
    profile : dict[str, float]
        Same s_k values as QPM — used as initial mean μ₀.
    n_samples : int
        Samples drawn per run (matched to QPM n_shots = 1024).
    rng_seed : int | None
        Optional RNG seed for reproducible runs.
    """

    def __init__(
        self,
        profile: dict[str, float],
        n_samples: int = 1024,
        rng_seed: int | None = None,
    ):
        self.mu0 = np.array([
            float(np.clip(profile.get(lbl, 0.5), 0.0, 1.0))
            for lbl in QUBIT_LABELS
        ])
        self.n_samples = n_samples
        self._rng = np.random.default_rng(rng_seed)

    def _apply_sequence(self, d_sequence: list[list[float]]) -> np.ndarray:
        """Compute updated mean μ_T after applying all d-vector steps linearly.

        μ_t = clip(μ_{t-1} + W·d_t, 0, 1)   for each step t.

        Note: this update IS commutative (A→B and B→A give the same final μ),
        which is the key structural difference from QPM's non-commutative Ry
        evolution.
        """
        mu = self.mu0.copy()
        for d in d_sequence:
            d_arr = np.array([float(np.clip(v, 0.0, 1.0)) for v in d])
            mu = np.clip(mu + W @ d_arr, 0.0, 1.0)
        return mu

    def run(
        self,
        d_sequence: list | list[list[float]],
        n_samples: int | None = None,
    ) -> dict:
        """Draw samples and return marginals in QPM-compatible format.

        Parameters
        ----------
        d_sequence : list[float] | list[list[float]]
            Single d-vector or a sequence for multi-step testing.
        n_samples : int | None
            Override default sample count.

        Returns
        -------
        dict
            marginals     : dict[str, float]   — mean of each trait dimension
            samples       : np.ndarray (n, 11) — raw sample matrix
            purity_approx : float              — same formula as QPM
            n_samples     : int
        """
        n = n_samples or self.n_samples

        if d_sequence and not isinstance(d_sequence[0], (list, tuple, np.ndarray)):
            d_sequence = [list(d_sequence)]

        mu = self._apply_sequence(d_sequence)
        # Σ scaled to 0.01 to match QPM measurement noise regime
        samples = self._rng.multivariate_normal(mu, SIGMA * 0.01, size=n)
        samples = np.clip(samples, 0.0, 1.0)

        marginals = {
            QUBIT_LABELS[k]: float(np.mean(samples[:, k]))
            for k in range(N_TRAIT_QUBITS)
        }
        return {
            "marginals": marginals,
            "samples": samples,
            "purity_approx": _purity_approx(marginals),
            "n_samples": n,
        }

    def entropy(
        self,
        d_sequence: list | list[list[float]],
        n_samples: int | None = None,
        n_bins: int = 20,
    ) -> float:
        """Mean Shannon entropy of the output distribution (Battery B metric).

        Matched exactly to QPM.entropy() for direct comparison.
        """
        n = n_samples or self.n_samples
        if d_sequence and not isinstance(d_sequence[0], (list, tuple, np.ndarray)):
            d_sequence = [list(d_sequence)]

        mu = self._apply_sequence(d_sequence)
        samples = self._rng.multivariate_normal(mu, SIGMA * 0.01, size=n)
        samples = np.clip(samples, 0.0, 1.0)

        entropies = []
        for k in range(N_TRAIT_QUBITS):
            hist, _ = np.histogram(
                samples[:, k], bins=n_bins, range=(0.0, 1.0), density=True
            )
            h = hist[hist > 0]
            entropies.append(-float(np.sum(h * np.log(h + 1e-10))) / n_bins)
        return float(np.mean(entropies))
