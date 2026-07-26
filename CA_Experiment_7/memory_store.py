"""Episodic-RAG session memory for Experiment 7 (plan §5.2 / §5.3).

An in-session store of past `(user, assistant)` exchanges, embedded with
all-MiniLM-L6-v2. At each new turn it returns a compact memory block, capped at a
fixed token budget, built one of two ways:

  * `retrieve(query, ...)` — semantic top-k over the whole session (condition C2).
  * `recent(...)`          — the last-N exchanges, no retrieval (condition C1).

Both fill the *same* token budget, so C1 vs C2 isolates retrieval (semantic,
whole-history) from recency (last-N) rather than how much context each gets. The
budget cap is the knob RQ4/H4 measure — it keeps per-turn injected context bounded
regardless of how large the store grows.

There is no `<|memory|>` special token in the frozen Experiment 6 tokenizer, so the
memory block is injected **inline into the context channel** (a natural-language
"Relevant earlier exchanges:" header the model already knows how to ground on),
prepended to the turn's own retrieved passage. A memory-native `<|memory|>` channel
would require retraining the tokenizer — noted as a Phase-B option in the plan.

The embedder is injectable (`SessionMemory(embedder=fn)`) so the retrieval/budget
logic is testable offline without loading sentence-transformers.
"""

from __future__ import annotations

import numpy as np

MEMORY_HEADER = "Relevant earlier exchanges (from your session memory):"


def _default_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                      device: str | None = None):
    """Lazily build an all-MiniLM-L6-v2 bi-encoder → callable(list[str]) -> (n, d) unit vectors."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)

    def embed(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False),
                          dtype=np.float32)
    return embed


class SessionMemory:
    """Per-conversation episodic store. Reset (new instance) per script × condition."""

    def __init__(self, embedder=None, reranker=None):
        self.turns: list[dict] = []          # {turn_id, user, assistant, text}
        self._emb: np.ndarray | None = None  # (n, d) unit vectors, aligned to self.turns
        self._embedder = embedder            # callable(list[str]) -> (n, d); lazy if None
        self._reranker = reranker            # optional callable(query, list[str]) -> list[float]

    # ── indexing ────────────────────────────────────────────────────────
    def _ensure_embedder(self):
        if self._embedder is None:
            self._embedder = _default_embedder()

    def add(self, turn_id: int, user: str, assistant: str) -> None:
        """Index one completed exchange."""
        self._ensure_embedder()
        text = f"{user}\n{assistant}".strip()
        vec = self._embedder([text])                       # (1, d)
        self.turns.append({"turn_id": turn_id, "user": user or "",
                           "assistant": assistant or "", "text": text})
        self._emb = vec if self._emb is None else np.vstack([self._emb, vec])

    def __len__(self) -> int:
        return len(self.turns)

    # ── snippet formatting + budget ─────────────────────────────────────
    @staticmethod
    def _snippet(rec: dict) -> str:
        u = " ".join(str(rec["user"]).split())
        a = " ".join(str(rec["assistant"]).split())
        return f"(turn {rec['turn_id']}) Alex asked: {u} — you answered: {a}"

    def _pack(self, recs: list[dict], budget_tokens: int, tok) -> tuple[str, list[int]]:
        """Greedily pack snippets (most-relevant/most-recent first) under the token
        budget; the last snippet is truncated to fit rather than dropped whole.
        `tok` is an ADATokenizer (or None → 4-chars/token estimate)."""
        def ntok(s: str) -> int:
            return len(tok.encode(s)) if tok is not None else max(1, len(s) // 4)

        header = MEMORY_HEADER
        used, lines, budget_left = [], [], budget_tokens - ntok(header) - 1
        for rec in recs:
            line = self._snippet(rec)
            cost = ntok(line) + 1
            if cost <= budget_left:
                lines.append(line); used.append(rec["turn_id"]); budget_left -= cost
            elif not lines:                                # always emit at least a truncated top hit
                approx_chars = max(0, budget_left) * (1 if tok is not None else 4) * 4
                lines.append(line[:max(40, approx_chars)]); used.append(rec["turn_id"]); break
            else:
                break
        if not lines:
            return "", []
        return header + "\n" + "\n".join(lines), used

    # ── retrieval (C2) ──────────────────────────────────────────────────
    def retrieve(self, query: str, k: int, budget_tokens: int, tok=None,
                 rerank: bool = False) -> tuple[str, list[int]]:
        """Semantic top-k over the whole session → packed memory block."""
        if not self.turns:
            return "", []
        self._ensure_embedder()
        q = self._embedder([query])[0]                     # (d,)
        sims = self._emb @ q                               # cosine (unit vectors)
        order = np.argsort(-sims)[: max(k, 1) if not rerank else max(k * 3, k)]
        cands = [self.turns[i] for i in order]
        if rerank and self._reranker is not None and len(cands) > 1:
            scores = self._reranker(query, [c["text"] for c in cands])
            cands = [c for _, c in sorted(zip(scores, cands), key=lambda t: -t[0])]
        cands = cands[:k]
        return self._pack(cands, budget_tokens, tok)

    # ── recency (C1) ────────────────────────────────────────────────────
    def recent(self, n: int, budget_tokens: int, tok=None) -> tuple[str, list[int]]:
        """Last-n exchanges (most-recent first) → packed memory block."""
        if not self.turns:
            return "", []
        recs = list(reversed(self.turns[-n:]))
        return self._pack(recs, budget_tokens, tok)

    # ── memory-block dispatch by condition ──────────────────────────────
    def block_for(self, condition: str, query: str, *, k: int, n: int,
                  budget_tokens: int, tok=None, rerank: bool = False) -> tuple[str, list[int]]:
        """Return (memory_text, used_turn_ids) for C0/C1/C2."""
        if condition == "C0":
            return "", []
        if condition == "C1":
            return self.recent(n, budget_tokens, tok)
        if condition == "C2":
            return self.retrieve(query, k, budget_tokens, tok, rerank=rerank)
        raise ValueError(f"unknown condition {condition!r}")


if __name__ == "__main__":
    # Offline smoke test with a deterministic stub embedder (bag-of-words hashing).
    def stub_embedder(texts):
        import numpy as _np
        vecs = []
        for t in texts:
            v = _np.zeros(64, dtype=_np.float32)
            for w in t.lower().split():
                v[hash(w) % 64] += 1.0
            nrm = _np.linalg.norm(v) or 1.0
            vecs.append(v / nrm)
        return _np.vstack(vecs)

    mem = SessionMemory(embedder=stub_embedder)
    mem.add(5,  "What is the melting point of tungsten?", "3422 °C, per the materials passage.")
    mem.add(20, "What is Planck's constant?", "6.626e-34 J·s, per the physics passage.")
    mem.add(50, "What is the speed of light?", "299,792,458 m/s in a vacuum.")
    for i in range(51, 120):
        mem.add(i, f"filler question {i}", f"filler answer {i}")

    block, used = mem.retrieve("Earlier you told me about tungsten's melting point — what was it?",
                               k=3, budget_tokens=120, tok=None)
    print("RETRIEVE (C2):\n" + block)
    print("used turns:", used, "\n")
    rblock, rused = mem.recent(3, budget_tokens=120, tok=None)
    print("RECENT (C1):\n" + rblock)
    print("used turns:", rused)
    assert 5 in used, "C2 must surface the tungsten anchor at turn 5 from 100+ turns back"
    assert 5 not in rused, "C1 sliding window should NOT reach turn 5"
    print("\nOK: C2 recalls the far anchor; C1 does not.")
