"""Multi-session persistent Episodic Register for Experiment 8 (plan §5.1).

Experiment 7's `SessionMemory` is a *within-session* store: it is reset per script,
and recall is over exchanges from the same conversation. Experiment 8 needs the
episodic memory to be **dynamic and cross-session** — the concrete Phase-1 Episodic
Register (paper §17): at the end of each simulated session a compact summary of its
salient events is written to a store that *persists across sessions*, and a later
session recalls those events via the same extract-then-style path (retrieve →
span-extract → style).

`EpisodicRegister` subclasses `SessionMemory` and reuses its embedding, retrieval,
budget-packing, and `block_for` dispatch unchanged. It adds:

  * **session bookkeeping** — `begin_session(id)` marks the active session; written
    records are tagged with the session they came from so snippets can report the
    session gap (the multi-session analogue of Experiment 7's lag);
  * **an explicit write path** — `write_event(session_id, turn, user, value)` stores
    a *real* prior-session fact (the dynamic analogue of `salient_past_events`), and
    `write_session_summary(...)` stores a one-line session digest (plan §11.4:
    per-salient-event vs per-session granularity — both are supported; the eval
    chooses);
  * **`is_empty()`** — day 1 (D0): an empty register → the extractive path returns
    nothing → the agent disclaims (the honest answer), which the re-fit teaches it
    to do gracefully (plan §5.1 / H1).

The store is the *only* place episodic content lives. Nothing here is baked into
the weights — that is the whole point of Experiment 8.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_EXP7 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "CA_Experiment_7"))
if _EXP7 not in sys.path:
    sys.path.insert(0, _EXP7)

from memory_store import SessionMemory, MEMORY_HEADER  # noqa: E402,F401  (re-export header)


class EpisodicRegister(SessionMemory):
    """Persistent, cross-session episodic store. One per simulated user history."""

    def __init__(self, embedder=None, reranker=None):
        super().__init__(embedder=embedder, reranker=reranker)
        self.session_id = 0                    # active session (0 = none begun yet)

    # ── session bookkeeping ─────────────────────────────────────────────
    def begin_session(self, session_id: int) -> None:
        """Mark the session recall queries are being asked from (sets the gap origin)."""
        self.session_id = int(session_id)

    def is_empty(self) -> bool:
        """True on day 1 — no prior-session records to recall (drives the disclaim path)."""
        return len(self.turns) == 0

    # ── write path (populate the register from REAL prior sessions) ──────
    def _index(self, rec: dict) -> None:
        self._ensure_embedder()
        vec = self._embedder([rec["text"]])                # (1, d)
        self.turns.append(rec)
        self._emb = vec if self._emb is None else np.vstack([self._emb, vec])

    def write_event(self, session_id: int, turn: int, user: str, value: str,
                    topic: str = "") -> None:
        """Store one real prior-session event (the dynamic analogue of a
        salient_past_event). `value` is the fact ADA actually stated — for the
        extract-then-style path this is the span-extracted clean value, so the
        register holds facts, not the generative head's garble."""
        user, value = (user or "").strip(), (value or "").strip()
        self._index({"session_id": int(session_id), "turn_id": int(turn),
                     "user": user, "assistant": value, "topic": topic,
                     "kind": "event", "text": f"{user}\n{value}".strip()})

    def write_session_summary(self, session_id: int, summary: str,
                              topics: list[str] | None = None) -> None:
        """Store a one-line digest of a whole session (coarser granularity, plan §11.4)."""
        summary = (summary or "").strip()
        self._index({"session_id": int(session_id), "turn_id": -1,
                     "user": "", "assistant": summary, "topic": ", ".join(topics or []),
                     "kind": "summary", "text": summary})

    # ── snippet: report the SESSION the record came from (gap = self.session_id - sid) ──
    def _snippet(self, rec: dict) -> str:
        sid = rec.get("session_id", "?")
        if rec.get("kind") == "summary":
            body = " ".join(str(rec["assistant"]).split())
            return f"(session {sid}) {body}"
        u = " ".join(str(rec["user"]).split())
        a = " ".join(str(rec["assistant"]).split())
        return f"(session {sid}) Alex asked: {u} — you answered: {a}"

    # `retrieve`, `recent`, `_pack`, `block_for` are inherited from SessionMemory:
    # the retrieval maths and the token-budget cap (RQ4/H4: O(1) per turn regardless
    # of how large the register grows across sessions) are identical to Experiment 7.

    def stats(self) -> dict:
        sids = sorted({r.get("session_id") for r in self.turns})
        return {"records": len(self.turns),
                "sessions": [s for s in sids if s not in (None, 0)],
                "events": sum(1 for r in self.turns if r.get("kind") == "event"),
                "summaries": sum(1 for r in self.turns if r.get("kind") == "summary")}


if __name__ == "__main__":
    # Offline smoke test: deterministic bag-of-words stub embedder (no MiniLM).
    def stub(texts):
        out = []
        for t in texts:
            v = np.zeros(64, dtype=np.float32)
            for w in t.lower().split():
                v[hash(w) % 64] += 1.0
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.vstack(out)

    reg = EpisodicRegister(embedder=stub)

    # Day 1 — empty register → disclaim path.
    reg.begin_session(1)
    assert reg.is_empty(), "fresh register must be empty on day 1"
    blk, used = reg.block_for("C2", "Last time, what did you tell me about tungsten?",
                              k=3, n=3, budget_tokens=120, tok=None)
    assert blk == "" and used == [], "empty register must yield no memory block (→ disclaim)"
    print("day-1: empty register → no block (agent will disclaim). OK")

    # Sessions 1-3 write real events; session 4 recalls one across a 3-session gap.
    reg.write_event(1, 5, "What's the melting point of tungsten?", "3422 °C", topic="tungsten")
    reg.write_event(2, 8, "How tall is Everest?", "8849 metres", topic="everest")
    reg.write_event(3, 4, "What is Planck's constant?", "6.626e-34 J·s", topic="planck")
    reg.write_session_summary(3, "Covered Planck's constant and two materials questions.",
                              topics=["planck", "materials"])
    reg.begin_session(4)
    assert not reg.is_empty()
    blk, used = reg.retrieve("Earlier you told me tungsten's melting point — what was it?",
                             k=3, budget_tokens=160, tok=None)
    print("\nRETRIEVE (session 4, gap to session 1):\n" + blk)
    assert 5 in used, "must surface the tungsten event planted in session 1"
    assert "3422" in blk, "the recalled block must carry the real value"
    print("\nstats:", reg.stats())
    print("OK: cross-session recall from the persistent register.")
