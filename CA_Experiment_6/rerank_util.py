"""Sentence reranker for grounded QA (the symbolic-localization component).

The from-scratch 80M model can copy a context span but can't *localize* which
span answers the question across a long passage (Exp 6 QA diagnosis). This module
supplies that localization the neuro-symbolic way: embed the question and each
context sentence with all-MiniLM-L6-v2 (already in the §7.3 provisioning list),
keep the top-k most similar sentences, and hand the model only that shortened
context. On the SQuAD reading-comprehension eval this stands in for the retriever
of plan-§option-2 (the passage is given, so "retrieval" = rerank-down-to-answer).

Lazy, dependency-light: sentence-transformers is imported only when a reranker is
constructed, so `evaluate.py` still loads without it. A deterministic lexical
(token-overlap) fallback is used when sentence-transformers is unavailable — good
enough to exercise the pipeline offline / in tests.
"""

from __future__ import annotations

import re

# Sentence splitter: keep it simple + deterministic (no nltk dependency).
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return parts or [text]


class SentenceReranker:
    """Rank context sentences by similarity to the question; return the top-k
    (in original order) joined back into a shortened context."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str | None = None):
        self.model_name = model_name
        self._model = None
        self._backend = "lexical"
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            self._backend = "minilm"
        except Exception as e:                       # noqa: BLE001 — offline / not installed
            print(f"  [rerank] sentence-transformers unavailable ({type(e).__name__}); "
                  f"using lexical fallback", flush=True)

    @property
    def backend(self) -> str:
        return self._backend

    def _scores(self, question: str, sentences: list[str]) -> list[float]:
        if self._backend == "minilm":
            import numpy as np
            emb = self._model.encode([question] + sentences, convert_to_numpy=True,
                                     normalize_embeddings=True)
            q, s = emb[0], emb[1:]
            return list((s @ q).astype(float))       # cosine (embeddings are normed)
        # Lexical fallback: normalized token-overlap (Jaccard-ish).
        def toks(t):
            return set(re.findall(r"[a-z0-9]+", t.lower()))
        qt = toks(question)
        out = []
        for s in sentences:
            st = toks(s)
            out.append(len(qt & st) / (len(qt | st) or 1))
        return out

    def shrink(self, question: str, context: str, top_k: int = 2) -> str:
        """Return the top_k question-relevant sentences of `context`, in original
        order, joined by spaces. Falls back to the full context if it has ≤ top_k
        sentences."""
        sents = split_sentences(context)
        if len(sents) <= top_k:
            return context
        scores = self._scores(question, sents)
        top_idx = sorted(sorted(range(len(sents)), key=lambda i: scores[i],
                                reverse=True)[:top_k])
        return " ".join(sents[i] for i in top_idx)
