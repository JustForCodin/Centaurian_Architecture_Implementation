"""Symbolic extractive answer retriever for grounded QA (extract-then-style, plan option 2).

On the given-passage SQuAD eval the "retriever" is a *within-passage symbolic
extractor*: infer the expected answer TYPE from the question (rule-based), rank
context sentences by relevance (MiniLM reranker), extract candidate spans of that
type (spaCy NER + regex fallback), score by type-match + question-keyword overlap,
and return the best candidate — or abstain when the passage yields no confident
typed candidate. This supplies the exact answer the 80M generative reader cannot
localise; the model's job reduces to the two things it can do — the abstain
decision and voicing the answer in ADA's register (extract-then-style).

For open-domain daily-QA the same `extract()` interface would be backed by the
KG/QLever layer instead of within-passage NER; this module is the SQuAD-eval
instantiation of plan-§option-2.

Dependency-light: spaCy (en_core_web_sm) is used when available for NER; a regex
fallback (years/centuries/numbers/capitalised sequences) runs otherwise, so the
pipeline works offline. The reranker degrades to lexical overlap without
sentence-transformers (see rerank_util).
"""

from __future__ import annotations

import re

from rerank_util import SentenceReranker, split_sentences

# ── Expected-answer-type rules: question pattern → spaCy entity labels ────

_TYPE_RULES = [
    (re.compile(r"\b(who|whom|whose)\b", re.I), {"PERSON", "ORG", "NORP"}),
    (re.compile(r"\bwhere\b|\bwhat\s+(country|city|place|region|state|nation|location)\b", re.I),
     {"GPE", "LOC", "FAC"}),
    (re.compile(r"\bwhen\b|\bwhat\s+(year|century|date|time)\b|\bhow\s+long\b", re.I),
     {"DATE", "TIME"}),
    (re.compile(r"\bhow\s+(many|much|old)\b|\bwhat\s+(number|percentage|amount|fraction)\b", re.I),
     {"CARDINAL", "QUANTITY", "PERCENT", "MONEY", "ORDINAL"}),
]

# Regex candidates (fallback + augmentation over NER).
_RE_CENTURY = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)\s+centur(?:y|ies)\b", re.I)
_RE_YEAR = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})s?\b")
_RE_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")
_RE_CAP_SEQ = re.compile(r"\b([A-Z][a-zA-Z.]+(?:\s+(?:of\s+|the\s+)?[A-Z][a-zA-Z.]+)*)\b")

_STOP = set(
    "the a an of in on at to for and or but is are was were be been being by with "
    "from as that this these those it its their his her they he she we you i what "
    "which who whom whose when where why how did do does done can could would should "
    "will shall may might must have has had not no".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) > 2 and w not in _STOP}


class SymbolicRetriever:
    """Within-passage symbolic extractive answer selector."""

    def __init__(self, reranker: SentenceReranker | None = None,
                 use_spacy: bool = True, top_k: int = 2, device: str | None = None):
        self.reranker = reranker or SentenceReranker(device=device)
        self.top_k = top_k
        self.nlp = None
        self.backend = "regex"
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
                self.backend = "spacy"
            except Exception as e:                   # noqa: BLE001 — model not installed
                print(f"  [retriever] spaCy unavailable ({type(e).__name__}); "
                      f"using regex NER fallback", flush=True)

    # ── answer-type inference ──
    def _expected_types(self, question: str):
        for pat, types in _TYPE_RULES:
            if pat.search(question):
                return types
        return None                                  # open type

    # ── candidate span generation for one sentence ──
    def _candidates(self, sentence: str) -> list[tuple[str, str]]:
        cands: list[tuple[str, str]] = []
        if self.nlp is not None:
            for ent in self.nlp(sentence).ents:
                cands.append((ent.text.strip(), ent.label_))
        else:
            for m in _RE_CENTURY.finditer(sentence):
                cands.append((m.group(0), "DATE"))
            for m in _RE_YEAR.finditer(sentence):
                cands.append((m.group(0), "DATE"))
            for m in _RE_CAP_SEQ.finditer(sentence):
                cands.append((m.group(1).strip(), "ENT"))
            for m in _RE_NUMBER.finditer(sentence):
                cands.append((m.group(0), "CARDINAL"))
        return cands

    def _type_ok(self, label: str, types) -> bool:
        if types is None:
            return True
        if self.backend == "regex":
            # regex labels are coarse (DATE/CARDINAL/ENT); accept DATE/CARDINAL when
            # the question wants them, else fall through to scoring on ENT.
            if label in ("DATE", "CARDINAL"):
                return label in types or bool(types & {"DATE", "TIME", "CARDINAL",
                                                        "QUANTITY", "PERCENT", "MONEY", "ORDINAL"})
            return True
        return label in types

    # ── main entry ──
    def extract(self, question: str, context: str, tau: float = 0.0):
        """Return (answer_span | None, score). None = abstain (score < tau or no
        typed candidate). The candidate is a substring of the context."""
        sents = split_sentences(context)
        if not sents:
            return None, 0.0
        sent_scores = self.reranker._scores(question, sents)
        order = sorted(range(len(sents)), key=lambda i: sent_scores[i],
                       reverse=True)[: self.top_k]
        types = self._expected_types(question)
        qwords = _content_words(question)

        best, best_score = None, -1.0
        for si in order:
            sent = sents[si]
            sent_rel = max(float(sent_scores[si]), 0.0)
            sent_overlap = len(qwords & _content_words(sent)) / (len(qwords) or 1)
            for text, label in self._candidates(sent):
                if not text or text.lower() in question.lower():
                    continue                         # don't echo the question
                if not self._type_ok(label, types):
                    continue
                type_bonus = 0.3 if (types is not None and label in types) else 0.0
                score = 0.5 * sent_rel + 0.4 * sent_overlap + type_bonus
                if score > best_score:
                    best, best_score = text, score
        if best is None or best_score < tau:
            return None, max(best_score, 0.0)
        return best, best_score
