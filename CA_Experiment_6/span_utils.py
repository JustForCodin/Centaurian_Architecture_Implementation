"""Extractive-QA span head — input formatting, char→token alignment, decode.

Phase 1 of the "reach 0.70 from scratch" plan. Instead of *generating* the
answer (which a small causal decoder can't localise), a discriminative span
head predicts answer start/end token positions over the context — the BERT
recipe, far more sample-efficient.

Input layout (fully bidirectional / encoder-style — callers pass
prefix_lens = full length so every token attends to every other):

    <|bos|> <|user|> {question} <|context|> {context tokens ...}

Position 0 (<|bos|>) doubles as the NULL / no-answer anchor: unanswerable
examples are labelled (0, 0), and at inference an answer is emitted only when
the best context span outscores the null span by a threshold (SQuAD2 style).

Special tokens are inserted as raw ids (not encoded strings) so the tokenizer's
character offsets stay clean for answer alignment.
"""

from __future__ import annotations

from tokenizer_util import ADATokenizer


def _prefix_ids(tok: ADATokenizer, question: str) -> list[int]:
    q_ids, _ = tok.encode_with_offsets(question)
    return [tok.bos_id, tok.user_id] + q_ids + [tok.context_id]


def _align(offsets: list[tuple[int, int]], c_start: int, c_end: int):
    """Map an answer char span [c_start, c_end) to (start_tok, end_tok) indices
    into `offsets` (context-local). Returns (None, None) if unalignable."""
    start_tok = end_tok = None
    for i, (a, b) in enumerate(offsets):
        if b <= a:                       # empty piece
            continue
        if b > c_start and start_tok is None:
            start_tok = i
        if a < c_end:
            end_tok = i
    if start_tok is None or end_tok is None or end_tok < start_tok:
        return None, None
    return start_tok, end_tok


def build_span_input(tok: ADATokenizer, question: str, context: str, max_len: int,
                     answer_char: tuple[int, int] | None = None) -> dict | None:
    """Build the span-head input for (question, context).

    Returns {ids, ctx_lo, offsets, label} where
      ids     : full token sequence
      ctx_lo  : token index at which the context begins in `ids`
      offsets : char offsets (in ORIGINAL context coords) for each context token
      label   : (start_pos, end_pos) token indices in `ids`, or None
    `answer_char` = (start,end) char span → produce a training label; None → eval.
    Context is answer-centred-windowed (train) or head-truncated (eval) to fit
    max_len. Returns None if an answer cannot be aligned/fit."""
    prefix = _prefix_ids(tok, question)
    ctx_ids, offs = tok.encode_with_offsets(context)
    if len(prefix) >= max_len:                       # pathological long question
        prefix = prefix[: max_len - 1]
    ctx_lo = len(prefix)
    budget = max_len - ctx_lo

    a_start = a_end = None
    if answer_char is not None:
        a_start, a_end = _align(offs, answer_char[0], answer_char[1])
        if a_start is None:
            return None                              # gold span didn't align

    if len(ctx_ids) > budget:
        if a_start is not None:
            w0 = max(0, min(a_start - budget // 4, len(ctx_ids) - budget))
            if a_end - w0 >= budget:                 # answer would be cut → drop
                return None
        else:
            w0 = 0
        ctx_ids = ctx_ids[w0:w0 + budget]
        offs = offs[w0:w0 + budget]
        if a_start is not None:
            a_start -= w0
            a_end -= w0

    ids = prefix + ctx_ids
    label = (ctx_lo + a_start, ctx_lo + a_end) if a_start is not None else None
    return {"ids": ids, "ctx_lo": ctx_lo, "offsets": offs, "label": label}


def decode_span(context: str, offsets: list[tuple[int, int]], i: int, j: int) -> str:
    """Context substring for context-local token indices [i, j] (inclusive)."""
    return context[offsets[i][0]: offsets[j][1]].strip()


def best_span(start_logits, end_logits, ctx_lo: int, n_ctx: int,
              max_ans_tok: int = 30, topk: int = 20):
    """Search the best (i, j) context span (context-local indices) maximising
    start+end logit, with j>=i and length ≤ max_ans_tok. Returns
    (best_score, i, j, null_score) where null_score is the position-0 span."""
    import torch
    s = start_logits[ctx_lo: ctx_lo + n_ctx]
    e = end_logits[ctx_lo: ctx_lo + n_ctx]
    null = float(start_logits[0] + end_logits[0])
    if n_ctx == 0:
        return -1e9, 0, 0, null
    ks = min(topk, n_ctx)
    top_s = torch.topk(s, ks).indices.tolist()
    top_e = torch.topk(e, ks).indices.tolist()
    best = (-1e30, 0, 0)
    for i in top_s:
        for j in top_e:
            if j < i or (j - i + 1) > max_ans_tok:
                continue
            sc = float(s[i] + e[j])
            if sc > best[0]:
                best = (sc, i, j)
    return best[0], best[1], best[2], null
