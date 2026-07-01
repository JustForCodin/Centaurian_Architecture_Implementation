"""Load/wrap the from-scratch ADA BPE tokenizer (plan §5.1, 16k vocab).

Thin wrapper over the HuggingFace `tokenizers` library so training/eval code
never touches special-token bookkeeping directly.
"""

from __future__ import annotations

from pathlib import Path

from ca_assets import (
    SPECIAL_TOKENS, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    SYSTEM_TOKEN, PERSONA_TOKEN, CONTEXT_TOKEN, USER_TOKEN, ASSISTANT_TOKEN, EOT_TOKEN,
)

DEFAULT_TOKENIZER_PATH = Path(__file__).parent / "tokenizer" / "ada_bpe.json"


class ADATokenizer:
    """Encode/decode with named special-token ids exposed as attributes."""

    def __init__(self, tk):
        self.tk = tk
        self.vocab_size = tk.get_vocab_size()
        # Named special-token ids
        self.bos_id = tk.token_to_id(BOS_TOKEN)
        self.eos_id = tk.token_to_id(EOS_TOKEN)
        self.pad_id = tk.token_to_id(PAD_TOKEN)
        self.system_id = tk.token_to_id(SYSTEM_TOKEN)
        self.persona_id = tk.token_to_id(PERSONA_TOKEN)
        self.context_id = tk.token_to_id(CONTEXT_TOKEN)
        self.user_id = tk.token_to_id(USER_TOKEN)
        self.assistant_id = tk.token_to_id(ASSISTANT_TOKEN)
        self.eot_id = tk.token_to_id(EOT_TOKEN)
        missing = [t for t in SPECIAL_TOKENS if tk.token_to_id(t) is None]
        if missing:
            raise ValueError(f"tokenizer is missing special tokens: {missing}")

    @classmethod
    def load(cls, path=DEFAULT_TOKENIZER_PATH) -> "ADATokenizer":
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(str(path)))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tk.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self.tk.decode(ids, skip_special_tokens=skip_special)
