"""Pure-Python BPE tokenizer loaded from tokenizer.json.

No HuggingFace dependency — reads the standard tokenizers JSON format
directly and implements encode/decode with Metaspace pre-tokenisation
and byte-fallback decoding.
"""

import json
import os
import re


class Tokenizer:
    """SentencePiece-style BPE tokenizer.

    Loads vocabulary and merge rules from a ``tokenizer.json`` file
    (the format written by HuggingFace ``tokenizers``).  Supports:

    - Metaspace pre-tokeniser (``▁`` replaces leading spaces)
    - Byte-fallback for unknown characters
    - BOS / EOS special tokens
    - Chat-template rendering (Jinja-free, TinyLlama format)
    """

    def __init__(self, path: str) -> None:
        tok_path = os.path.join(path, "tokenizer.json")
        cfg_path = os.path.join(path, "tokenizer_config.json")

        with open(tok_path, encoding="utf-8") as f:
            data = json.load(f)
        with open(cfg_path, encoding="utf-8") as f:
            config = json.load(f)

        model = data["model"]

        # Vocabulary: token_str → id
        self._vocab: dict[str, int] = model["vocab"]
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}

        # Merge priority: (piece_a, piece_b) → rank (lower = higher priority)
        self._merges: dict[tuple[str, str], int] = {}
        for rank, entry in enumerate(model.get("merges", [])):
            if isinstance(entry, list):
                a, b = entry
            else:
                a, b = entry.split(" ", 1)
            self._merges[(a, b)] = rank

        # Special tokens
        self.bos_token: str = config.get("bos_token", "<s>")
        self.eos_token: str = config.get("eos_token", "</s>")
        self.unk_token: str = config.get("unk_token", "<unk>")
        self.bos_token_id: int = self._vocab.get(self.bos_token, 1)
        self.eos_token_id: int = self._vocab.get(self.eos_token, 2)
        self.unk_token_id: int = self._vocab.get(self.unk_token, 0)

        # Added / special token set for fast lookup
        self._special_tokens: set[str] = {
            at["content"] for at in data.get("added_tokens", [])
        }

        # Byte-fallback tokens: <0xHH> → byte value
        self._byte_tokens: dict[str, int] = {}
        for tok, tid in self._vocab.items():
            if tok.startswith("<0x") and tok.endswith(">") and len(tok) == 6:
                self._byte_tokens[tok] = int(tok[3:5], 16)

        # Pre-tokeniser config
        pt = data.get("pre_tokenizer", {})
        self._meta_replacement = pt.get("replacement", "▁")
        self._add_prefix_space = pt.get("add_prefix_space", True)

        # Chat template
        tpl_path = os.path.join(path, "chat_template.jinja")
        self._chat_template: str | None = None
        if os.path.exists(tpl_path):
            with open(tpl_path, encoding="utf-8") as f:
                self._chat_template = f.read()

    # ------------------------------------------------------------------
    # Pre-tokenisation (Metaspace)
    # ------------------------------------------------------------------
    def _pre_tokenize(self, text: str, add_prefix: bool = True) -> list[str]:
        """Split text into words using Metaspace convention.

        Replaces every space with ``▁`` and splits so that each word
        starts with ``▁`` (the Metaspace boundary).  Consecutive ``▁``
        are kept together as part of the same word.
        """
        if add_prefix and self._add_prefix_space and not text.startswith(" "):
            text = " " + text
        # Replace spaces with ▁
        text = text.replace(" ", self._meta_replacement)
        if not text:
            return []
        # Split before ▁ only when preceded by a non-▁ character
        r = re.escape(self._meta_replacement)
        words = re.split(f"(?<=[^{r}])(?={r})", text)
        return [w for w in words if w]

    # ------------------------------------------------------------------
    # BPE merge
    # ------------------------------------------------------------------
    def _bpe(self, word: list[str]) -> list[str]:
        """Apply BPE merges to a list of characters/pieces."""
        if len(word) <= 1:
            return word

        while True:
            # Find the pair with lowest merge rank
            best_pair = None
            best_rank = float("inf")
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self._merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # Merge all occurrences of best_pair
            merged = best_pair[0] + best_pair[1]
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == best_pair[0]
                    and word[i + 1] == best_pair[1]
                ):
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

            if len(word) == 1:
                break

        return word

    # ------------------------------------------------------------------
    # Byte-fallback encoding for unknown characters
    # ------------------------------------------------------------------
    def _encode_byte_fallback(self, char: str) -> list[int]:
        """Encode a single character as <0xHH> byte tokens."""
        ids = []
        for b in char.encode("utf-8"):
            tok = f"<0x{b:02X}>"
            ids.append(self._vocab.get(tok, self.unk_token_id))
        return ids

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        """Tokenize *text* into a list of token IDs.

        Adds BOS token by default (matching LLaMA convention).
        """
        # Handle special tokens in the text
        special_pattern = "|".join(
            re.escape(t) for t in sorted(self._special_tokens, key=len, reverse=True)
        )
        if special_pattern:
            parts = re.split(f"({special_pattern})", text)
        else:
            parts = [text]

        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_token_id)

        is_first_text = True
        for part in parts:
            if not part:
                continue
            if part in self._special_tokens:
                tid = self._vocab.get(part)
                if tid is not None:
                    ids.append(tid)
                continue

            # Pre-tokenize into words
            words = self._pre_tokenize(part, add_prefix=is_first_text)
            is_first_text = False
            for word in words:
                # Split word into characters for BPE
                chars = list(word)
                pieces = self._bpe(chars)

                for piece in pieces:
                    tid = self._vocab.get(piece)
                    if tid is not None:
                        ids.append(tid)
                    else:
                        # Byte-fallback for unknown pieces
                        for ch in piece:
                            ids.extend(self._encode_byte_fallback(ch))

        return ids

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    def decode(
        self, ids: list[int], skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to a string."""
        pieces: list[str] = []
        for tid in ids:
            tok = self._id_to_token.get(tid, "")
            if skip_special_tokens and tok in self._special_tokens:
                continue
            pieces.append(tok)

        text = "".join(pieces)

        # Byte-fallback decoding: convert <0xHH> sequences to bytes
        byte_pattern = re.compile(r"(<0x[0-9A-Fa-f]{2}>)+")
        def _decode_bytes(match: re.Match) -> str:
            hex_tokens = re.findall(r"<0x([0-9A-Fa-f]{2})>", match.group())
            return bytes(int(h, 16) for h in hex_tokens).decode("utf-8", errors="replace")

        text = byte_pattern.sub(_decode_bytes, text)

        # Undo metaspace: ▁ → space
        text = text.replace(self._meta_replacement, " ")

        # Strip leading space that was added by pre-tokenisation
        if text.startswith(" "):
            text = text[1:]

        return text

    # ------------------------------------------------------------------
    # Chat template (TinyLlama format, no Jinja dependency)
    # ------------------------------------------------------------------
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """Render a chat conversation into a prompt string."""
        parts: list[str] = []
        role_tags = {
            "user": "<|user|>",
            "system": "<|system|>",
            "assistant": "<|assistant|>",
        }
        for msg in messages:
            role = msg.get("role", "user")
            tag = role_tags.get(role, f"<|{role}|>")
            parts.append(f"{tag}\n{msg['content']}{self.eos_token}\n")

        if add_generation_prompt:
            parts.append("<|assistant|>\n")

        return "".join(parts)
