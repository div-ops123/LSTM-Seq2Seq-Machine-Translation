"""Text normalization for the English-Igbo parallel corpus.

Both languages get the same treatment: lowercase, strip invisible/encoding
artifacts, and isolate punctuation as its own token. Unicode letters,
digits, and combining marks (accents/tone marks) are kept attached to their
base character.

`normalize_ig` deliberately does NOT perform any Unicode decomposition or
accent-stripping. The PyTorch seq2seq tutorial's `unicodeToAscii` (NFD
decompose + drop category "Mn") is fine for French accents, but Igbo's
dotted vowels (i / ị, o / ọ, u / ụ) are distinct phonemes, not accented
variants of the same letter -- e.g. NFD-stripping turns "akwụkwọ" (book)
into "akwukwo". Applying that function here would silently corrupt the
Igbo vocabulary.
"""
import re
import unicodedata

# Zero-width / directional / BOM characters that show up as encoding noise
# in this corpus and carry no linguistic content.
_INVISIBLE = {"﻿", "‎", "‏", "​", "‌"}


def _strip_invisible(text: str) -> str:
    return "".join(ch for ch in text if ch not in _INVISIBLE)


def _is_word_char(ch: str) -> bool:
    category = unicodedata.category(ch)
    # L* = letter, N* = number, M* = combining mark (accents/tone marks)
    return category[0] in ("L", "N", "M")


def _isolate_punctuation(text: str) -> str:
    out = []
    for ch in text:
        if ch.isspace():
            out.append(" ")
        elif _is_word_char(ch):
            out.append(ch)
        else:
            out.append(f" {ch} ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def normalize_en(text: str) -> str:
    text = _strip_invisible(text).lower().strip()
    return _isolate_punctuation(text)


def normalize_ig(text: str) -> str:
    text = _strip_invisible(text).lower().strip()
    return _isolate_punctuation(text)
