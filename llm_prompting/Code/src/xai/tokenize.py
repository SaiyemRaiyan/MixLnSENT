"""Whitespace tokenization that retains punctuation and source character spans."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Token:
    index: int
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    return [
        Token(index, match.group(), match.start(), match.end())
        for index, match in enumerate(re.finditer(r"\S+", text))
    ]


def render_tokens(tokens: list[Token], selected: set[int], replacement: str | None = None) -> str:
    pieces = [
        token.text if token.index in selected else replacement
        for token in tokens
    ]
    return " ".join(piece for piece in pieces if piece is not None).strip()


def delete_tokens(text: str, indices: set[int] | list[int]) -> str:
    tokens = tokenize(text)
    removed = set(indices)
    if removed - {token.index for token in tokens}:
        raise IndexError("token index outside sentence")
    return render_tokens(tokens, {token.index for token in tokens} - removed)


def mask_tokens(text: str, indices: set[int] | list[int]) -> str:
    tokens = tokenize(text)
    masked = set(indices)
    if masked - {token.index for token in tokens}:
        raise IndexError("token index outside sentence")
    return " ".join(
        "[MASK]" if token.index in masked else token.text
        for token in tokens
    )


def keep_tokens(text: str, indices: set[int] | list[int]) -> str:
    tokens = tokenize(text)
    keep = set(indices)
    if keep - {token.index for token in tokens}:
        raise IndexError("token index outside sentence")
    return render_tokens(tokens, keep)
