"""The structured attachment list a channel hands to the provider with a prompt.

An image reaches the model as an ACP image block ONLY when the receiving
channel says the user attached one: the dashboard's upload list, a Slack or
Discord file, a Telegram photo. The prompt TEXT is never scanned for image
paths -- see :func:`kiro_crew.acp.prompt_blocks.build_prompt_blocks`, which
builds its image blocks from this list alone.

Why the list is structured rather than inferred from the text: a path in text is
a mention, not an upload. The session ledger's ``artifact <name>: <path>.png``
snapshot line rides in every nudge cycle; a nudge body, an injected envelope, an
agent's own ``![shot](...png)`` reply quoted back and a consolidation prompt all
name files the user never attached to THIS message. Inlining a mention re-sends
the file, and because the backend replays every stored image block, a screenshot
named in a per-cycle snapshot grows the request by its full encoded size on every
turn until the backend rejects the body. Only the channel that received a file
knows the user attached it, so only the channel may say so.

A LEAF module (stdlib only): it is imported by the ACP prompt builder, by
``kiro_crew.messaging`` (which the builder must not import back) and by the
dashboard runner, so it may reach nothing that reaches any of them.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

#: Longest name the ``[image: <name>]`` marker carries. The name is the
#: sender's own filename -- text the model reads inline -- so it is bounded
#: and flattened to one line rather than trusted to be either.
NAME_MAX_CHARS = 120


def bounded_name(name: str) -> str:
    """*name* flattened to one line and cut to :data:`NAME_MAX_CHARS`.

    The one rule for every name a record carries or shows: a channel's
    filename is sender-supplied, so it is bounded where it is STORED
    (ingestion builds the record with it) as well as where it is read.
    """
    return " ".join((name or "").split())[:NAME_MAX_CHARS]


#: A Windows drive-letter or UNC path, the shape the dashboard composer
#: rewrites to forward slashes inside a markdown destination (a destination
#: cannot carry raw backslashes: CommonMark eats ``\`` before punctuation).
_WINDOWS_SHAPE_RE = re.compile(r"^(?:[A-Za-z]:|\\\\[^\\/]+)[\\/]")

#: A destination the composer emits VERBATIM. Anything outside this set is
#: emitted percent-escaped and ``<...>``-wrapped (``mdImageDest`` in the
#: frontend's ``utils/fileTokens.ts``; this is its mirror). ``re.ASCII``
#: because JavaScript's ``\w`` is ``[A-Za-z0-9_]`` while Python's is
#: Unicode-aware: without it a Cyrillic or accented filename reads as plain
#: here and wrapped there, and the two spellings never meet.
_PLAIN_DEST_RE = re.compile(r"^[\w/.@:~-]*$", re.ASCII)


@dataclass(frozen=True)
class PromptAttachment:
    """One file the user attached to THIS message, as the receiving channel saw it.

    ``path`` is the absolute local path the bytes are read from; ``name`` is what
    the model is told it was given (``[image: <name>]``), defaulting to the
    path's basename. The record carries no type: the wire type is always
    derived from the file's leading bytes by the prompt builder, so a declared
    one would be a claim nothing reads.
    """

    path: str
    name: str = ""

    @property
    def display_name(self) -> str:
        """The name the model sees in the ``[image: <name>]`` marker.

        One line, at most :data:`NAME_MAX_CHARS` characters, whichever of the
        two sources supplies it: a channel's filename is sender-supplied text,
        and a basename is whatever the path carries.
        """
        name = bounded_name(self.name)
        if name:
            return name
        fallback = bounded_name(os.path.basename(self.path.rstrip("/\\")))
        return fallback or bounded_name(self.path) or self.path[:NAME_MAX_CHARS]


def markdown_image_dest(path: str) -> str:
    """The destination the dashboard composer writes for *path* in ``![image](...)``.

    Mirrors the frontend's ``mdImageDest`` exactly: a Windows drive-letter or
    UNC path is spelled with forward slashes; a destination made only of
    word characters, ``/``, ``.``, ``@``, ``:``, ``~`` and ``-`` is emitted as
    is; anything else has ``%`` escaped to ``%25`` and ``\\``, ``<``, ``>``
    backslash-escaped, and is wrapped in ``<...>``. Two consumers need the
    same answer: the prompt builder, to rewrite the line it inlined to the
    marker, and the dashboard's queued-edit prune, to see that a user removed
    a picture whose line the composer had escaped.
    """
    normalized = path.replace("\\", "/") if _WINDOWS_SHAPE_RE.match(path) else path
    if _PLAIN_DEST_RE.match(normalized) and "%" not in normalized:
        return normalized
    escaped = normalized.replace("%", "%25")
    escaped = re.sub(r"([\\<>])", r"\\\1", escaped)
    return f"<{escaped}>"


def path_spellings(path: str) -> tuple[str, ...]:
    """Every spelling a channel may have written *path* into the text in.

    The path itself; its forward-slash form when the path is Windows-shaped;
    and the exact markdown destination the dashboard composer emits. Order is
    longest-first so a substitution over the text replaces the wrapped
    destination whole rather than the bare path inside it. Deduplicated.
    """
    out: list[str] = [path]
    # Only a Windows-shaped path is ever spelled with forward slashes by a
    # channel (the same rule markdown_image_dest applies). A POSIX name that
    # merely holds a backslash keeps its one spelling: its slash-translated
    # sibling is an unrelated path, and marking it would rewrite the user's
    # own prose wherever that sibling appeared.
    if _WINDOWS_SHAPE_RE.match(path):
        slashed = path.replace("\\", "/")
        if slashed != path:
            out.append(slashed)
    dest = markdown_image_dest(path)
    if dest not in out:
        out.append(dest)
    return tuple(sorted(set(out), key=len, reverse=True))


#: What may stand right before or right after a spelling for it to count as
#: naming the path: the text's edges, whitespace (Slack writes the path as a
#: bare line), the parentheses of a markdown destination, or a quote. Any other
#: neighbour makes the match a piece of a LONGER token -- ``/tmp/a.png`` inside
#: ``/tmp/a.png.bak`` or ``x/tmp/a.png`` -- which names a different file.
_SPAN_DELIMITERS = frozenset("()\"'`")


def _delimited(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return (not before or before.isspace() or before in _SPAN_DELIMITERS) and (
        not after or after.isspace() or after in _SPAN_DELIMITERS
    )


def path_spans(path: str, text: str) -> list[tuple[int, int]]:
    """Every span of *text* that names *path*, as ``(start, end)`` pairs.

    A spelling (:func:`path_spellings`) counts only where it stands delimited
    (:data:`_SPAN_DELIMITERS`): a bare substring test would read
    ``/tmp/a.png`` in ``/tmp/a.png.bak`` and keep a removed picture alive
    through a different file, or let a rewrite corrupt that longer path.
    Each spelling is scanned once, left to right, resuming after an accepted
    span; the spans of all spellings are then merged in one sorted pass that
    keeps the earliest (and, at a tie, the longest) span and drops any that
    overlaps a kept one -- so a wrapped destination is one span rather than
    the bare path inside it. Linear in the text plus ``k log k`` in the number
    of spans: this runs on the gateway loop from the queued-edit prune over
    caller-typed text, so nothing here may grow with the square of the matches.
    Sorted by position, non-overlapping.
    """
    found: list[tuple[int, int]] = []
    for spelling in path_spellings(path):
        width = len(spelling)
        start = 0
        while True:
            at = text.find(spelling, start)
            if at < 0:
                break
            end = at + width
            if _delimited(text, at, end):
                found.append((at, end))
                start = end
            else:
                start = at + 1
    found.sort(key=lambda span: (span[0], span[0] - span[1]))
    spans: list[tuple[int, int]] = []
    claimed_end = -1
    for at, end in found:
        if at < claimed_end:
            continue
        spans.append((at, end))
        claimed_end = end
    return spans


def named_in_text(path: str, text: str) -> bool:
    """Whether *text* names *path* -- delimited, in any spelling a channel writes."""
    return bool(path_spans(path, text))


def image_attachments(paths: Iterable[str]) -> tuple[PromptAttachment, ...]:
    """Structured attachments for the image files a channel received.

    Order is kept and an empty or non-string entry is dropped; a path that
    appears twice is kept once, first position wins. The builder validates
    every entry against the filesystem and the bytes, so nothing here reads a
    file.
    """
    out: list[PromptAttachment] = []
    seen: set[str] = set()
    for raw in paths:
        if not isinstance(raw, str):
            continue
        path = raw.strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(PromptAttachment(path=path))
    return tuple(out)
