"""Per-session inline-image ledger: dedup and an aggregate budget over prompt blocks.

The per-image caps in :mod:`kiro_crew.imaging` bound ONE image. Nothing there
bounds what a session inlines in total, and the total is what the backend
measures: kiro-cli replays the whole conversation to the model on every turn, so
every inlined image is re-sent on every later turn and the request body grows by
that image's full base64 size per turn until the backend refuses the body. The
context window is not what gives out -- usage read 3.6% when a request was
refused -- the wire bytes are. This module is the layer over the FINISHED block
list that bounds that growth, whatever produced the blocks:

* **Dedup.** Every inlined image is keyed by the SHA-256 of its base64 payload
  and the keys are kept per session. A payload already inlined in this session
  is not inlined again -- the conversation the model sees already carries it --
  and its text marker becomes ``[image: <name>, sent earlier]``. This holds for
  an automation that names the same file every cycle and for a person who
  pastes the same screenshot twice.
* **Budget.** A per-prompt cap on image count and on total base64 bytes, plus a
  per-session running total of inlined base64 bytes. A block that would cross
  any of them is degraded to a text marker that keeps the file path, so a
  tool-capable agent can still open the file.

The ledger describes the conversation the model is REPLAYED, so it follows that
conversation's own life: it lives on the session's durable record -- the
``SessionMap`` entry, reached through the store the session manager registers
here -- so it survives a gateway restart; it names the native conversation it
describes (the ACP session id, ``sid``, its images were inlined into), and a
prompt on a different sid reads it as empty, because a new native conversation
(``/new``, a discarded conversation, a provider switch, a fresh session whose sid
promotion is deferred behind a history replay) carries none of the old images; a
confirmed native clear empties it; and a kiro-cli compaction REFUNDS it, because
a compaction summarizes the older history into text and the images in that
history leave the replay -- only the newest prompts stay verbatim, so only their
bytes stay charged (:func:`compact_ledger`). A session with no durable record (a
stateless cron or subagent session, the direct client) keeps an in-memory ledger
for the life of its handle.

A LEAF module, like :mod:`kiro_crew.imaging`: it imports nothing from
``kiro_crew.acp`` (which imports it) and nothing from ``kiro_crew.session_map``
(which implements :class:`ImageLedgerStore`), so both sides can import it.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeGuard

logger = logging.getLogger(__name__)

#: Smallest request-body ceiling measured so far on a backend route, as bytes of
#: base64 the request carried. One route accepted a request replaying 9 copies
#: of a 3,380,356-byte base64 image (30.4 MB) and refused the 10th (33.8 MB)
#: with ``Improperly formed request``; 32 MiB (33,554,432) lies inside that
#: bracket. A second route accepted 135 MB and kept going, so this is the
#: SMALLEST ceiling seen, and every budget below is a share of it -- an image
#: that fits under it fits on every route measured.
_SMALLEST_MEASURED_REQUEST_BODY_CEILING = 32 * 1024 * 1024

#: Per-session running total of inlined base64 bytes: three quarters of the
#: smallest measured ceiling. The remaining quarter is what the same replayed
#: request carries besides images -- the conversation text, tool results and
#: JSON framing -- so the images alone can never bring the body to the ceiling.
MAX_SESSION_IMAGE_B64_BYTES = _SMALLEST_MEASURED_REQUEST_BODY_CEILING * 3 // 4

#: Per-prompt total of inlined base64 bytes: half the session allowance, so one
#: prompt cannot spend the whole session budget and leave nothing for the
#: screenshot the next question needs. This cap holds even where no ledger is
#: available (nothing persisted yet, the first prompt of a session).
MAX_PROMPT_IMAGE_B64_BYTES = MAX_SESSION_IMAGE_B64_BYTES // 2

#: Per-prompt image count. An inlined image costs about 1,600 tokens of the
#: window whatever its pixel count (measured on a 1M-token window: every image
#: moved the context meter by the same 0.16%, a 2000x1200 frame and a 921x972
#: one alike), and the replay re-spends that on every later turn, so 20 images
#: are ~32k tokens per turn for the rest of the conversation. It is also the
#: count above which the backend applies the many-image dimension rule that
#: ``kiro_crew.imaging.MAX_IMAGE_EDGE_PX`` documents.
MAX_PROMPT_IMAGE_BLOCKS = 20

#: Bound on the digests a ledger retains, oldest evicted first. The byte total
#: is the hard bound on growth; the digest list only decides which repeats are
#: recognised, and an evicted digest costs one re-inline of that image, never a
#: wrong dedup. Each digest is a fixed 64-hex-character SHA-256, so the list is
#: at most 16 KiB on the session record.
MAX_LEDGER_HASHES = 256

#: What a kiro-cli compaction leaves in the replay. It summarizes the history
#: into text except for the newest user/assistant PAIRS -- at least this many,
#: and as many more as it takes, walked newest-first, to reach two percent of the
#: context window measured in raw bytes with an image at its full byte weight --
#: and then re-sends the prompt that was in flight. A prompt within this many
#: positions of the newest one is kept whatever its size; an older one is kept
#: only while the walk has not yet reached its target.
COMPACTION_KEPT_PAIRS = 2

#: The walk's target in raw bytes, sized for the largest context window served
#: (a million tokens at four bytes per token). A smaller window stops the walk
#: sooner and keeps LESS, so sizing the record for the largest one can only
#: over-count what stays charged. The walk here sees each image prompt's own
#: raw image bytes and the prompt text written after it -- never the assistant's
#: replies or tool results -- which likewise reaches further back than
#: kiro-cli's own walk.
COMPACTION_WALK_TARGET_BYTES = 1_000_000 * 2 // 100 * 4

#: Bound on the per-prompt records a ledger keeps for the walk. A record is
#: dropped once no compaction could keep its prompt, so the list only grows past
#: a handful when consecutive prompts inline images too small to reach the
#: target between them; past this many the oldest is treated as summarized.
MAX_RECENT_PROMPTS = 32

#: The only shape a retained digest may have: the lowercase hex SHA-256 that
#: :func:`image_digest` produces. Anything else in a record -- a wrong length, an
#: uppercase or non-hex character -- is dropped at retention, so a malformed
#: entry neither occupies a slot nor evicts a real digest within the read window
#: (:func:`normalize_ledger`), which every record the writer stores fits whole.
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")

#: Host-side annotation the prompt builder attaches to each image block::
#:
#:     {"path": <the path as written>, "spans": [[s, e], ...]}
#:
#: ``path`` is what the degraded marker names so a tool-capable agent can still
#: open the file. ``spans`` are the ``[start, end)`` offsets, in the prompt's
#: first text block, of every marker the builder wrote for THIS block -- the
#: substitutions it performed, not a search for their text -- so the layer
#: rewrites exactly those characters when it drops the block and never a
#: neighbour's marker, nor a bracketed string the user happened to type.
#: Stripped -- with every other ``_``-prefixed key -- before the list is
#: returned, so it never reaches the wire. The annotation is the contract: the
#: one producer of image blocks (``build_prompt_blocks``) always writes it. A
#: block without it is still deduped and budgeted, and when dropped leaves the
#: text as it is.
IMAGE_BLOCK_SOURCE_KEY = "_source"

_REASON_SENT_EARLIER = "sent_earlier"
_REASON_OVER_BUDGET = "over_budget"


class ImageLedgerStore(Protocol):
    """The durable home of a session's ledger -- ``SessionMap`` implements it."""

    def get_image_ledger(self, key: str) -> dict[str, Any] | None:
        """The ledger stored for *key*, ``None`` when *key* has no durable record."""

    def set_image_ledger(self, key: str, ledger: dict[str, Any]) -> bool:
        """Store *ledger* on *key*'s EXISTING record; ``False`` when there is none."""


# The live store, registered by the session manager that owns the live
# SessionMap. MODULE-level for the same reason the map's own listeners are: the
# prompt path runs inside the ACP layer, which holds a session KEY and nothing
# that reaches the manager, and a throwaway ``SessionMap()`` is read-only by
# that class's contract -- only the live instance may write.
_STORE: ImageLedgerStore | None = None


def set_image_ledger_store(store: ImageLedgerStore | None) -> None:
    """Register (or clear, with ``None``) the durable ledger store."""
    global _STORE
    _STORE = store


def empty_ledger(sid: str = "") -> dict[str, Any]:
    """A ledger for native conversation *sid* that has inlined nothing."""
    return {"sid": sid, "hashes": [], "b64_bytes": 0, "recent": []}


def _normalize_recent(raw: object) -> list[dict[str, int]]:
    """*raw* as well-formed per-prompt records -- the newest :data:`MAX_RECENT_PROMPTS`
    entries carrying the prompt's inlined base64 bytes (``b``), the prompt text
    written after it (``t``, raw bytes) and its position (``after``), each a
    non-negative integer -- anything else dropped."""
    if not isinstance(raw, list):
        return []
    recent: list[dict[str, int]] = []
    for entry in raw[-MAX_RECENT_PROMPTS:]:
        if not isinstance(entry, dict):
            continue
        b, t, after = entry.get("b"), entry.get("t"), entry.get("after")
        if not (_is_count(b) and _is_count(t) and _is_count(after)):
            continue
        recent.append(
            {
                "b": b,
                "t": min(t, COMPACTION_WALK_TARGET_BYTES),
                "after": min(after, COMPACTION_KEPT_PAIRS + 1),
            }
        )
    return recent


def _is_count(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def normalize_ledger(raw: object) -> dict[str, Any]:
    """*raw* as a well-formed ledger, dropping anything a ledger cannot hold.

    Applied at every point of retention -- when a ledger is read back from the
    session record and again before one is stored -- so a hand-edited or
    corrupt record can neither grow the list past :data:`MAX_LEDGER_HASHES`
    nor retain a string that is not a digest. Anything malformed reads as an
    empty ledger, which only ever costs a re-inline. The ``sid`` is kept as
    given (or ``""``); the session record bounds its length at retention with
    the one ACP-session-id bound the map already applies to every sid it holds.

    The read is bounded BEFORE it validates: only the newest
    :data:`MAX_LEDGER_HASHES` raw entries are examined, so a record of any
    length costs the constant, not the record -- this runs on the event loop
    from ``load_image_ledger``, and the writer never stores more than the bound,
    so a longer list is foreign (hand-edited, corrupt, another version) and its
    overflow is counted and logged once rather than walked. Every record the
    writer can produce fits the window whole, where a malformed entry is still
    dropped ahead of the cap and so occupies no slot and evicts no real digest.
    """
    if not isinstance(raw, dict):
        return empty_ledger()
    hashes_raw = raw.get("hashes")
    hashes: list[str] = []
    if isinstance(hashes_raw, list):
        overflow = len(hashes_raw) - MAX_LEDGER_HASHES
        if overflow > 0:
            # Counts only -- never a digest, which could be joined to an image.
            logger.warning(
                "image ledger: record holds %d entr%s past the %d-digest bound; "
                "only the newest %d were read",
                overflow,
                "y" if overflow == 1 else "ies",
                MAX_LEDGER_HASHES,
                MAX_LEDGER_HASHES,
            )
        window = hashes_raw[-MAX_LEDGER_HASHES:] if overflow > 0 else hashes_raw
        hashes = [h for h in window if isinstance(h, str) and _DIGEST_RE.fullmatch(h)]
    b64_raw = raw.get("b64_bytes")
    b64_bytes = b64_raw if isinstance(b64_raw, int) and not isinstance(b64_raw, bool) else 0
    sid_raw = raw.get("sid")
    return {
        "sid": sid_raw if isinstance(sid_raw, str) else "",
        "hashes": hashes,
        "b64_bytes": max(0, b64_bytes),
        "recent": _normalize_recent(raw.get("recent")),
    }


def _kept_by_compaction(recent: list[dict[str, int]]) -> list[dict[str, int]]:
    """The per-prompt records a kiro-cli compaction could still leave in the replay.

    Newest first: the prompt in flight (position 0) is re-sent whole; a prompt
    within :data:`COMPACTION_KEPT_PAIRS` positions is kept whatever its size;
    an older one is kept only while what was written after it -- the raw image
    bytes of the newer image prompts plus the prompt text recorded on it --
    has not reached :data:`COMPACTION_WALK_TARGET_BYTES`. The prompt that
    reaches the target is itself kept, as kiro-cli keeps the pair that crosses
    it, and everything older is summarized. Order is preserved (oldest first).
    """
    kept: list[dict[str, int]] = []
    newer_image_bytes = 0
    for entry in reversed(recent):
        after = entry["after"]
        if after == 0:
            kept.append(entry)
            continue
        walked = newer_image_bytes + entry["t"]
        if after > COMPACTION_KEPT_PAIRS and walked >= COMPACTION_WALK_TARGET_BYTES:
            break
        kept.append(entry)
        newer_image_bytes += entry["b"] * 3 // 4
    kept.reverse()
    return kept


def _advance_recent(
    recent: list[dict[str, int]], inlined_b64_bytes: int, text_bytes: int
) -> list[dict[str, int]]:
    """*recent* after one more prompt is written: *text_bytes* of text, and
    *inlined_b64_bytes* of images.

    Every earlier record moves one position further from the newest prompt (its
    position saturates one past :data:`COMPACTION_KEPT_PAIRS`, beyond which only
    the walk can keep it) and gains the prompt's text; the prompt gains a record
    when it inlined anything; and records no compaction could keep are dropped
    now rather than carried: later prompts only push a record further out and
    add bytes ahead of it, so a record dropped today would never be kept.
    """
    aged = [
        {
            "b": e["b"],
            "t": min(e["t"] + text_bytes, COMPACTION_WALK_TARGET_BYTES),
            "after": min(e["after"] + 1, COMPACTION_KEPT_PAIRS + 1),
        }
        for e in recent
    ]
    if inlined_b64_bytes > 0:
        aged.append({"b": inlined_b64_bytes, "t": 0, "after": 0})
    return _kept_by_compaction(aged)[-MAX_RECENT_PROMPTS:]


def compact_ledger(ledger: dict[str, Any] | None) -> dict[str, Any]:
    """*ledger* after a kiro-cli compaction of the conversation it describes.

    The summarized history carries its images only as text, so every digest is
    forgotten -- a picture attached again is inlined again, at worst once more
    than needed for one the kept tail still carries -- and the byte total drops
    to what the kept prompts inlined, the bytes the replay can still carry. The
    per-prompt records themselves stay: they are what the NEXT compaction reads.
    """
    state = normalize_ledger(ledger)
    kept = _kept_by_compaction(state["recent"])
    return {
        "sid": state["sid"],
        "hashes": [],
        "b64_bytes": sum(e["b"] for e in kept),
        "recent": kept,
    }


def load_image_ledger(session_key: str, session_id: str) -> dict[str, Any] | None:
    """The durable ledger for *session_key*'s conversation *session_id*, or ``None``.

    ``None`` means the session has no durable record and the caller must fall
    back to an in-memory ledger. A record that describes a DIFFERENT native
    conversation -- the entry still carries the previous sid's ledger because a
    fresh session's sid promotion is deferred behind a history replay, or the
    record predates the sid -- reads as an empty ledger for *session_id*: the
    new conversation carries none of the old images, and treating it otherwise
    would drop a picture attached to a conversation that never received it.
    """
    store = _STORE
    if not session_key or store is None:
        return None
    raw = store.get_image_ledger(session_key)
    if raw is None:
        return None
    ledger = normalize_ledger(raw)
    return ledger if ledger["sid"] == session_id else empty_ledger(session_id)


def store_image_ledger(session_key: str, ledger: dict[str, Any]) -> bool:
    """Persist *ledger* for *session_key*; ``False`` when nothing durable took it.

    A failed write is reported, never raised: the prompt it belongs to is
    already built, and losing one ledger update costs at most one re-inline on
    the next turn.
    """
    store = _STORE
    if not session_key or store is None:
        return False
    try:
        return bool(store.set_image_ledger(session_key, normalize_ledger(ledger)))
    except OSError:
        logger.warning("image ledger: could not persist for session %s", session_key, exc_info=True)
        return False


def _is_image_block(block: object) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") == "image"
        and isinstance(block.get("data"), str)
    )


def _is_text_block(block: object) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def _text_bytes(blocks: list[dict[str, Any]]) -> int:
    """UTF-8 bytes of the text blocks: what the runtime's history holds for the prompt besides images."""
    return sum(len(b["text"].encode("utf-8")) for b in blocks if _is_text_block(b))


def has_image_blocks(blocks: list[dict[str, Any]]) -> bool:
    """Whether *blocks* carries at least one image block the layer would judge."""
    return any(_is_image_block(b) for b in blocks)


def image_digest(data: str) -> str:
    """SHA-256 of an image block's base64 payload, the ledger's key.

    Hashed as the base64 text rather than the decoded bytes: base64 is a fixed
    bijection, so two blocks share a digest exactly when they would put the same
    bytes on the wire, and no decode buffer is allocated per image per turn.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ImageBudgetResult:
    """What :func:`apply_image_budget` decided, plus the ledger to carry forward."""

    blocks: list[dict[str, Any]]
    ledger: dict[str, Any]
    inlined: int
    sent_earlier: int
    over_budget: int
    evicted: int


def apply_image_budget(
    blocks: list[dict[str, Any]], ledger: dict[str, Any] | None
) -> ImageBudgetResult:
    """Dedup and budget the image blocks in *blocks* against *ledger*.

    A pure function over the finished block list and the session's ledger: it
    reads only ``type``, ``data`` and the optional
    :data:`IMAGE_BLOCK_SOURCE_KEY` annotation the builder writes. Blocks are judged in
    order. An image whose digest the ledger already holds -- or that an earlier
    block of this same prompt already inlined -- is dropped and its marker
    rewritten to ``[image: <name>, sent earlier; file: <path>]``. Otherwise the
    block is kept only while the prompt stays within :data:`MAX_PROMPT_IMAGE_BLOCKS`
    and :data:`MAX_PROMPT_IMAGE_B64_BYTES` and the session total stays within
    :data:`MAX_SESSION_IMAGE_B64_BYTES`; a block that would cross any of them is dropped and
    its marker rewritten to ``[image: <name>, not inlined: over the image
    budget; file: <path>]``. Only KEPT blocks enter the ledger: a block the
    budget refused was never sent, so a later prompt may still inline it. The
    returned ledger keeps the input ledger's ``sid``.

    Every prompt, image-bearing or not, is one more position between the
    earlier image prompts and the newest one, which is what a compaction later
    reads (:func:`compact_ledger`); a text-only prompt therefore comes back
    with its blocks untouched -- the same list object -- and a ledger whose
    per-prompt records have moved one position.

    Returns fresh objects and mutates neither input. Kept image blocks come back
    without any ``_``-prefixed key, and non-image blocks pass through unchanged
    except for the text rewritten for a degraded image.
    """
    state = normalize_ledger(ledger)
    if not has_image_blocks(blocks):
        return ImageBudgetResult(
            blocks=blocks,
            ledger={**state, "recent": _advance_recent(state["recent"], 0, _text_bytes(blocks))},
            inlined=0,
            sent_earlier=0,
            over_budget=0,
            evicted=0,
        )
    known: set[str] = set(state["hashes"])
    hashes: list[str] = list(state["hashes"])
    session_bytes: int = state["b64_bytes"]

    out: list[dict[str, Any]] = []
    degraded: list[tuple[dict[str, Any], str]] = []
    inlined = 0
    prompt_bytes = 0
    sent_earlier = 0
    over_budget = 0
    for block in blocks:
        if not _is_image_block(block):
            out.append(block)
            continue
        data: str = block["data"]
        digest = image_digest(data)
        source = block.get(IMAGE_BLOCK_SOURCE_KEY)
        annotation = source if isinstance(source, dict) else {}
        if digest in known:
            sent_earlier += 1
            degraded.append((annotation, _REASON_SENT_EARLIER))
            continue
        size = len(data)
        if (
            inlined + 1 > MAX_PROMPT_IMAGE_BLOCKS
            or prompt_bytes + size > MAX_PROMPT_IMAGE_B64_BYTES
            or session_bytes + size > MAX_SESSION_IMAGE_B64_BYTES
        ):
            over_budget += 1
            degraded.append((annotation, _REASON_OVER_BUDGET))
            continue
        known.add(digest)
        hashes.append(digest)
        inlined += 1
        prompt_bytes += size
        session_bytes += size
        out.append({k: v for k, v in block.items() if not str(k).startswith("_")})

    if degraded:
        out = _rewrite_markers(out, degraded)
    evicted = max(0, len(hashes) - MAX_LEDGER_HASHES)
    return ImageBudgetResult(
        blocks=out,
        ledger={
            "sid": state["sid"],
            "hashes": hashes[-MAX_LEDGER_HASHES:],
            "b64_bytes": session_bytes,
            "recent": _advance_recent(state["recent"], prompt_bytes, _text_bytes(out)),
        },
        inlined=inlined,
        sent_earlier=sent_earlier,
        over_budget=over_budget,
        evicted=evicted,
    )


def _degraded_text(marker: str, annotation: dict[str, Any], reason: str) -> str:
    """The text that replaces *marker* (the producer's own ``[image: ...]``) for a dropped block.

    Keeps the marker's bracketed text and appends the reason -- and the file
    path, when the annotation carries one. The path rides along even for a
    repeat, because the ledger can outlive the picture: a turn the runtime never
    appended to its history (it died before answering) stays charged, and a
    compaction on a backend whose kept tail is unmeasured refunds nothing, so
    the path keeps the file reachable to a tool-capable agent. For a block over
    the budget the path is the builder's own fallback for an image it cannot
    inline.
    """
    path = annotation.get("path")
    has_path = isinstance(path, str) and bool(path)
    if reason == _REASON_SENT_EARLIER:
        suffix = f", sent earlier; file: {path}]" if has_path else ", sent earlier]"
    else:
        suffix = (
            f", not inlined: over the image budget; file: {path}]"
            if has_path
            else ", not inlined: over the image budget]"
        )
    return marker[:-1] + suffix


def _rewrite_markers(
    blocks: list[dict[str, Any]], degraded: list[tuple[dict[str, Any], str]]
) -> list[dict[str, Any]]:
    """Rewrite each degraded image's own markers, at the offsets its producer recorded.

    Only the characters the builder substituted are touched -- the spans the
    annotation carries, applied right to left so earlier offsets stay valid --
    never a search for the marker's text, which would also rewrite a bracketed
    string the user typed or a neighbour's identical marker. The spans are the
    builder's own substitution record for the prompt's first text block, computed
    in the same pass that wrote the markers, so they are used as given; the
    annotation is the contract, and the one producer of image blocks always
    writes it. A degraded block that carries none leaves the text as it is. Text
    blocks are copied before they are edited, so the caller's list is never
    mutated.
    """
    out = [dict(b) if _is_text_block(b) else b for b in blocks]
    first = next((i for i, b in enumerate(out) if _is_text_block(b)), None)
    if first is None:
        return out
    text = out[first]["text"]
    edits: list[tuple[int, int, str]] = []
    for annotation, reason in degraded:
        for start, end in annotation.get("spans") or ():
            edits.append((start, end, _degraded_text(text[start:end], annotation, reason)))
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    out[first]["text"] = text
    return out


def withheld_notice(over_budget: int) -> str:
    """The sentence a surface shows its user for *over_budget* images kept off the wire.

    The prompt path yields it as an event once the prompt is written; the
    dashboard appends it as a notice row. It names counts and caps only, never
    a file, a path or the picture itself, so it can be shown on any surface.
    """
    count = "One image was" if over_budget == 1 else f"{over_budget} images were"
    return (
        f"\u26a0\ufe0f {count} not sent to the model: over the image budget "
        f"({MAX_PROMPT_IMAGE_BLOCKS} images or {MAX_PROMPT_IMAGE_B64_BYTES // (1024 * 1024)} MiB "
        f"per message, {MAX_SESSION_IMAGE_B64_BYTES // (1024 * 1024)} MiB per conversation, "
        "refunded when the conversation is compacted). The file path stayed in the "
        "message for an agent with file tools; /new starts a conversation with a fresh budget."
    )


class SessionImageBudget:
    """The layer bound to one runtime session: applies the budget and keeps its ledger.

    Owned by the object that sends ``session/prompt`` for a session (the ACP
    session handle, the direct client). *session_key* and *session_id* are read
    on every call: a pooled handle is rebound to its owning session on claim,
    and the direct client's native sid changes on a reset. The durable ledger
    is used whenever the session has a durable record and describes this sid;
    otherwise the ledger lives here, for as long as the owner does.

    The ledger moves in two steps. :meth:`apply` judges the blocks and STAGES the
    recomputed ledger; :meth:`commit` records it once the prompt has actually
    been written to the runtime, and :meth:`discard` drops it when the write
    never happened. Charging at build time instead would record an image the
    conversation never received: a runtime that dies between the build and the
    write makes the caller re-queue the same message, and the retry would then
    read the undelivered image as "sent earlier" and drop it.
    """

    def __init__(self, session_key: Callable[[], str], session_id: Callable[[], str]) -> None:
        self._session_key = session_key
        self._session_id = session_id
        self._local: dict[str, Any] = empty_ledger()
        # ``(session key, ledger is durable, ledger)`` staged by ``apply`` for the
        # prompt being built; ``None`` when nothing is owed.
        self._pending: tuple[str, bool, dict[str, Any]] | None = None
        # Image blocks the staged prompt kept off the wire as over the budget;
        # what ``commit`` hands back so the owner can tell its user.
        self._pending_withheld = 0

    def _current(self) -> tuple[str, bool, dict[str, Any]]:
        """``(session key, ledger is durable, ledger)`` for this owner's conversation now."""
        key = self._session_key() or ""
        sid = self._session_id() or ""
        durable = load_image_ledger(key, sid)
        if durable is not None:
            return key, True, durable
        if self._local["sid"] != sid:
            # The in-memory ledger describes the conversation it was built in;
            # a new native conversation on this owner starts from nothing.
            self._local = empty_ledger(sid)
        return key, False, self._local

    def _record(self, key: str, durable: bool, ledger: dict[str, Any]) -> None:
        if durable:
            # SessionMap's on-loop mutation marks the map dirty and defers the
            # file write to its worker thread (its own threading contract);
            # nothing here waits on disk.
            store_image_ledger(key, ledger)
        else:
            self._local = ledger

    async def apply(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """*blocks* with this session's dedup and budget applied; stages the ledger.

        A list without image blocks is returned as is -- the same object -- and
        only moves the ledger's per-prompt records one position, inline, since
        that touches no payload. Hashing is offloaded: a prompt can carry several
        multi-megabyte payloads, and digesting them on the event loop would pause
        every other session's streaming for the duration. A stage left over from
        a build that was never written is replaced, not accumulated.
        """
        self._pending = None
        self._pending_withheld = 0
        key, durable, ledger = self._current()
        if not has_image_blocks(blocks):
            result = apply_image_budget(blocks, ledger)
        else:
            result = await asyncio.to_thread(apply_image_budget, blocks, ledger)
        if result.ledger != ledger:
            self._pending = (key, durable, result.ledger)
        self._pending_withheld = result.over_budget
        if result.sent_earlier or result.over_budget or result.evicted:
            # Content-free counts only, like the structure summary logged
            # beside it: never a name, a path or a byte of an image.
            logger.info(
                "acp prompt: %d image block(s) inlined, %d dropped as sent earlier, "
                "%d dropped as over the image budget, %d ledger digest(s) evicted",
                result.inlined,
                result.sent_earlier,
                result.over_budget,
                result.evicted,
            )
        return result.blocks

    def commit(self) -> int:
        """Record the staged ledger: the prompt it describes reached the runtime.

        Called right after the ``session/prompt`` write succeeds. Returns how
        many of that prompt's image blocks were kept off the wire as over the
        budget, so the owner can tell its user; ``0`` when nothing was staged (a
        text-only prompt that moved no record, a command turn).
        """
        pending, self._pending = self._pending, None
        withheld, self._pending_withheld = self._pending_withheld, 0
        if pending is not None:
            self._record(*pending)
        return withheld

    def discard(self) -> None:
        """Drop the staged ledger: the prompt it describes was never written."""
        self._pending = None
        self._pending_withheld = 0

    def reset(self) -> None:
        """Forget every inlined image: the native conversation was emptied.

        A confirmed native clear keeps the session's ``sid`` while dropping its
        whole history, so nothing the ledger names is in the conversation any
        more and a picture attached again must be inlined again. Clears the
        durable record when the session has one, the in-memory ledger otherwise,
        and any stage in flight.
        """
        self._pending = None
        self._pending_withheld = 0
        key, durable, _ledger = self._current()
        sid = self._session_id() or ""
        self._local = empty_ledger(sid)
        if durable:
            store_image_ledger(key, empty_ledger(sid))

    def compacted(self) -> None:
        """Refund the ledger: kiro-cli compacted the conversation it describes.

        Called on the completed compaction status this session owns. The
        conversation keeps its ``sid``, so neither the sid scoping nor the clear
        reset fires, yet the older history -- and every image in it -- has just
        become summary text. :func:`compact_ledger` forgets the digests and keeps
        charged only the bytes of the prompts the kept tail can still replay, so
        a conversation that had reached its budget can inline images again. Only
        kiro-cli's compaction calls this: its kept tail is measured, and a refund
        against an unmeasured tail could re-open the growth the budget stops.
        """
        key, durable, ledger = self._current()
        self._record(key, durable, compact_ledger(ledger))
