"""The committed-generation pointer, read the same way by both processes.

Each cycle publishes its two authority files as a PAIR into a WRITER-UNIQUE generation --
``gen/<id>/session_map.json`` and ``gen/<id>/open_slots.json``, where ``<id>`` is minted
fresh by :func:`keys.new_generation_id` and never rewritten. A single object -- the
pointer -- names the generation id whose pair is committed, and the commit is a
compare-and-swap on that one object. Three consequences follow, and they are the whole
reason the protocol has this shape:

* Two writers racing in the task-replacement window each mint a DISTINCT id and write a
  distinct generation, so neither overwrites the other's pair and no committed pair is a
  cross-writer tear. The compare-and-swap on the pointer settles which one generation is
  the committed one.
* A cycle interrupted between the pair's two PUTs damages only its own generation, which
  no pointer references. The pointer still names the previous generation, whose objects
  are immutable and were never rewritten, so a replacement boots from a coherent older
  pair rather than a torn newer one.
* Commitment is a single object, and a single object is either there or it is not. There
  is no state in which half a commitment is visible.

The pointer's ABSENCE is meaningful rather than an error. A bucket written before this
protocol holds the authority objects at their ``data/`` keys with no pointer, and that is
read as GENERATION 0. Those objects are never deleted, moved or rewritten, so a bucket
does not have to be migrated to be read, and a writer that predates the protocol keeps
producing buckets this one understands.

The interpretation lives here and neither process keeps its own copy, for the reason
``keys.py`` gives for key derivation: the writer and the reader agreeing with each other
while both disagree with the contract is the failure this shape makes unrepresentable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from ..common import Settings, keys
from ..common.config import MAX_OBJECT_BYTES
from .store import ObjectAbsent, ObjectStore, StoreUnusable

log = logging.getLogger("smc.sidecar.generation")

__all__ = [
    "Pointer",
    "PointerUnusable",
    "read_pointer",
    "pointer_body",
]


class PointerUnusable(RuntimeError):
    """The pointer is present and cannot be used.

    Distinct from absent, and the distinction decides a boot. Absent means no generation
    has been committed, so the legacy keys are generation 0 and a task starts from them.
    Present-but-unusable means a generation may well be committed and this task cannot
    tell which -- reading that as absence would boot from objects the pointer was steering
    away from.
    """


@dataclass(frozen=True)
class Pointer:
    """The committed generation: which generation id, and which files it was committed with."""

    generation: str
    authority: frozenset[str]
    #: The MONOTONIC WRITER EPOCH this pointer was committed at: a non-negative integer that
    #: every commit advances by exactly one (``committed.epoch + 1``, or ``0`` for the first
    #: commit). It fences publication by RECENCY, not just by liveness. The compare-and-swap
    #: on the ETag below already rejects a commit whose read of the pointer is stale, but a
    #: CAS orders writes by the pointer's ETag alone -- it says only "no one has moved this
    #: object since I read it", not "the generation I am about to name is newer than the one
    #: committed". Two overlapping replacement writers, or a clock that stepped backward
    #: under the wall-clock timestamp in the generation id, can both read the same pointer
    #: and each hold a valid ``If-Match``; whichever's PUT lands second wins the CAS even
    #: when it names the OLDER work, rolling the committed generation backward. The epoch
    #: closes that: :func:`~..backup._commit_generation` refuses to publish an epoch that is
    #: not strictly greater than the committed one, so the loser of any race re-reads the now
    #: current pointer and rebuilds on it rather than clobbering the winner. The epoch is
    #: derived from the COMMITTED value (not from any writer's clock or identity), so it is
    #: monotonic by construction and immune to a wall-clock regression.
    epoch: int = 0
    #: The object's ETag when this pointer was read, or ``None`` when the store could not
    #: supply one. It is the compare-and-swap validator the commit re-presents as
    #: ``If-Match``: a writer that advanced the pointer since this read changes the ETag, so a
    #: stale commit is rejected rather than overwriting. ``None`` is a MISSING validator, and
    #: the commit fails CLOSED on it -- committing unconditionally would defeat the guard.
    #:
    #: The epoch and the ETag are two independent fences and BOTH are required: the ETag
    #: catches a concurrent write to the pointer object, and the epoch catches a write that
    #: would name older work even when the CAS itself would admit it.
    etag: str | None = None


def pointer_body(generation_id: str, epoch: int) -> bytes:
    """The pointer's bytes for a commitment of *generation_id* at monotonic *epoch*.

    One function so the writer's bytes and the reader's expectations cannot drift; the
    reader's own parsing is the other half and lives in :func:`read_pointer`. *epoch* is the
    monotonic writer epoch this commit publishes -- one greater than the committed pointer's,
    or ``0`` for the first commit -- and the reader carries it back on :class:`Pointer` so the
    next commit can refuse to regress it.
    """
    return json.dumps(
        {
            "generation": generation_id,
            "epoch": epoch,
            "authority": sorted(keys.AUTHORITY_NAMES),
        },
        sort_keys=True,
    ).encode("utf-8")


def read_pointer(settings: Settings, store: ObjectStore) -> Pointer | None:
    """The committed generation, or ``None`` when no pointer has been published.

    ``None`` is the generation-0 answer: the bucket either holds the legacy authority keys
    or holds nothing at all, and both are states a task may boot from.

    Raises :class:`PointerUnusable` for every other way this can go -- a read that fails,
    bytes that do not parse, a slot this writer does not publish into, a missing file list,
    an epoch that is not a non-negative integer. A pointer that exists and cannot be trusted
    is not permission to look elsewhere.
    """
    key = keys.authority_pointer_key(settings)
    try:
        raw, etag = store.get_with_etag(key, limit=MAX_OBJECT_BYTES)
    except ObjectAbsent:
        return None
    except StoreUnusable:
        # Not translated: the bucket itself cannot be read, so every later read meets the
        # same answer and the process must end on it rather than report a pointer problem.
        raise
    except Exception as exc:  # noqa: BLE001 - translated, never swallowed
        raise PointerUnusable(
            f"the generation pointer could not be read from the bucket ({exc}). This is "
            "not the same as it being absent: absent means no generation is committed and "
            "the legacy keys are this bucket's truth, while unreadable means one may be "
            "committed and this task cannot tell which."
        ) from exc
    # ``etag`` rode the SAME ``get_object`` as the bytes (one GET), so it is the validator for
    # exactly the pointer this cycle read -- no second HEAD a concurrent commit could slip
    # between (which would let a stale ``If-Match`` pass) and nothing unbudgeted for the final
    # cycle's drain window to be killed inside. A store that supplies no ETag leaves it None,
    # and the commit fails CLOSED on a missing validator.
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise PointerUnusable(
            f"the generation pointer in the bucket does not parse ({exc}), so it cannot "
            "say which generation is committed."
        ) from exc
    if not isinstance(parsed, dict):
        raise PointerUnusable(
            f"the generation pointer in the bucket is a JSON {type(parsed).__name__}, not "
            "an object, so it names no generation."
        )
    generation_id = parsed.get("generation")
    if not keys.is_generation_id(generation_id):
        raise PointerUnusable(
            f"the generation pointer names generation {generation_id!r}, which is not a "
            "generation id this writer mints (a zero-padded nanosecond timestamp, a hyphen "
            "and hex). No cycle of this writer published it, so the objects it points at are "
            "not a generation this task can read."
        )
    assert isinstance(generation_id, str)  # narrowed by is_generation_id above
    # The MONOTONIC EPOCH. A pointer written before this field existed has no ``epoch`` key,
    # and that reads as 0 rather than as an error, the same tolerance the missing-name rule
    # below relies on: a bucket written by an earlier writer stays readable, and the first
    # commit that carries an epoch advances it to 1. A present epoch must be a non-negative
    # int -- a float, a string, or a negative would let a later commit compute a successor
    # that is not strictly greater, defeating the fence, so it is unusable rather than
    # coerced. ``bool`` is an ``int`` subclass, so ``True``/``False`` is refused explicitly.
    raw_epoch = parsed.get("epoch", 0)
    if isinstance(raw_epoch, bool) or not isinstance(raw_epoch, int) or raw_epoch < 0:
        raise PointerUnusable(
            f"the generation pointer carries epoch {raw_epoch!r}, which is not a "
            "non-negative integer, so the monotonic fence that keeps a commit from rolling "
            "the committed generation backward cannot advance from it."
        )
    listed = parsed.get("authority")
    if not isinstance(listed, list) or not all(isinstance(name, str) for name in listed):
        raise PointerUnusable(
            "the generation pointer has no 'authority' list of names, so it cannot say "
            "which files the committed generation contains."
        )
    named = frozenset(listed)
    missing = [name for name in keys.AUTHORITY_NAMES if name not in named]
    if missing:
        raise PointerUnusable(
            f"the generation pointer commits a generation without {', '.join(missing)}, "
            "and this writer commits the authority pair whole or not at all. Read as a "
            "partial generation it would present a name this task knows as legitimately "
            "absent, and the backend would flush its own empty view over it -- so the "
            "pointer is unusable rather than a generation missing a member."
        )
    # Names this version does not know are dropped, not refused: a bucket written by a
    # newer writer that commits a third authority file still names a generation whose
    # pair this one can read, and refusing it would make a rollback unbootable. The
    # check above is what keeps that tolerance from also admitting a pointer that
    # under-lists a name this version DOES know.
    return Pointer(
        generation=generation_id,
        authority=frozenset(n for n in named if n in keys.AUTHORITY_NAMES),
        epoch=raw_epoch,
        etag=etag,
    )
