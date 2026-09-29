"""One backup cycle: what must be durable, and how each object is copied.

## What is in the set

Three kinds, and the set is a definition rather than a filter, so "was this file
backed up" has an answer that does not depend on what the directory happened to
hold:

1. **The two authority files**, ``session_map.json`` and ``open_slots.json``. They
   turn a slot id back into a conversation, so without them the transcripts are on
   disk and the conversation list is empty.
2. **Every live transcript**, the ``.jsonl`` files directly under the sessions
   directory. One per conversation this task has served.
3. **Every archived segment** under ``sessions/archive/``. Rotation moves the older
   part of a long conversation there and the container never reads it back -- the
   front's fetch may not list, and finding a segment requires listing -- so these are
   uploaded for the owner's control plane, which has credentials of its own. Leaving
   them behind would be silent loss of the older half of every long conversation.

Anything else under the sessions directory is not in the set. The front writes a
temporary file there while fetching and unlinks it itself, and a name that is
neither that nor a transcript is not a conversation.

## What order they go in, and why the order is a correctness rule

The transcripts go first and the authority files last, in their own phase. The
authority files are the INDEX: a replacement reads them to decide which conversations
exist, and the front then fetches each named transcript lazily. So an authority table
newer than the transcripts it names points at objects that are not in the bucket, and
the front reads an absent transcript as a conversation with no history -- a live
conversation served empty, with nothing raised anywhere. The opposite skew is
harmless: an authority table older than the transcripts names only slots whose bytes
are already there, and a transcript it does not name yet is unreferenced rather than
misread.

Which is why the authority files are OPENED first, before a single transcript is
listed, and sent from those descriptors at the end. Opening fixes the instant a file
describes, so the pair is one coherent snapshot of the index taken before the
enumeration it indexes. Reading them at send time instead let a slot table flushed
during the cycle name a transcript that cycle never listed.

That rests on the backend publishing both files the way it publishes a transcript, a
temporary file and a rename, which leaves an open descriptor addressing the whole
previous version. It does: ``session_map.json`` and ``open_slots.json`` are both written
through an atomic replace. A writer that truncated one in place instead would take the
snapshot property away without changing anything here, so it is pinned by a test rather
than left as an assumption.

The authority phase is SKIPPED when the transcript phase suffered a refusal a LATER
CYCLE COULD GET PAST -- a failed upload, an object the drain window could not fit, a
directory missing right now. Publishing it then would advance the index past bytes this
cycle failed to write; withholding it leaves the pair at the last cycle that completed,
which is older and coherent, and the next cycle publishes a pair the bucket supports.

It is NOT skipped for a refusal decided by an entry's SHAPE -- a symlink or a FIFO where
a transcript belongs, a linked archive root. Withholding is a WAIT, and every later cycle
meets that entry too, so the wait never ends: the index would freeze at the moment the
entry appeared while transcripts kept uploading past it, and the next replacement would
restore a conversation list predating every conversation served since. The cycle is still
incomplete and the entry is still named; see :class:`RefusedEntry` for the split and for
the bounded residue it accepts in exchange.

One residual remains in the pair itself. The two files are two PUTs, so a failure
between them leaves the bucket holding one from this cycle's snapshot and one from an
earlier cycle's. Both were opened before this cycle's enumeration, so neither names a
transcript that is absent, and the failure raises rather than passing quietly; the cost
is one interval in which the two files disagree about which slots exist, which the next
cycle resolves.

## How one object is copied

``open_snapshot`` opens the file ONCE and records the length that descriptor's file
had at that moment. The upload then sends exactly that many bytes from that
descriptor. Three properties follow, and they are the three constraints this design
has to hold at the same time:

* **Consistent** without a lock. The backend publishes a transcript with a temporary
  file and a rename, so it never writes into the bytes behind an open descriptor --
  it swaps the directory entry to a different inode. A descriptor opened before the
  swap keeps addressing a whole, finished version, and a file that is appended to
  instead is uploaded as the prefix that existed at open time, which is also a
  version that was really on disk.
* **Bounded** on disk. Nothing is copied first. A cycle spends one descriptor and one
  fixed transport buffer per object, so an oversized artifact cannot fill the
  filesystem the app is writing to.
* **Nothing dropped.** There is no size at which an object is skipped. An entry that
  genuinely cannot be uploaded is recorded and the cycle ends by RAISING
  :class:`BackupIncomplete` -- after uploading everything it could, so one bad entry
  does not cost every other conversation its backup.

## Every shape an entry can have, and what happens to it

| entry                                      | verdict                                |
| ------------------------------------------ | -------------------------------------- |
| regular file, one link                     | uploaded                               |
| regular file, several links                | uploaded: the descriptor still         |
|                                            | addresses real bytes, and this side     |
|                                            | only reads them                        |
| regular file that grew since it was opened  | uploaded to its length at open         |
| regular file that shrank since it was opened| uploaded short, and the declared length |
|                                            | makes the transport fail rather than    |
|                                            | pad; recorded, so the cycle raises      |
| zero bytes                                 | uploaded: an empty conversation is a    |
|                                            | conversation                            |
| above the reader's ceiling                 | uploaded, with a warning naming it:     |
|                                            | backed up, and the front will refuse to |
|                                            | restore it, so an operator hears it     |
|                                            | before a customer does                  |
| symlink                                    | recorded; the cycle raises              |
| reached through a symlinked directory      | recorded; the cycle raises, and a       |
|                                            | linked archive root is refused before    |
|                                            | anything under it is listed at all       |
| directory, FIFO or socket                  | recorded; the cycle raises              |
| gone between listing and opening           | counted as gone; the cycle continues,   |
|                                            | because a deleted conversation is not   |
|                                            | a backup failure                        |
| unchanged since its last upload            | not re-uploaded                         |

## What the fingerprint is for, and what it is not

Change detection is a COST decision, not a correctness one. The fingerprint is the
inode, the length and the modification time as they were at open, and an object is
re-uploaded whenever it differs from the one last uploaded successfully. It lives in
memory, so a restarted sidecar re-uploads everything once: paying for a full cycle is
the right way to be wrong here, and persisting the state would put a second authority
on disk to keep in agreement with the bucket.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import logging
import os
import stat
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Iterator

from ..common import Settings, keys
from ..common.config import (
    BACKUP_ATTEMPT_COST_SECS,
    BACKUP_PER_OBJECT_BUDGET_SECS,
    MAX_OBJECT_BYTES,
)

# One spelling of the transcript filename, rather than a second copy of the prefix rule
# here: the index names a conversation by its slot key and the file carries a
# ``dashboard_`` prefix plus a character substitution, so a local re-derivation would be a
# second definition of the same mapping and would drift from the one that names the file.
from ..front.transcript import transcript_stem
from . import generation
from .store import (
    ObjectStore,
    PreconditionFailed,
    StoreUnusable,
    UploadCancelled,
    UploadDeadlineExceeded,
)

log = logging.getLogger("smc.sidecar.backup")

__all__ = [
    "Fingerprint",
    "Snapshot",
    "BackupSet",
    "CycleResult",
    "BackupIncomplete",
    "open_snapshot",
    "objects_to_back_up",
    "run_cycle",
]

#: Flags for opening a file to be uploaded.
#:
#: ``O_NOFOLLOW`` refuses a symlink at the final component, so a link planted where a
#: transcript belongs is reported instead of followed to whatever it points at.
#: ``O_NONBLOCK`` is what keeps the open from hanging: opening a FIFO for reading blocks
#: until a writer arrives, and an entry planted as a FIFO would otherwise stall the cycle
#: indefinitely rather than be refused.
_OPEN_FLAGS: int = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)

#: Flags for opening one DIRECTORY component on the way down to a file.
#:
#: ``O_NOFOLLOW`` is what makes the descent safe. ``O_NOFOLLOW`` on the file alone
#: guards only the last name, so a link planted at ``sessions/archive`` -- a directory
#: the agent writes in -- is descended normally and every regular file behind it opens
#: and uploads. Walking down with this flag at each step means a link ANYWHERE in the
#: chain is refused instead, so the only files that reach the bucket are files reached
#: through real directories inside the data home.
_DIR_FLAGS: int = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)

#: How many unreached objects a stop or deadline gate names BY NAME before it stops looking
#: and records one count-free remainder marker instead. The gate is reached mid-stream over a
#: LAZY iterator whose tail is the archive tree, and retention-off lets that tree grow without
#: bound -- so draining the iterator to name every unreached object rebuilds the whole
#: inventory in a list at exactly the moment the cycle is trying to stop, the same
#: unbounded-retention hazard the streamed enumeration removes from the walk. The cycle's
#: completeness reads only whether ANYTHING was refused (see :attr:`CycleResult.complete`), so
#: the current item alone already makes it incomplete; the bounded sample gives an operator
#: names to act on without walking the tail. Sized to match the archive sample cap so a small
#: remainder is named whole and the tests that assert exact skipped-name sets are unaffected.
_UNREACHED_SAMPLE_CAP = 64

#: The count-free marker recorded once when the unreached remainder runs past the sample cap.
#: It is a refusal like any other -- it withholds the pair and makes the cycle incomplete --
#: but names no object, because naming the rest would mean draining the tail this marker
#: exists to avoid.
_UNREACHED_REMAINDER_NAME = "<remainder>"

#: How many archive-object fingerprints the lifetime *state* map keeps before the OLDEST are
#: evicted. The map exists so an unchanged object is not re-uploaded, and for a live transcript
#: (a bounded, flat set) it stays small. The ARCHIVE is different: rotation nests it and
#: retention-off never prunes it, so one entry per archived segment would grow the map without
#: bound for the process's whole life -- state tracking the very thing the streamed enumeration
#: was made to stop holding. Archive segments are IMMUTABLE, so the only cost of forgetting one
#: is re-uploading identical bytes once; that makes a bounded, evict-oldest cap safe where it
#: would not be for a mutable object. Sized well above any real cycle's live+recent-archive
#: working set, so an ordinary run never evicts and only a pathologically large archive does.
_ARCHIVE_STATE_CAP = 4096

#: How many entries the bounded per-object records keep BY NAME before keeping only the
#: running count. The ONE cap for every structure that samples the same object population:
#: :class:`CycleResult`'s uploaded/unchanged/gone/refused/unreachable records AND
#: :class:`_ArchiveWalkSink`'s archive-walk refusals, which fold into ``CycleResult.refused``.
#: Both are capped here so one refusal population is not truncated by two different numbers.
#: The archive puts one entry per segment through these every cycle, so an unbounded list
#: would hold the whole inventory for the log's sake alone. Sized well above any real cycle's
#: working set, so an ordinary run's lists are complete and the tests that assert exact
#: contents are unaffected.
_RESULT_SAMPLE_CAP = 4096

#: How many characters of a refusal's REASON are kept in the sample. A refusal carries an
#: exception string, and a store or transport error can be arbitrarily long (a stack-shaped
#: message, an echoed payload), so a capped COUNT of entries is not enough on its own -- one
#: entry can still be huge. The reason is truncated to this, with a marker, so a sampled
#: refusal costs a bounded number of bytes. The name is a key this task formed and is already
#: bounded, so only the reason is cut.
_REFUSAL_REASON_MAX_CHARS = 200


class _ArchiveWalkSink:
    """The bounded record of what the archive walk could not enumerate.

    :func:`_archived_segments` streams file paths so the whole inventory is never held in a
    list at any level. Its REFUSALS -- a directory the walk met an error on, a linked
    subdirectory it dropped -- are the other per-archive collection, and are bounded here to
    a capped sample plus a true count rather than an unbounded list. ``refused`` withholds
    the authority pair; ``unreachable`` does not, split on the same permanence rule
    :class:`RefusedEntry` states.

    The counts are what the cycle's completeness and withhold decisions read (a refusal is
    present or it is not); the sampled names are for the log and the incompleteness report.
    """

    __slots__ = ("refused", "unreachable", "refused_count", "unreachable_count")

    def __init__(self) -> None:
        self.refused: list[tuple[str, str]] = []
        self.unreachable: list[tuple[str, str]] = []
        self.refused_count = 0
        self.unreachable_count = 0

    def reset(self) -> None:
        """Empty the record for a fresh walk.

        ``BackupSet.data`` is re-iterable and each iteration re-walks the archive, so the
        sink is reset at the start of each walk -- it then reflects the most recent walk,
        which in the one place that reads it (:func:`run_cycle`, after its single upload-phase
        consumption) is the only walk.
        """
        self.refused = []
        self.unreachable = []
        self.refused_count = 0
        self.unreachable_count = 0

    def refuse(self, entry: tuple[str, str]) -> None:
        self.refused_count += 1
        if len(self.refused) < _RESULT_SAMPLE_CAP:
            self.refused.append(entry)

    def cannot_reach(self, entry: tuple[str, str]) -> None:
        self.unreachable_count += 1
        if len(self.unreachable) < _RESULT_SAMPLE_CAP:
            self.unreachable.append(entry)


@dataclass(frozen=True)
class Fingerprint:
    """What an object looked like when it was last uploaded successfully."""

    inode: int
    size: int
    mtime_ns: int


class DurableState(dict):  # type: ignore[type-arg]
    """The lifetime "already uploaded" map, with an O(1) archive-population cap.

    It IS a ``dict`` -- every reader and writer treats it as ``dict[str, Fingerprint]`` --
    with one addition: a companion FIFO, :attr:`archive_order`, that :func:`_record_durable`
    keeps in step with the archive keys it inserts, so the oldest archive key is found and
    the archive population sized without rescanning the whole map on every insertion. That
    rescan would be O(n) per insert and O(n^2) across one large archive walk, the very
    unbounded cost the archive cap exists to prevent -- so the container that holds the cap
    must not reintroduce it. Only archive keys go in the FIFO; live and authority keys are
    the bounded flat set the map holds whole and never counts against the cap.

    A plain ``dict`` (a unit test that passes ``{}``) has no companion and :func:`_record_durable`
    falls back to a scan -- correct, just not O(1); the process constructs this.
    """

    def __init__(self) -> None:
        super().__init__()
        self.archive_order: deque[str] = deque()


@dataclass(frozen=True)
class Snapshot:
    """An open descriptor and the length its file had when it was opened.

    The pair IS the snapshot. Neither half is a snapshot alone: the descriptor without
    the length would upload however much had arrived by the time the transport got
    there, and the length without the descriptor would have to re-open the name, which
    is a second resolution of one path with a window in between.
    """

    fh: BinaryIO
    fingerprint: Fingerprint

    @property
    def size(self) -> int:
        return self.fingerprint.size

    def close(self) -> None:
        self.fh.close()


class RefusedEntry(RuntimeError):
    """This entry is not a file whose bytes can be uploaded.

    *permanent* says whether a LATER cycle could reach it. An entry's SHAPE is what makes
    a refusal permanent -- a symlink where a transcript belongs, a FIFO, a linked archive
    root -- and nothing the backup does changes it: every cycle meets the same answer
    until someone removes the name. A transient refusal is the ordinary case and the
    opposite: a directory missing right now, a descriptor that could not be opened, with
    bytes that may be perfectly live behind it.

    The cycle needs the difference because WITHHOLDING THE AUTHORITY PAIR IS A WAIT. It
    holds the index back one cycle so the next one can publish a pair the bucket's objects
    support, which is right when what it waits for will arrive. Against a permanent
    refusal the wait never ends: the pair is withheld on every later cycle too, the index
    freezes at the moment the entry appeared, and a replacement task restores a
    conversation list that predates every conversation served since -- while their
    transcripts keep uploading, unreferenced.

    So a permanent refusal still makes the cycle incomplete, and it does not withhold. The
    residue is stated rather than hidden: if a planted name happens to collide with a
    session the index does name, that one conversation's history is absent behind an index
    that advanced past it. That is bounded, it is named in the cycle's own report, and
    removing the file repairs it -- where the freeze is unbounded, silent, and repaired by
    nothing.
    """

    def __init__(self, message: str, *, permanent: bool = False) -> None:
        super().__init__(message)
        self.permanent = permanent


def _descend(root: Path, parts: tuple[str, ...]) -> int:
    """Open the directory at *root* / *parts*, refusing a symlink at any component.

    Returns a descriptor the caller must close. ``ELOOP`` from any step means a
    directory in the chain is a link, which is refused rather than followed: a link out
    of the data home turns "back up this task's own state" into "upload whatever it
    points at", and the sessions tree is one the agent writes in.
    """
    fd = os.open(str(root), _DIR_FLAGS)
    try:
        for name in parts:
            nxt = os.open(name, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nxt
    except OSError:
        os.close(fd)
        raise
    return fd


def _open_within(root: Path, path: Path) -> int:
    """Open *path* for reading, having walked to it from *root* one component at a time.

    *root* is the trust anchor -- the data home, which is the container's own mount and
    not a path the agent can replace. Every component below it is opened with the link
    refused, so the descriptor returned addresses a file inside the real data home and
    not one reached through a directory something swapped for a link.

    Opening the full path in one call cannot do this: ``O_NOFOLLOW`` applies to the last
    component only, and the kernel resolves the rest normally.

    A missing DIRECTORY on the way down is a refusal, while a missing leaf is left to the
    caller as ``FileNotFoundError``. The two are different events wearing one errno: the
    leaf is a conversation the owner deleted, and a directory is this task's whole state
    tree becoming unreachable -- an unmounted data home, a removed ``sessions/archive`` --
    with the transcripts still live behind it. Read as a deletion, that publishes an index
    for conversations the cycle never looked at.
    """
    rel = path.relative_to(root)
    parts = rel.parts
    try:
        fd = _descend(root, parts[:-1])
    except FileNotFoundError as exc:
        raise RefusedEntry(
            f"a directory on the way down to it is missing ({exc}); the file itself was "
            "listed moments ago, so its bytes are not known to be gone -- this is the data "
            "home or a directory inside it becoming unreachable, which must not be recorded "
            "as a conversation the owner deleted"
        ) from exc
    try:
        return os.open(parts[-1], _OPEN_FLAGS, dir_fd=fd)
    finally:
        os.close(fd)


@dataclass
class CycleResult:
    """What one cycle did, per object, for the log and for the tests."""

    #: The keys uploaded / found unchanged this cycle. Bounded: a capped SAMPLE of the keys
    #: plus a true count (``uploaded_count`` / ``unchanged_count``), because the archive
    #: contributes one entry per segment every cycle and retention-off lets the archive grow
    #: without bound -- an unbounded list here would hold the whole inventory the streamed
    #: enumeration exists to avoid holding. Nothing reads their IDENTITY to decide anything
    #: (the summary reads the counts; the withhold and completeness checks read ``refused`` /
    #: ``unreachable`` / ``gone_referenced``), so a sample serves the log while the count
    #: stays exact. Recorded through :meth:`record_uploaded` / :meth:`record_unchanged`.
    uploaded: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    uploaded_count: int = 0
    unchanged_count: int = 0
    gone: list[str] = field(default_factory=list)
    above_ceiling: list[str] = field(default_factory=list)
    #: Exact counts for ``gone`` / ``above_ceiling`` above and ``gone_undurable`` /
    #: ``gone_referenced`` below. Same reason as the other lists: the archive puts one entry
    #: per segment through each of these, so they are capped, count-tracked samples. Nothing
    #: reads their identity to decide anything -- the withhold decision reads
    #: ``gone_referenced_count`` and the summary reads the counts.
    gone_count: int = 0
    above_ceiling_count: int = 0
    refused: list[tuple[str, str]] = field(default_factory=list)
    #: Entries whose BYTES could not be reached at all: a name that is not a regular file,
    #: an archive root that is a link, a subtree that could not be enumerated. Held apart
    #: from ``refused`` because the two decide the authority pair differently, and only
    #: their CAUSE says which is which.
    #:
    #: A refusal means an object the pair can name is not in the bucket, so publishing the
    #: pair would point the front at bytes that are not there -- the pair is withheld. An
    #: unreachable entry is not an object the pair names: the backend never wrote it, so
    #: the index does not reference it and withholding the pair protects nothing. It only
    #: freezes it, and it freezes it FOREVER, because a planted name stays planted: every
    #: later cycle meets the same entry, the pair is never republished, and a replacement
    #: task restores an index from before the entry appeared while transcripts keep
    #: uploading past it.
    #:
    #: The cycle is incomplete either way, which is why this is a second list and not a
    #: log line: something in the set did not reach the bucket and the exit code has to
    #: say so.
    unreachable: list[tuple[str, str]] = field(default_factory=list)
    #: Exact counts for the two lists above. Like ``uploaded``/``unchanged``, the lists are a
    #: capped, length-bounded SAMPLE while these stay exact: on a pathological run every
    #: archived segment can fail its upload, so an entry per failure -- each carrying a full
    #: exception string -- would retain the whole inventory on the failure path, the same
    #: unbounded-retention hazard the streamed enumeration removes from the success path. The
    #: withhold and completeness checks read only whether these are NON-EMPTY (see
    #: :attr:`complete`), which the sample preserves, and the summary reads the count.
    refused_count: int = 0
    unreachable_count: int = 0
    #: Entries that vanished between the listing and their open with nothing in the bucket:
    #: candidates only, and not a verdict. Whether one matters depends on the captured
    #: index, which the authority phase reads -- so this list is informational and
    #: ``gone_referenced`` is what decides anything.
    gone_undurable: list[tuple[str, str]] = field(default_factory=list)
    gone_undurable_count: int = 0
    #: The ``gone_undurable`` entries the CAPTURED authority pair actually names. These
    #: decide the pair like a refusal rather than like an unreachable entry, and the reason
    #: is the premise the whole skew argument rests on: an index OLDER than its bytes is
    #: harmless only because every slot it names already has bytes in the bucket. The pair
    #: is captured BEFORE the enumeration, so it still names a conversation deleted during
    #: the cycle -- and when that conversation was created and deleted inside one interval,
    #: no earlier cycle uploaded it, so the committed index would name a slot the bucket has
    #: never held. The front then fetches an absent object and reads it as a conversation
    #: that never had history.
    #:
    #: Membership in the captured index is the whole test, not the disappearance. A
    #: conversation the index does not name is not something the pair can send a reader to,
    #: so racing its deletion stays the non-failure it has always been -- which is the
    #: routine case, since an owner deleting a conversation is ordinary use.
    #:
    #: Withholding here cannot freeze the index, which is what separates it from an
    #: unreachable entry: this verdict is a race WITHIN one cycle, not a shape on disk. A
    #: file that is genuinely deleted is not listed by the next cycle at all, so it cannot
    #: be gone again, and the pair publishes on that next cycle.
    gone_referenced: list[str] = field(default_factory=list)
    gone_referenced_count: int = 0
    #: Authority-phase keys this cycle did not publish, so they stay as the last complete
    #: cycle left them: the pair when a transcript in the same cycle was refused, and the
    #: generation pointer when the pair is whole but the pointer itself could not be sent
    #: on an interval cycle. On the final cycle an unsent pointer is a refusal, not this.
    withheld: list[str] = field(default_factory=list)

    def record_uploaded(self, key: str) -> None:
        """Count an uploaded key, keeping its name only while under the sample cap."""
        self.uploaded_count += 1
        if len(self.uploaded) < _RESULT_SAMPLE_CAP:
            self.uploaded.append(key)

    def record_unchanged(self, key: str) -> None:
        """Count an unchanged key, keeping its name only while under the sample cap."""
        self.unchanged_count += 1
        if len(self.unchanged) < _RESULT_SAMPLE_CAP:
            self.unchanged.append(key)

    @staticmethod
    def _bounded_reason(reason: str) -> str:
        if len(reason) <= _REFUSAL_REASON_MAX_CHARS:
            return reason
        return reason[:_REFUSAL_REASON_MAX_CHARS] + "… (truncated)"

    def record_refused(self, name: str, reason: str) -> None:
        """Count a refusal, keeping a length-bounded sample while under the cap.

        A refusal withholds the authority pair and makes the cycle incomplete; the checks
        read whether ANY refusal is present, which the sample preserves, so a bounded sample
        plus the exact count carries every decision while a pathological run -- every archived
        segment failing its upload -- cannot retain the whole inventory here.
        """
        self.refused_count += 1
        if len(self.refused) < _RESULT_SAMPLE_CAP:
            self.refused.append((name, self._bounded_reason(reason)))

    def record_unreachable(self, name: str, reason: str) -> None:
        """Count an unreachable entry, keeping a length-bounded sample while under the cap."""
        self.unreachable_count += 1
        if len(self.unreachable) < _RESULT_SAMPLE_CAP:
            self.unreachable.append((name, self._bounded_reason(reason)))

    def record_gone(self, name: str) -> None:
        """Count a gone object, keeping its name only while under the sample cap."""
        self.gone_count += 1
        if len(self.gone) < _RESULT_SAMPLE_CAP:
            self.gone.append(name)

    def record_above_ceiling(self, key: str) -> None:
        """Count an over-ceiling object, keeping its key only while under the sample cap."""
        self.above_ceiling_count += 1
        if len(self.above_ceiling) < _RESULT_SAMPLE_CAP:
            self.above_ceiling.append(key)

    def record_gone_undurable(self, name: str, key: str) -> None:
        """Count a vanished-and-undurable candidate, keeping a sample while under the cap.

        A candidate only, not a verdict -- :meth:`record_gone_referenced` is what the withhold
        decision reads. Bounded like the rest because the archive can put one entry per segment
        through here on a pathological run.
        """
        self.gone_undurable_count += 1
        if len(self.gone_undurable) < _RESULT_SAMPLE_CAP:
            self.gone_undurable.append((name, key))

    def record_gone_referenced(self, name: str) -> None:
        """Count a vanished object the captured index names, keeping a sample under the cap.

        This is the verdict the withhold decision reads -- via ``gone_referenced_count``, which
        stays exact so a run that overflowed the sample still withholds the pair.
        """
        self.gone_referenced_count += 1
        if len(self.gone_referenced) < _RESULT_SAMPLE_CAP:
            self.gone_referenced.append(name)

    def fold_archive_refusals(self, sink: "_ArchiveWalkSink") -> None:
        """Merge a streamed archive walk's refusals in, preserving its exact counts.

        The sink already holds a capped sample plus a TRUE count; folding it must not shrink
        the count to the sample it carries. So the counts take the sink's true totals and the
        sample entries fill this result's own sample only up to its cap -- a walk that met more
        faults than either cap keeps a bounded sample and the exact number, never one entry per
        fault.
        """
        self.refused_count += sink.refused_count
        for name, reason in sink.refused:
            if len(self.refused) >= _RESULT_SAMPLE_CAP:
                break
            self.refused.append((name, self._bounded_reason(reason)))
        self.unreachable_count += sink.unreachable_count
        for name, reason in sink.unreachable:
            if len(self.unreachable) >= _RESULT_SAMPLE_CAP:
                break
            self.unreachable.append((name, self._bounded_reason(reason)))

    @property
    def complete(self) -> bool:
        # ``gone_referenced`` counts because it WITHHOLDS the pair: a cycle that did not
        # preserve the index has not done its job, and on the final cycle that verdict is
        # the exit code, which is the only way the loss is announced rather than silent.
        # ``gone_undurable`` does NOT count: it is a candidate list, and a candidate the
        # captured index never named cost the pair nothing. Read the COUNTS, not the sample
        # lists: a run that overflowed a sample cap still had those events and must still be
        # incomplete even if the sample were somehow shorter.
        return (
            self.refused_count == 0
            and self.unreachable_count == 0
            and self.gone_referenced_count == 0
        )

    def summary(self) -> str:
        # Every tally reads the true COUNT, not the sample length, so a run that overflowed a
        # sample cap reports how much it did rather than the cap.
        return (
            f"{self.uploaded_count} uploaded, {self.unchanged_count} unchanged, "
            f"{self.gone_count} gone ({self.gone_referenced_count} of them named by the "
            f"captured index and not in the bucket), "
            f"{self.refused_count} refused, "
            f"{self.unreachable_count} unreachable, "
            f"{len(self.withheld)} authority withheld"
        )


class BackupIncomplete(RuntimeError):
    """At least one object in the set could not be uploaded.

    Raised at the END of the cycle, with everything that could be uploaded already
    uploaded. The distinction matters: refusing the whole cycle on the first bad entry
    would cost every other conversation its backup, and dropping the bad entry with a
    log line would be the silent loss this design exists to prevent. So the cycle does
    all the work it can and then cannot be ignored.

    Both causes are named, because the remedies differ: a refused upload is retried by
    the next cycle, while an unreachable entry stays unreachable until someone removes
    the name -- and an operator reading only "could not be uploaded" would wait for a
    retry that can never succeed. A third cause needs neither remedy: an entry that
    vanished mid-cycle while the captured index still named it, with nothing in the bucket
    behind that name, is listed so the reason the pair was withheld is legible -- and the
    next cycle, which will not list that name at all, publishes the pair with no
    intervention.
    """

    def __init__(self, result: CycleResult) -> None:
        self.result = result
        blocked = result.refused + result.unreachable
        blocked = blocked + [
            (name, "vanished mid-cycle while the captured index still named it")
            for name in result.gone_referenced
        ]
        detail = "; ".join(f"{name}: {why}" for name, why in blocked)
        super().__init__(
            f"{len(blocked)} object(s) in the backup set did not reach the bucket "
            f"({len(result.refused)} refused, {len(result.unreachable)} unreachable) "
            f"({detail}). Everything else in this cycle was uploaded."
        )


def open_snapshot(path: Path, *, root: Path) -> Snapshot | None:
    """Open *path* for upload, or ``None`` when it is not there any more.

    ``None`` means the LEAF was listed and then removed, which is a conversation the
    owner deleted rather than a backup failure. A directory on the way down being absent
    is not that -- the bytes behind it may be live -- so it raises like every other way
    this can fail: :class:`RefusedEntry`, because those are entries whose bytes cannot be
    shown to belong in the bucket or shown to be gone.

    Shape is decided on the DESCRIPTOR, never on the name: a check by name followed by
    an open by name is two resolutions of one path with a window in between. Opening
    first with the link refused and then reading ``fstat`` off the descriptor means the
    entry judged is exactly the entry that will be uploaded.

    *root* is the data home, and the open walks down to *path* from it one component at
    a time with each link refused, so an ancestor directory replaced by a link is a
    refusal here and not a file uploaded from outside the data home.
    """
    try:
        fd = _open_within(root, path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            # Both codes mean the same refusal. ``O_NOFOLLOW`` on a symlink reports
            # ELOOP for the final component and, combined with ``O_DIRECTORY``, ENOTDIR
            # for a directory component -- so the two are one case: something on the way
            # to this file is a link or is not the directory it is supposed to be.
            raise RefusedEntry(
                f"it, or a directory on the way down to it, is a symlink or is not a "
                f"directory ({exc}); a link where this task's own state belongs points "
                "at bytes it does not own",
                permanent=True,
            ) from exc
        raise RefusedEntry(f"it could not be opened ({exc})") from exc
    try:
        st = os.fstat(fd)
    except OSError as exc:  # pragma: no cover - fstat on a fresh descriptor
        os.close(fd)
        raise RefusedEntry(f"its shape could not be read ({exc})") from exc
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise RefusedEntry(
            f"it is not a regular file (mode {st.st_mode:#o}); a directory, socket or "
            "FIFO holds no transcript bytes to upload",
            permanent=True,
        )
    return Snapshot(
        fh=os.fdopen(fd, "rb"),
        fingerprint=Fingerprint(inode=st.st_ino, size=st.st_size, mtime_ns=st.st_mtime_ns),
    )


def _live_transcripts(settings: Settings) -> tuple[list[Path], list[tuple[str, str]]]:
    """The ``.jsonl`` files directly under the sessions directory, and any root refusal.

    Returns ``(found, refused)``. A missing directory yields nothing and refuses nothing: a
    task that has served no turn has no sessions directory yet, and that is a first boot
    rather than a fault.

    The root is opened with every link refused BEFORE anything under it is listed, for the
    reason the archive root is: ``os.scandir`` resolves the path it is given, so a link
    planted at ``sessions/`` is followed and its target's files are listed as this task's
    transcripts. Each one is then refused on the way down -- :func:`_open_within` walks from
    the data home with ``O_NOFOLLOW`` and raises at the ``sessions`` component -- so the
    cycle uploads nothing while every refusal is a per-ENTRY one. Listing through the link
    and refusing afterwards is what turns a replaced root into an index that names
    conversations whose bytes are not in the bucket.

    A root refusal WITHHOLDS the authority pair, and unlike a per-entry refusal it does so
    whatever its cause. The residue :class:`RefusedEntry` accepts is one planted NAME among
    transcripts that are otherwise reaching the bucket, where freezing the index for all of
    them is the worse trade. A refused ROOT is the whole tree: nothing is reaching the
    bucket, so an index that advances describes a state the bucket does not hold at all,
    and the pointer staying at the last complete generation costs nothing that was going to
    be uploaded anyway.
    """
    root = settings.sessions_dir
    try:
        fd = _descend(settings.data_home, root.relative_to(settings.data_home).parts)
    except FileNotFoundError:
        return [], []
    except OSError as exc:
        return [], [
            (
                root.name,
                f"the sessions directory, or a directory above it, could not be opened as "
                f"a real directory inside the data home ({exc}); nothing under it is "
                "listed, because a link there names files this task does not own, and the "
                "authority pair is withheld rather than published over a tree that was "
                "never enumerated",
            )
        ]
    # Listed THROUGH that descriptor, not by the name again. Checking a descended descriptor
    # and then re-resolving the root by name is two lookups of one directory: the root can be
    # swapped between them, so the listing walks the very tree the check refused while the
    # check reports it sound -- the same one-directory-throughout rule publication follows,
    # and the refusal is worth nothing without it.
    # The descriptor is held until the LAST ``DirEntry`` has been classified, not released
    # when the listing ends. ``os.scandir(fd)`` hands back entries whose ``dir_fd`` is this
    # very descriptor and whose ``path`` is the bare name, so an entry whose type ``readdir``
    # did not report -- ``DT_UNKNOWN``, which CPython documents for network filesystems and
    # the EFS-mounted data home is one -- answers ``is_dir`` by ``fstatat`` through it. Closed
    # first, that call is ``EBADF`` on exactly the filesystem this task runs on, and it raises
    # from inside ``objects_to_back_up``, which ``run_cycle`` calls ABOVE its own ``try`` --
    # so every cycle would die identically for the life of the task and nothing would ever be
    # uploaded. The close stays in a ``finally`` so it still happens on the early return and
    # on any raise.
    try:
        try:
            with os.scandir(fd) as scan:
                entries = sorted(scan, key=lambda e: e.name)
        except FileNotFoundError:
            return [], []
        found: list[Path] = []
        for entry in entries:
            if not entry.name.endswith(keys.TRANSCRIPT_SUFFIX):
                continue
            # ``follow_symlinks=False`` so a link to a directory is not read as one file;
            # the shape is decided again on the descriptor, and this only decides what to
            # put in the list.
            if entry.is_dir(follow_symlinks=False):
                continue
            found.append(root / entry.name)
        return found, []
    finally:
        os.close(fd)


def _is_shape_error(exc: OSError) -> bool:
    """Whether this failure is the PATH's shape rather than a condition that may pass.

    ``ELOOP`` and ``ENOTDIR`` say the name is a link or is not a directory, and nothing the
    backup does changes that -- every later cycle meets the identical error. That is exactly
    the permanence :class:`RefusedEntry` splits on, and it decides the authority pair: a
    withholding that can never end freezes the index instead of protecting it.

    Everything else is treated as passable, which is the safe direction for an unknown errno:
    it withholds the pair for one cycle rather than letting the index advance over a subtree
    that may be perfectly live behind a transient fault.
    """
    return exc.errno in (errno.ELOOP, errno.ENOTDIR)


def _archived_segments(
    settings: Settings,
    sink: _ArchiveWalkSink,
) -> Iterator[Path]:
    """Stream every file under the archive directory, at any depth, recording tree refusals.

    Yields each archived file path as the walk reaches it and NEVER accumulates them into a
    list at any level, so the whole archive inventory -- which retention-off accumulation
    lets grow without bound -- is never held in memory. The pairing and the upload phase pull
    one path at a time, so the sidecar cannot exhaust its allocation before the phase's
    deadline gate runs. Walked rather than globbed at one level because rotation is free to
    nest, and a segment missed here is the older half of a conversation lost at the next task
    replacement.

    The refusals the walk finds -- a directory it could not list, a linked subdirectory it
    dropped, or a refused root -- go into *sink*, bounded there to a capped sample plus a true
    count (see :class:`_ArchiveWalkSink`) rather than into an unbounded list of their own. The
    caller reads the sink AFTER the stream is drained: :func:`run_cycle` extends its result
    from it before the withhold check, so a walk refusal still withholds the authority pair
    and a shape refusal still leaves the pointer free to advance, exactly as when the lists
    were returned. ``refused`` a later cycle could get past; ``unreachable`` it could not --
    the same permanence split :class:`RefusedEntry` states.

    The chain down to the archive directory is opened first with every link refused. It
    has to be, because ``os.walk``'s ``followlinks=False`` governs directories it FINDS
    and not the root it is given: a link planted at ``sessions/archive`` is descended,
    and then every regular file behind it is a file with a key of its own and no reason
    to be in this bucket. When the chain is refused nothing under it is listed, and the
    refusal is recorded so the cycle ends loudly instead of quietly backing up less.
    """
    root = settings.archive_dir
    try:
        fd = _descend(settings.data_home, root.relative_to(settings.data_home).parts)
    except FileNotFoundError:
        return
    except OSError as exc:
        blocked = (
            root.name,
            f"the archive directory, or a directory above it, could not be opened "
            f"as a real directory inside the data home ({exc}); nothing under it is "
            "listed, because a link there points at files this task does not own",
        )
        # A refused ROOT withholds -- which is where it parts from the per-entry split in
        # :class:`RefusedEntry`. That split accepts one planted NAME behind an advancing index
        # because the freeze would cost every other conversation its updates. A root is not
        # one name: the whole subtree goes unenumerated, so an index published over it names
        # conversations whose segments are not in the bucket, and the replacement reads those
        # absent objects as conversations that never had history.
        #
        # But it withholds only for a cause a later cycle could get PAST. The permanence rule
        # is the same one :class:`RefusedEntry` states, and it has to be applied here too: a
        # link or a non-directory at this name is a SHAPE, so every later cycle meets the
        # identical error, the pair is withheld forever, and the index freezes at the moment
        # the name appeared while live transcripts keep uploading past it -- the unbounded
        # freeze that class says must never happen, reached through the guard meant to stop
        # the bounded loss. So a shape refusal is unreachable: the cycle still fails loudly
        # and names the entry, and the pointer is free to advance.
        (sink.cannot_reach if _is_shape_error(exc) else sink.refuse)(blocked)
        return
    # Every error the walk meets is COLLECTED rather than skipped. ``os.fwalk`` swallows an
    # OSError and continues when ``onerror`` is unset, so a directory this uid cannot open
    # contributed no segments, no refusal and no log: the cycle reported itself complete, the
    # pointer advanced over an index naming conversations whose older halves were never
    # uploaded, and the archive lives on an ephemeral disk -- so those segments were gone with
    # no record of which ones. Split on the same permanence rule as the root, into the sink.

    def collect(exc: OSError) -> None:
        where = getattr(exc, "filename", None) or root.name
        entry = (
            str(where),
            f"a directory under the archive could not be listed ({exc}); the segments under "
            "it are not in this cycle's set",
        )
        (sink.cannot_reach if _is_shape_error(exc) else sink.refuse)(entry)

    # Walked THROUGH the descended descriptor rather than from the name a second time, for the
    # reason :func:`_live_transcripts` is: a check on one lookup and a walk on another are two
    # directories the moment the root moves between them, so the refusal the check earns is
    # spent walking the tree it refused. ``fwalk`` starts at the inode the descent validated.
    # The descriptor is held until the walk is EXHAUSTED, across every path this generator
    # yields; the ``finally`` closes it when the generator is fully consumed or closed. The
    # upload phase always drains this iterator to the end -- naming any unreached tail on a
    # deadline or stop -- so the close is reached even when the cycle stops mid-stream.
    try:
        for parent, dirnames, filenames, dir_fd in os.fwalk(
            dir_fd=fd, follow_symlinks=False, onerror=collect
        ):
            dirnames.sort()
            base = root if parent == "." else root / parent
            # A LINKED subdirectory is the one drop the collector above cannot see.
            # ``fwalk`` with ``follow_symlinks=False`` does not descend it -- and does not
            # report it either: it opens the name, compares that descriptor's ``stat``
            # against the name's ``lstat``, and on a mismatch simply drops the entry
            # without calling ``onerror``. So its segments reach neither the stream nor
            # either refusal list, the cycle reports itself COMPLETE, the pointer advances
            # over an index naming conversations whose archived halves were never uploaded,
            # and the archive is on an ephemeral disk -- gone, with no record of which ones.
            # That is the same plant this function already answers loudly one component
            # higher, at the archive root, so going silent one level down is an
            # inconsistency in this defence rather than a case it decided to accept.
            # Named here, and removed from ``dirnames`` so the drop is this function's own
            # rather than a side effect of the walk. Permanent, like the root's shape
            # refusals: a link does not become a directory on the next cycle, so it is
            # UNREACHABLE and the pointer stays free to advance rather than the pair being
            # withheld forever.
            for name in list(dirnames):
                try:
                    linked = stat.S_ISLNK(os.lstat(name, dir_fd=dir_fd).st_mode)
                except OSError as exc:
                    dirnames.remove(name)
                    collect(exc)
                    continue
                if not linked:
                    continue
                dirnames.remove(name)
                sink.cannot_reach(
                    (
                        str(base / name),
                        "a directory under the archive is a symbolic link; nothing under it "
                        "is listed, because a link there names files this task does not own, "
                        "and its segments are not in this cycle's set",
                    )
                )
            for name in sorted(filenames):
                yield base / name
    finally:
        os.close(fd)


@dataclass(frozen=True)
class BackupSet:
    """What one cycle should upload, in two phases, and what it already could not reach.

    The two lists are separate because the authority files are POINTERS: they name the
    transcripts, so they are only true once those transcripts are in the bucket. Holding
    them in their own phase is what lets the cycle publish them last, and withhold them
    entirely when a transcript did not make it.

    The refusals belong here rather than being discovered later because some of them are
    decided while LISTING, not while opening: a linked archive directory means a whole
    subtree is not enumerated, and that has to reach the cycle as a refusal. A set that
    returned only items would report a short cycle as a complete one.

    ``refused`` and ``unreachable`` carry the LIVE and AUTHORITY refusals, complete when the
    set is built. The ARCHIVE refusals live on ``archive_sink`` instead, because the archive
    is streamed: its walk runs as ``data`` is consumed, so its refusals are known only once
    that stream is drained. :func:`run_cycle` reads the sink after the upload phase and
    before its withhold check, folding the archive refusals into the same split. Both are
    split exactly as :class:`RefusedEntry` splits them: the cycle withholds the authority
    pair for a refusal a later cycle can get past, and must not for one it cannot, because
    that withholding would never end.

    ``data`` is a lazy iterable rather than a built list: it lists the live transcripts (one
    flat directory, bounded) and STREAMS the archive tree one path at a time, so the upload
    phase's deadline gate is reached with the archive inventory never held in a list at any
    level and stops the cycle mid-stream instead of after the whole inventory is in memory.
    It is re-iterable -- each iteration re-lists the live paths and re-walks the archive
    afresh, resetting ``archive_sink`` -- so a caller that reads it more than once is safe.
    """

    data: Iterable[tuple[str, Path]]
    authority: list[tuple[str, Snapshot]]
    authority_gone: list[str]
    refused: list[tuple[str, str]]
    unreachable: list[tuple[str, str]]
    archive_sink: _ArchiveWalkSink
    #: The writer-unique id this cycle's pair is published under, which the pointer commits.
    #: Minted per cycle so two concurrent writers address distinct generations.
    generation_id: str

    def close_authority(self) -> None:
        """Release the authority descriptors, uploaded or not.

        The withheld path never uploads them, so closing cannot live at the upload site.
        """
        for _key, snapshot in self.authority:
            snapshot.close()


def objects_to_back_up(settings: Settings, *, generation_id: str | None = None) -> BackupSet:
    """The cycle's two phases: every transcript, then the authority files that name them.

    The authority files are OPENED FIRST, before a single transcript is listed, and
    uploaded from those descriptors at the end of the cycle. Opening is what fixes the
    instant they describe: a descriptor's bounded length is the file as it was at open
    time, so the pair is one coherent snapshot of the index taken BEFORE the enumeration
    it indexes. Reading them at upload time instead let a slot table flushed during the
    cycle name a transcript that cycle never listed -- an index pointing at bytes that
    are not in the bucket.

    They are uploaded last for the same reason they are opened first. An index newer than
    the transcripts it names sends the front to an absent object, and the front reads that
    as a conversation that never had history: a live conversation served empty, with
    nothing raised. An index OLDER than the transcripts is the harmless direction, because
    every slot it names already has its bytes there and a transcript it does not name yet
    is unreferenced rather than misread.

    A missing authority file is not a failure. On a first boot the backend has not written
    one yet, and there is no index to preserve. Such a cycle publishes the file it has and
    no completeness record, so the bucket keeps saying that no whole pair has been
    published -- which is what lets the next task boot instead of refusing.
    """
    refused: list[tuple[str, str]] = []
    unreachable: list[tuple[str, str]] = []
    authority: list[tuple[str, Snapshot]] = []
    gone: list[str] = []
    # A cycle publishes its pair into a WRITER-UNIQUE generation. Minted here when the caller
    # did not supply one, so two writers racing this window each address a distinct
    # ``gen/<id>/`` and neither can overwrite the other's pair -- the pointer's compare-and-swap
    # then settles which single generation is committed.
    if generation_id is None:
        generation_id = keys.new_generation_id()
    for name in keys.AUTHORITY_NAMES:
        path = settings.config_dir / name
        try:
            snapshot = open_snapshot(path, root=settings.data_home)
        except RefusedEntry as exc:
            log.error("backup: refusing %s -- %s", name, exc)
            (unreachable if exc.permanent else refused).append((name, str(exc)))
            continue
        if snapshot is None:
            log.info("backup: %s is not there yet; there is no index to preserve", name)
            gone.append(name)
            continue
        authority.append((keys.authority_generation_key(settings, generation_id, name), snapshot))
    # The open descriptors are OWNED here until :class:`BackupSet` takes them. Their only
    # close site is the ``finally: plan.close_authority()`` in :func:`run_cycle`, and
    # ``run_cycle`` calls this function ABOVE that ``try`` -- so a raise from the live
    # enumerator below escapes with the snapshots still open, and a task that raises once
    # per interval leaks two descriptors a cycle until ``EMFILE`` makes ``open_snapshot``
    # fail for a reason that looks nothing like the cause.
    #
    # The live transcripts are listed EAGERLY here: the sessions directory is one flat
    # level, so its listing is bounded by the live session count and holding it is not the
    # unbounded-inventory hazard. Listing it at plan time is also what the durability
    # contract needs -- a transcript listed now and unlinked before the upload opens it is
    # a deletion (gone), and a directory above it that vanishes is a refusal, and the
    # cycle can only tell those apart because the path was in the set BEFORE the open. The
    # ARCHIVE is the unbounded one -- rotation nests it and retention-off lets it grow -- so
    # it is STREAMED instead (see below), never held as a list.
    try:
        live, live_refused = _live_transcripts(settings)
        refused.extend(live_refused)
    except BaseException:
        for _key, snapshot in authority:
            snapshot.close()
        raise

    # The archive is enumerated LAZILY and its refusals collected into this bounded sink as
    # the walk runs. ``data`` streams the archive one path at a time (below), so the whole
    # archive inventory is never held in a list at any level -- the retention-off growth that
    # would otherwise exhaust the sidecar before the upload phase's deadline gate could bound
    # it. The sink's refusals are complete only once that stream is drained, so unlike the
    # live and authority refusals they are NOT in ``plan.refused``/``plan.unreachable`` at
    # build time: :func:`run_cycle` reads them from ``plan.archive_sink`` after the upload
    # phase and before its withhold check, where the whole-plan refusal split is decided.
    archive_sink = _ArchiveWalkSink()

    # ``data`` pairs each transcript with its key ON DEMAND rather than building the whole
    # list of pairs here. The live transcripts are listed above (a flat directory bounded by
    # its entries); the archive is streamed through :func:`_archived_segments`, which yields
    # each path as its walk reaches it and never accumulates them. Building a key/path tuple
    # for every archived object up front would be a second materialisation of the whole
    # inventory that the phase's deadline gate never gets to bound: the sidecar can exhaust
    # its allocation before a single deadline check runs, and the final cycle loses
    # everything since the prior one. Yielding the pairs lets the gate stop mid-stream.
    #
    # Re-iterable: each iteration re-lists the live paths from the same bounded list and
    # re-walks the archive afresh. Re-walking re-collects the archive refusals, so the sink
    # is RESET at the start of each iteration -- the sink then reflects the most recent walk,
    # which in the one place that reads it (:func:`run_cycle`, after its single consumption
    # via the upload phase) is the only walk. A caller that iterates for the paths alone and
    # ignores the sink is unaffected.
    def _pairs() -> Iterator[tuple[str, Path]]:
        archive_sink.reset()  # a fresh walk re-collects its own refusals
        for path in live:
            yield keys.data_key(settings, path), path
        for path in _archived_segments(settings, archive_sink):
            yield keys.data_key(settings, path), path

    class _DataPairs:
        """A re-iterable view over the cycle's transcript key/path pairs.

        Live paths come from a bounded list; archive paths are streamed afresh each
        iteration, so nothing holds the whole archive inventory.
        """

        __slots__ = ()

        def __iter__(self) -> Iterator[tuple[str, Path]]:
            return _pairs()

    return BackupSet(
        data=_DataPairs(),
        authority=authority,
        authority_gone=gone,
        refused=refused,
        unreachable=unreachable,
        archive_sink=archive_sink,
        generation_id=generation_id,
    )


def run_cycle(
    settings: Settings,
    store: ObjectStore,
    *,
    state: dict[str, Fingerprint],
    deadline: float | None = None,
    yield_when: Callable[[], bool] | None = None,
) -> CycleResult:
    """Upload everything in the set that has changed. Raise if anything was refused.

    *state* is read and written in place, so the caller keeps one map across cycles and
    an object unchanged since its last successful upload is not sent again.

    The authority phase runs only when the transcript phase reached everything it was
    asked for. A cycle that could not commit one transcript leaves the authority files
    as the last complete cycle wrote them, which is an older but coherent pair, rather
    than advancing the index past the bytes.

    A failure between the two authority PUTs leaves the bucket holding one file from this
    cycle's snapshot and one from an earlier cycle's. That skew is in the harmless
    direction -- both were opened before this cycle's enumeration, so neither names a
    transcript that is not in the bucket -- and it is not silent: the failure is a refusal,
    the cycle raises on it, and the next cycle publishes the pair together. What it costs
    is one interval in which the two files disagree about which slots exist.

    *deadline* is a ``time.monotonic`` reading after which no further object is attempted.
    The final cycle passes one, because it runs inside a drain window and uploads
    sequentially: without a bound the window elapses mid-PUT and the process is SIGKILLed,
    which loses the object in flight and says nothing about the ones behind it. With one,
    every object the cycle could not reach is recorded as a refusal by name, the cycle is
    incomplete, and the process exits non-zero on a report an operator can act on. The
    ordinary interval cycles pass none: they have a next interval.

    The transcript phase stops EARLY enough to leave the index its own room. Both phases
    are bounded by the same deadline, but a data phase allowed to spend all of it would
    reach the end with nothing left for the authority pair, and the authority PUTs would
    then run past the window and be killed mid-request -- publishing one file and not the
    other, which is the torn index the two-phase order exists to avoid.
    """
    result = CycleResult()
    publish = True
    try:
        committed = generation.read_pointer(settings, store)
    except generation.PointerUnusable as exc:
        log.error(
            "backup: %s The authority pair is NOT published this cycle, because committing a "
            "new generation without knowing which one is currently committed can overwrite "
            "the generation a replacement would boot from. Transcripts still upload.",
            exc,
        )
        committed = None
        publish = False
    plan = objects_to_back_up(settings)
    for _name, _reason in plan.refused:
        result.record_refused(_name, _reason)
    for _name, _reason in plan.unreachable:
        result.record_unreachable(_name, _reason)
    result.gone.extend(plan.authority_gone)
    if plan.authority and not _record_is_due(plan):
        # SOME of the pair, not all of it, which is not the same as none of it. With no whole
        # pair there is no generation to commit, so the pointer stays where it is -- and a
        # restore that finds no pointer at all reads the legacy keys, which this writer never
        # writes. Uploading the one file it has would therefore put it in a generation
        # nothing can reach, while the cycle reported itself complete and exited zero: an
        # index silently absent rather than an index preserved. The two files have
        # independent writers, so a one-file window is ordinary timing skew and not an
        # extreme state -- which is exactly why it must not be the quiet path.
        #
        # None of the pair stays a non-failure, as above: on a first boot there is no index
        # to preserve and nothing to say. Some of it is refused, which withholds the file
        # that IS there and makes the cycle incomplete, so the next cycle -- once both
        # writers have flushed -- publishes a whole pair into a generation a reader can
        # reach, and the exit code names the wait instead of hiding it.
        absent = [name for name in keys.AUTHORITY_NAMES if name in set(plan.authority_gone)]
        missing = absent or ["an authority file"]
        for name in missing:
            result.record_refused(
                name,
                "the authority pair is incomplete this cycle, so there is no generation to "
                "publish the rest of it into",
            )
    if not publish:
        for key, _snapshot in plan.authority:
            result.withheld.append(key)
        if deadline is not None:
            # The final cycle. Withholding alone would exit zero and report a lossless
            # stop, while the pointer still names the older generation and the pair this
            # drain flush produced is never published -- so the replacement boots an index
            # without the conversations served since the last interval publish, whose
            # transcripts are in the bucket and unreferenced. An interval cycle keeps the
            # plain withholding above, because its next cycle re-reads the pointer.
            for key, _snapshot in plan.authority:
                result.record_refused(
                    key, "the committed generation could not be read, so the pair is unpublished"
                )
    try:
        _upload_phase(
            plan.data,
            settings=settings,
            store=store,
            state=state,
            result=result,
            deadline=_reserve_for_authority(
                deadline,
                # Reserve for the authority phase only when it will RUN. When the pair is
                # withheld this cycle (the block above records every authority key as
                # withheld or refused and publishes none), reserving a PUT-worth of window
                # per authority object would carve time off the data phase for uploads that
                # never happen -- shrinking what the transcripts get for no gain. Derive the
                # count from the phase that will run: zero when it is withheld, else the pair
                # plus the pointer when a record is due.
                len(plan.authority) + (1 if _record_is_due(plan) else 0) if publish else 0,
            ),
            yield_when=yield_when,
        )
        # The archive is streamed, so its refusals are known only now that the upload phase
        # has drained the stream (or drained its tail to name what it did not reach). Fold
        # them into the result HERE -- after the phase, before the withhold check below --
        # so a directory the walk could not list or a linked subtree it dropped withholds
        # the pair, and a shape refusal leaves the pointer free to advance. The sink is
        # bounded (a capped sample plus a true count), and the checks below key on whether
        # there were ANY refusals, not their identity, so the bound changes no decision.
        # Folded so the result's own COUNTS take the sink's true counts (not the sample
        # length) while the sample entries fill the result's sample only up to its cap.
        result.fold_archive_refusals(plan.archive_sink)
        # The lifetime *state* map is kept from growing with the archive AT INSERTION time now
        # (see :func:`_record_durable`), not swept here: a single cycle can enumerate an
        # arbitrarily large archive, so trimming only after the phase would let the map hold
        # that whole cycle's archive keys transiently -- the peak the bound must forbid. The
        # per-insertion eviction holds the archive population at the cap at every moment.
        # Keyed on ``refused`` ALONE, never on the cycle being incomplete. A refusal means
        # an object the pair CAN name did not reach the bucket, so publishing the pair
        # would send the front to bytes that are not there. An unreachable entry is a name
        # the backend never wrote, so the pair does not reference it -- and because a shape
        # refusal stays put, withholding on one would withhold the pair on every later
        # cycle too: the index frozen permanently while transcripts keep uploading past
        # it, and a replacement task restoring the pair from before the entry appeared.
        if result.unreachable_count and not result.refused_count:
            log.warning(
                "backup: %d entries in the backup set could not be reached, so this cycle "
                "is incomplete -- the authority pair IS still published, because none of "
                "them is an object the pair can name: %s",
                result.unreachable_count,
                ", ".join(name for name, _why in result.unreachable),
            )
        _referenced_by_captured_index(plan, result, settings=settings)
        if result.refused_count or result.gone_referenced_count:
            for key, _snapshot in plan.authority:
                if key not in result.withheld:
                    result.withheld.append(key)
            log.error(
                "backup: %d object(s) refused and %d vanished while the captured index "
                "still named them, so the authority files are NOT published this cycle -- "
                "the pair in the bucket stays at the last complete cycle rather than "
                "naming transcripts that are not there: %s",
                result.refused_count,
                result.gone_referenced_count,
                ", ".join(result.gone_referenced) or "-",
            )
        elif publish:
            if _pair_unchanged(plan, settings=settings, state=state, committed=committed):
                for key, _snapshot in plan.authority:
                    result.record_unchanged(
                        keys.authority_generation_key(
                            settings,
                            committed.generation,  # type: ignore[union-attr]
                            key.rsplit("/", 1)[-1],
                        )
                    )
            else:
                _commit_authority(
                    plan.authority,
                    settings=settings,
                    store=store,
                    state=state,
                    result=result,
                    deadline=deadline,
                    yield_when=yield_when,
                )
                _commit_generation(
                    plan,
                    settings=settings,
                    store=store,
                    state=state,
                    result=result,
                    generation_id=plan.generation_id,
                    committed=committed,
                    deadline=deadline,
                )
    finally:
        plan.close_authority()
    log.info("backup: cycle complete -- %s", result.summary())
    if not result.complete:
        raise BackupIncomplete(result)
    return result


def _record_is_due(plan: BackupSet) -> bool:
    """Whether this cycle can commit a generation at all.

    True only when the plan holds EVERY authority file. A cycle that found one of them
    missing locally has no whole pair to publish, and a generation containing one file
    would be a committed generation the restore boots from while the backend flushes its
    own empty view of the other. Such a cycle leaves the pointer alone, so the bucket
    keeps saying that the last committed generation is the one before it.
    """
    published = {key.rsplit("/", 1)[-1] for key, _snapshot in plan.authority}
    return published == set(keys.AUTHORITY_NAMES)


def _pair_unchanged(
    plan: BackupSet,
    *,
    settings: Settings,
    state: dict[str, Fingerprint],
    committed: generation.Pointer | None,
) -> bool:
    """Whether the committed generation already holds exactly this cycle's pair.

    Without this the protocol republishes on every interval: each cycle mints a fresh
    generation id, so the pair's keys differ from the ones last uploaded and every cycle
    looks like a change. The comparison is therefore made against the COMMITTED generation's
    keys, which is where the last published bytes actually went, via the fingerprints this
    process recorded for them.

    False whenever anything is unknown -- no pointer, a name the commitment does not
    cover, a fingerprint this process never recorded -- because republishing a pair that
    was already there costs one cycle's bandwidth, while skipping one that was not costs
    the index.
    """
    if committed is None or not _record_is_due(plan):
        return False
    for key, snapshot in plan.authority:
        name = key.rsplit("/", 1)[-1]
        if name not in committed.authority:
            return False
        if (
            state.get(keys.authority_generation_key(settings, committed.generation, name))
            != snapshot.fingerprint
        ):
            return False
    return True


def _slots_named_by(authority: list[tuple[str, Snapshot]]) -> set[str] | None:
    """The slot ids the CAPTURED authority pair names, or ``None`` if that cannot be read.

    Read with :func:`os.pread` off the snapshot's own descriptor, so the offset the upload
    reads from is not moved and the bytes are the ones that will be published -- asking the
    path again would be a second resolution of one name with a window in between, which is
    the thing every other read here avoids.

    ``session_map.json`` names a conversation by its KEY, and ``open_slots.json`` by a
    member of its ``keys`` list. Both shapes are the ones the restore side validates and the
    backend's own loaders accept; anything else in the file is ignored here, because this
    answers only "could the published pair send a reader to this name".

    ``None`` means the question could not be answered -- bytes that do not decode or do not
    parse. The caller must treat that as "it might name anything", never as "it names
    nothing": an unreadable index is exactly when a wrong guess is least recoverable.
    """
    named: set[str] = set()
    for key, snapshot in authority:
        name = key.rsplit("/", 1)[-1]
        try:
            raw = os.pread(snapshot.fh.fileno(), snapshot.size, 0)
            parsed = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - answered as "unknown", never as "empty"
            log.error(
                "backup: the captured %s could not be read to see which conversations it "
                "names (%s), so this cycle cannot tell whether a vanished transcript is "
                "one of them",
                name,
                exc,
            )
            return None
        if not isinstance(parsed, dict):
            log.error(
                "backup: the captured %s is a JSON %s rather than an object, so which "
                "conversations it names cannot be established",
                name,
                type(parsed).__name__,
            )
            return None
        if name == "open_slots.json":
            listed = parsed.get("keys")
            if isinstance(listed, list):
                named.update(member for member in listed if isinstance(member, str))
        else:
            named.update(str(entry) for entry in parsed)
    return named


def _referenced_by_captured_index(
    plan: BackupSet, result: CycleResult, *, settings: Settings
) -> None:
    """Record which vanished-and-undurable entries the captured pair would send a reader to.

    Records directly into *result* through :meth:`CycleResult.record_gone_referenced`, so the
    verdict is a capped sample plus an exact ``gone_referenced_count`` -- the count is what the
    withhold decision reads, and it stays exact where the sample is truncated.

    An unreadable index answers with EVERY candidate rather than none: the pair is withheld,
    the cycle is incomplete, and the next cycle republishes -- where guessing "names nothing"
    would commit an index this cycle could not read against bytes it knows are absent. Because
    ``gone_undurable`` is itself a capped sample, the count of that candidate set (exact) is
    what drives the unreadable-index verdict, not the retained names: every undurable candidate
    is referenced, so the exact count carries the decision even past the sample.
    """
    if result.gone_undurable_count == 0:
        return
    named = _slots_named_by(plan.authority)
    if named is None:
        # Unreadable index: every undurable candidate is referenced. Record the sample we have
        # for the log, then set the count to the exact candidate total so the withhold decision
        # is not fooled by a truncated sample.
        for name, _key in result.gone_undurable:
            result.record_gone_referenced(name)
        result.gone_referenced_count = result.gone_undurable_count
        return
    # The two sides live in DIFFERENT namespaces, and comparing them directly is how this
    # check silently matched nothing: the index names a conversation by its SLOT KEY
    # (``cust-8831``) while its transcript is ``dashboard_cust-8831.jsonl``. So the slot keys
    # are mapped FORWARD through the front's own ``transcript_stem`` -- the function that
    # decides the real filename -- rather than the prefix being stripped off the filename
    # here. Stripping by hand would also miss the character substitution that function does,
    # so a key holding an unsafe character would map to a name this comparison never made.
    named_files = {
        f"{stem}{keys.TRANSCRIPT_SUFFIX}"
        for stem in (transcript_stem(slot) for slot in named)
        if stem
    }
    for name, _key in result.gone_undurable:
        if name in named_files:
            result.record_gone_referenced(name)


def _next_epoch(committed: "generation.Pointer | None") -> int:
    """The monotonic writer epoch a commit succeeding *committed* must publish.

    One greater than the committed pointer's, or 0 when nothing is committed yet. The value
    comes from the committed pointer alone -- no clock, no writer identity -- so it advances
    by construction and cannot inherit a backward wall-clock step in the timestamp embedded in
    a generation id. Its own function so the fence in :func:`_commit_generation` reads exactly
    one policy, and so a test can force a non-advancing successor to exercise that fence
    (which the ``+1`` here can never trip in a real run, Python ints not wrapping).
    """
    return 0 if committed is None else committed.epoch + 1


def _commit_generation(
    plan: BackupSet,
    *,
    settings: Settings,
    store: ObjectStore,
    state: dict[str, Fingerprint],
    result: CycleResult,
    generation_id: str,
    committed: "generation.Pointer | None" = None,
    deadline: float | None = None,
) -> None:
    """Publish the pointer naming *generation_id* as the committed generation. The LAST step.

    Last is the property the restore depends on, not a preference. A cycle interrupted
    anywhere in the authority phase therefore leaves the pointer naming the PREVIOUS
    generation, whose objects are all still there and were never rewritten -- so the
    replacement boots from a coherent older pair instead of a torn newer one. Committing
    first would invert that: the pointer would name a generation the crash never finished
    writing, and the restore would refuse a bucket whose previous generation was fine.

    The commit is fenced two ways, and both are required. The pointer PUT is a COMPARE-AND-
    SWAP on the ETag this cycle read, which rejects a commit whose read went stale. On top of
    that the pointer carries a MONOTONIC WRITER EPOCH: this commit publishes ``committed.epoch
    + 1`` and refuses to publish an epoch that does not strictly exceed the committed one, so
    a commit that would name OLDER work than what is committed cannot roll the durable state
    backward even in a race the CAS alone would admit.

    Skipped, not failed, when the pair was not whole or anything in this cycle was
    refused: a generation is committed only when it contains one.

    Sent once per published generation through the same *state* map the objects use, so an
    idle cycle that re-published nothing does not re-PUT the pointer. The fingerprint is
    derived from the pointer's own bytes, because it is the one object with no file behind
    it, and it is written only after the PUT returns, so a failed commit is retried.

    A pointer that cannot be written is recorded in ``withheld`` on an interval cycle, and
    that cycle stays complete: the previous generation is still committed and still whole,
    so nothing is lost -- this cycle's newer pair simply is not adopted yet, and the next
    cycle commits it.

    On the FINAL cycle the same failure is a refusal instead, because the recovery above
    is a later cycle and the final cycle has none. Left withheld it would exit zero and
    report a lossless stop while the replacement adopts the generation the pointer still
    names -- the older index, without the conversations this cycle wrote. ``deadline`` is
    what tells the two apart: only the final cycle sets one.
    """
    if not _record_is_due(plan) or result.refused or result.withheld:
        return
    key = keys.authority_pointer_key(settings)
    # The MONOTONIC WRITER EPOCH this commit publishes: one greater than the committed
    # pointer's, or 0 for the first commit. It is derived from the value THIS CYCLE READ, not
    # from any clock or writer identity, so it advances by construction and cannot inherit a
    # backward wall-clock step in the timestamp in the generation id.
    new_epoch = _next_epoch(committed)
    # FENCE publication by recency. The compare-and-swap below already rejects a commit whose
    # read of the pointer is stale, but a CAS orders writes by the pointer's ETag alone: two
    # overlapping replacement writers can each hold a valid ``If-Match`` and whichever PUT
    # lands second wins even when it names OLDER work, rolling the committed generation
    # backward. Refusing an epoch that does not strictly exceed the committed one closes
    # that: the successor is always ``committed.epoch + 1``, so the only way this guard trips
    # is a committed epoch at the integer ceiling, which cannot occur in any real run -- but
    # the check is kept so the invariant is enforced here rather than merely assumed, and so
    # a future path that constructs an epoch some other way cannot publish a non-advancing
    # one. The loser of a race re-reads the now-current pointer next cycle and rebuilds on it.
    if committed is not None and new_epoch <= committed.epoch:
        log.error(
            "backup: the monotonic writer epoch would not advance past the committed epoch "
            "%d, so generation %s is NOT committed and this cycle is refused rather than "
            "publishing work that does not strictly supersede what the pointer names.",
            committed.epoch,
            generation_id,
        )
        result.record_refused(key, "the writer epoch would not advance; commit refused")
        return
    body = generation.pointer_body(generation_id, new_epoch)
    # The fingerprint distinguishes one committed generation from the next so a pointer
    # naming a NEW generation (or a new epoch) is re-PUT rather than skipped as unchanged. The
    # pointer is the one object with no file behind it, so the fingerprint is derived from its
    # own bytes: ``mtime_ns`` carries a stable digest of the whole body -- generation id AND
    # epoch -- which differs whenever either does, and every body is the same length so
    # ``size`` alone could not tell two commitments apart.
    fingerprint = Fingerprint(
        inode=0,
        size=len(body),
        mtime_ns=int.from_bytes(hashlib.sha256(body).digest()[:8], "big"),
    )
    if state.get(key) == fingerprint:
        result.record_unchanged(key)
        return
    final = deadline is not None
    if deadline is not None and not _time_for_one_more(deadline, BACKUP_ATTEMPT_COST_SECS):
        log.error(
            "backup: the drain window cannot fit the generation pointer, so generation %s "
            "is NOT committed and this cycle is refused. The generation the pointer still "
            "names is whole, but it is the older one and no later cycle follows this.",
            generation_id,
        )
        result.record_refused(key, "the drain window could not fit the generation pointer")
        return
    # The commit is a COMPARE-AND-SWAP against the pointer's state AS THIS CYCLE READ IT, so a
    # second sidecar writing this prefix in the task-replacement window cannot clobber the
    # generation this one publishes. The validator is the ETag captured by ``read_pointer`` at
    # the START of the cycle -- NOT a fresh HEAD here -- for two reasons the review named: a
    # HEAD at commit time is unbudgeted work the final cycle's drain window could be SIGKILLed
    # inside, and reading the validator at cycle start widens the CAS window to cover the
    # authority writes this commit blesses, not just the pointer PUT. An existing pointer commits
    # ``If-Match`` its captured ETag; no pointer yet commits ``If-None-Match: *``. A writer
    # that advanced the pointer since this read changes the ETag, so this PUT is rejected
    # rather than overwriting -- the stale commit steps aside.
    #
    # FAIL CLOSED on a missing validator: a pointer that was present but whose ETag the store
    # could not supply has NO precondition available, and committing unconditionally would
    # defeat the guard against a concurrent writer. So the commit is refused rather than run
    # blind -- the generation the pointer still names is whole, and a later cycle re-reads a
    # validator and commits.
    if_match: str | None = None
    if_none_match: str | None = None
    if committed is None:
        if_none_match = "*"
    elif committed.etag is not None:
        if_match = committed.etag
    else:
        log.error(
            "backup: the committed generation pointer carried no ETag validator, so "
            "generation %s cannot be committed under a compare-and-swap and this cycle is "
            "refused rather than overwriting a concurrent writer's generation blind.",
            generation_id,
        )
        result.record_refused(key, "the generation pointer had no CAS validator; commit refused")
        return
    try:
        # Neither bounded nor cancellable, and both for the same reason: the body is a
        # few hundred bytes the transport reads in ONE call, so a predicate asked between
        # chunks has no second chunk to refuse and a deadline enforced by the body is
        # never re-consulted. What bounds this request is the client's own
        # ``connect_timeout`` and ``read_timeout`` at one attempt, which is a real bound
        # precisely because there is no transmission here to outgrow them.
        store.put(key, io.BytesIO(body), len(body), if_match=if_match, if_none_match=if_none_match)
    except StoreUnusable:
        raise
    except PreconditionFailed:
        # Another writer committed a newer generation since this cycle read the pointer, so
        # this commit is stale and is rejected rather than overwriting it. The cached belief
        # is dropped so a later cycle re-reads the now-current pointer and rebuilds on it, and
        # this cycle is recorded incomplete: it did not publish the pair it opened, and the
        # pointer names another writer's generation, not this one's.
        state.pop(key, None)
        log.warning(
            "backup: the generation pointer moved under this cycle before it could commit "
            "generation %s, so a concurrent writer's generation stands and this commit is "
            "rejected rather than clobbering it.",
            generation_id,
        )
        result.record_refused(
            key,
            "a concurrent writer committed a newer generation; this stale commit was rejected",
        )
        return
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        # The PUT's response was lost, so whether the pointer landed is UNKNOWN -- it may
        # hold the new generation, the old one, or nothing coherent. Any cached belief that
        # this key already holds *fingerprint* is therefore untrustworthy: left in place
        # it would make the early ``state.get(key) == fingerprint`` return above skip the
        # re-PUT on every later cycle and report the unpublished commit as an unchanged,
        # complete stop -- the generation frozen behind a belief a failed write installed.
        # Dropping it forces the next cycle to attempt the commit again.
        state.pop(key, None)
        if final:
            log.error(
                "backup: the generation pointer could not be published (%s), so generation "
                "%s is not committed and this cycle is refused. No later cycle follows this "
                "one, so the replacement would silently adopt the older generation.",
                exc,
                generation_id,
            )
            result.record_refused(key, f"the generation pointer could not be published ({exc})")
            return
        log.error(
            "backup: the generation pointer could not be published (%s), so generation %s "
            "is not committed. The generation it still names is whole, so nothing is lost; "
            "this cycle's pair is simply not adopted yet.",
            exc,
            generation_id,
        )
        result.withheld.append(key)
        return
    state[key] = fingerprint
    result.record_uploaded(key)


def _commit_authority(
    items: list[tuple[str, Snapshot]],
    *,
    settings: Settings,
    store: ObjectStore,
    state: dict[str, Fingerprint],
    result: CycleResult,
    deadline: float | None = None,
    yield_when: Callable[[], bool] | None = None,
) -> None:
    """Upload the already-open authority snapshots, recording each verdict in *result*.

    Separate from the transcript phase because these descriptors are opened by the plan,
    before the enumeration, and are closed by the caller whether or not this runs. Bounded
    by the same deadline: an index PUT that cannot finish inside the window would be killed
    mid-request, and publishing one of the pair without the other is the torn index the
    phase order exists to prevent.

    *yield_when* bounds this phase on an interval cycle the way the deadline bounds it on
    the final one. These uploads are as capable of outliving a stop as a transcript's, and
    a refusal here keeps the pointer where it is -- :func:`_commit_generation` returns on
    any refusal -- so a cut index leaves the previous generation committed and whole rather
    than half-replaced.
    """
    for index, (key, snapshot) in enumerate(items):
        if deadline is not None and not _time_for_one_more(deadline, BACKUP_ATTEMPT_COST_SECS):
            unreached = [k.rsplit("/", 1)[-1] for k, _s in items[index:]]
            log.error(
                "backup: the drain window cannot fit another upload, so %d authority "
                "file(s) are NOT published: %s. The pair in the bucket stays at the last "
                "complete cycle rather than being left half new.",
                len(unreached),
                ", ".join(unreached),
            )
            result.withheld.extend(k for k, _s in items[index:])
            for name in unreached:
                result.record_refused(
                    name, "not attempted: the drain window could not fit another upload"
                )
            return
        _commit_one(
            key,
            snapshot,
            name=key.rsplit("/", 1)[-1],
            settings=settings,
            store=store,
            state=state,
            result=result,
            # This phase's gate reserves one attempt per index object, so that is what
            # its PUT is bounded by -- the reservation and the bound are one number here
            # too, and an index PUT cannot eat the window the rest of the pair needs.
            budget=None if deadline is None else BACKUP_ATTEMPT_COST_SECS,
            cancel=yield_when,
        )


def _time_for_one_more(deadline: float, budget: float) -> bool:
    """Whether one more upload of *budget* seconds still fits before *deadline*.

    Measured against a budget rather than against any remaining time at all: starting a
    PUT with two seconds left buys nothing, because the kill lands mid-request and the
    object is lost anyway while the ones behind it go unmentioned.

    The two phases ask for different budgets, and the difference is the point. A transcript
    asks for a whole PUT including its retries, because it is the thing the window is for.
    An authority file asks for one attempt, because it is uploading inside a slice the data
    phase already set aside for it -- if the index needs retries the cycle is failing
    anyway, and this check stops it before it runs past the window rather than after.
    """
    return deadline - time.monotonic() >= budget


def _reserve_for_authority(deadline: float | None, count: int) -> float | None:
    """Pull *deadline* in by what the index needs, so the data phase leaves it room.

    Without this the transcripts can spend the whole window and the authority PUTs start
    with nothing left: they run past it and are killed mid-request, which publishes one
    file and not the other. That torn pair is the state the two-phase order exists to
    avoid, so the reservation is part of the ordering rather than a tuning choice.

    One attempt per file, not a whole retry budget per file. The budgets are what the drain
    window has to cover, and reserving the worst case for two small JSON files and a pointer
    would consume most of ``SIDECAR_DRAIN_SECS`` before a single transcript moved. The index phase's
    own deadline check is what covers a retry eating into the rest.
    """
    if deadline is None:
        return None
    return deadline - count * BACKUP_ATTEMPT_COST_SECS


def _record_durable(
    settings: Settings, state: dict[str, "Fingerprint"], key: str, fingerprint: "Fingerprint"
) -> None:
    """Record *key* as durable in *state*, capping the ARCHIVE population AT THIS INSERTION.

    The lifetime *state* map exists so an unchanged object is not re-uploaded. Live
    transcripts and the authority pair are a bounded, flat set the map is meant to hold
    whole, and they are never counted against the cap or evicted. The ARCHIVE is different:
    rotation nests it and retention-off never prunes it, so without a bound one entry per
    archived segment -- and each entry's KEY STRING, an unbounded-length identifier -- would
    grow the map for the process's whole life, the very unbounded retention the streamed
    enumeration was built to stop, reappearing in *state*.

    The cap is enforced HERE, as each archive key is inserted, not once at the end of a
    cycle. A single cycle can enumerate an arbitrarily large archive, so a post-cycle sweep
    would let the map hold that whole cycle's archive keys transiently before trimming them
    -- the peak the bound is supposed to forbid. Evicting the oldest archive key on the
    insertion that would exceed the cap holds the archive population at
    :data:`_ARCHIVE_STATE_CAP` at EVERY moment, so neither the count nor the retained key
    identity ever runs past it, mid-cycle or after.

    The archive-key ORDER is tracked in a companion FIFO on :class:`DurableState` so the
    oldest is found and the population sized in O(1), not by rescanning the whole map on
    every insertion (which would be O(n) per insert and O(n^2) across a large archive walk --
    the very cost the bound exists to avoid). A plain ``dict`` handed in (a unit test) has no
    companion, so it falls back to a scan, correct but unbounded in cost; the process uses
    :class:`DurableState`. An archived segment is immutable, so a dropped entry costs one
    re-upload of identical bytes the next time it is enumerated and nothing else.
    """
    already_present = key in state
    state[key] = fingerprint
    # A non-archive key -- a live transcript, an authority file -- is held whole and never
    # triggers an eviction; re-recording a key already present changes no archive count.
    if already_present or not keys.is_archive_key(settings, key):
        return
    order = getattr(state, "archive_order", None)
    if order is not None:
        # O(1) path: DurableState tracks archive-key insertion order and count in a FIFO.
        order.append(key)
        if len(order) > _ARCHIVE_STATE_CAP:
            oldest = order.popleft()
            state.pop(oldest, None)
        return
    # Fallback for a plain dict (unit tests only): rescan. Correct, not fast.
    archive_count = sum(1 for k in state if keys.is_archive_key(settings, k))
    if archive_count <= _ARCHIVE_STATE_CAP:
        return
    for candidate in state:  # insertion order: the first archive key is the oldest
        if keys.is_archive_key(settings, candidate):
            del state[candidate]
            break


def _name_unreached_remainder(
    current: Path, rest: Iterator[tuple[str, Path]], reason: str
) -> list[tuple[str, str]]:
    """Name the not-attempted *current* object plus a BOUNDED sample of what follows it.

    Called when a stop or a deadline gate fires mid-stream. *rest* is the tail of a lazy
    iterator whose end is the archive tree, and retention-off lets that tree grow without
    bound. Draining it to name every unreached object would rebuild the whole inventory in a
    list at the one moment the cycle is trying to STOP -- the same unbounded-retention hazard
    the streamed enumeration removes from the walk. So this pulls at most
    ``_UNREACHED_SAMPLE_CAP`` names and then stops, recording one count-free remainder marker
    when the tail runs past the cap rather than walking it to its end.

    Every entry returned is a refusal, so the cycle is incomplete whether the remainder was
    named whole or summarised -- the current object alone already carries that. A small
    remainder (every real cycle's) fits under the cap and is named in full; only a
    pathological tree is summarised, which is exactly the case that must not be materialised.
    """
    named: list[tuple[str, str]] = [(current.name, reason)]
    for _key, path in rest:
        if len(named) >= _UNREACHED_SAMPLE_CAP:
            # Do not touch *rest* again: advancing it once more is one step further into the
            # tail this cap exists to leave unwalked. The marker stands for "and the rest",
            # count-free because a count is a drain.
            named.append((_UNREACHED_REMAINDER_NAME, reason))
            break
        named.append((path.name, reason))
    return named


def _upload_phase(
    items: Iterable[tuple[str, Path]],
    *,
    settings: Settings,
    store: ObjectStore,
    state: dict[str, Fingerprint],
    result: CycleResult,
    deadline: float | None = None,
    yield_when: Callable[[], bool] | None = None,
) -> None:
    """Upload one phase of the set, recording every entry's verdict in *result*.

    Stops attempting objects once there is not enough of *deadline* left for one PUT's
    whole retry budget, and records every object from there on as refused BY NAME. Trying
    one more and being killed in the middle of it would lose that object and leave the rest
    unmentioned; stopping first costs the same objects and says which they are.

    *yield_when* lets an ordinary interval cycle stand down the moment a stop arrives. Its
    uploads predate the backend's flush, so everything it has left is something the FINAL
    cycle will send anyway -- continuing only spends the drain window that cycle needs.
    """
    # ``items`` is a lazy iterable (see :class:`BackupSet`): the pairs are produced on
    # demand, so this gate is reached before the whole inventory is built and stops the
    # cycle mid-stream rather than after it is all in memory. On a stop or a deadline the
    # current pair is not attempted, so it and the REST OF THE ITERATOR are the unreached
    # remainder -- named by :func:`_name_unreached_remainder`, which records the current
    # object and a BOUNDED sample of what follows rather than draining the tail (the archive
    # tree, unbounded under retention-off) back into a list at the moment the cycle is
    # stopping.
    it = iter(items)
    for key, path in it:
        if yield_when is not None and yield_when():
            unreached = _name_unreached_remainder(
                path,
                it,
                "not attempted: the stop arrived and the final cycle takes these",
            )
            log.info(
                "backup: the stop arrived mid-cycle, so this object and the objects after it "
                "are left to the final cycle rather than spending its drain window here; "
                "naming up to %d of them: %s",
                _UNREACHED_SAMPLE_CAP,
                ", ".join(name for name, _why in unreached),
            )
            for _name, _reason in unreached:
                result.record_refused(_name, _reason)
            return
        try:
            snapshot = open_snapshot(path, root=settings.data_home)
        except RefusedEntry as exc:
            # Split on the refusal's own permanence, not on the phase it surfaced in: a
            # missing directory here may be a data home that comes back, while a FIFO at
            # this name will still be a FIFO next cycle. Only the first is worth holding
            # the index back for; see :class:`RefusedEntry`.
            log.error("backup: refusing %s -- %s", path.name, exc)
            record = result.record_unreachable if exc.permanent else result.record_refused
            record(path.name, str(exc))
            continue
        if snapshot is None:
            # Not a verdict yet. Whether this one matters depends on the CAPTURED index,
            # which the authority phase holds, so this records the candidate and the
            # authority phase decides. Durability is the half that can be settled here:
            # bytes already in the bucket mean an older index naming it still resolves.
            log.info("backup: %s is gone; nothing to upload for it", path.name)
            if key not in state:
                result.record_gone_undurable(path.name, key)
            result.record_gone(path.name)
            continue
        try:
            # BEFORE the deadline gate, because an object the bucket already holds needs no
            # PUT and the gate exists to stop PUTs. Checked after the gate, a stop arriving
            # late in an interval cycle refuses every remaining object without opening one,
            # and a refusal withholds the authority pair -- so a drain with nothing to
            # upload would report itself lossy and exit non-zero while the index phase
            # still had most of the window.
            if _already_durable(key, snapshot, state):
                result.record_unchanged(key)
                continue
            if deadline is not None and not _time_for_one_more(
                deadline, BACKUP_PER_OBJECT_BUDGET_SECS
            ):
                # This pair is not attempted, so it and the rest of the iterator are the
                # unreached remainder. :func:`_name_unreached_remainder` records this object
                # and a bounded sample of what follows rather than draining the tail -- the
                # archive tree is unbounded under retention-off, and rebuilding it into a list
                # to name it is the hazard the streamed enumeration removes from the walk.
                unreached = _name_unreached_remainder(
                    path,
                    it,
                    "not attempted: the drain window could not fit another upload",
                )
                log.error(
                    "backup: the drain window has %.1fs left, less than the %.0fs one "
                    "upload can take, so this object and the objects after it are NOT "
                    "attempted; naming up to %d of them: %s",
                    max(0.0, deadline - time.monotonic()),
                    BACKUP_PER_OBJECT_BUDGET_SECS,
                    _UNREACHED_SAMPLE_CAP,
                    ", ".join(name for name, _why in unreached),
                )
                for _name, _reason in unreached:
                    result.record_refused(_name, _reason)
                return
            _commit_one(
                key,
                snapshot,
                name=path.name,
                settings=settings,
                store=store,
                state=state,
                result=result,
                # The same number the gate just reserved, so the PUT cannot outlive what
                # was set aside for it. None on an interval cycle: there is no window to
                # protect and a next cycle to finish the object, so cutting a slow upload
                # there would abandon one that was on its way.
                budget=None if deadline is None else BACKUP_PER_OBJECT_BUDGET_SECS,
                # The stop reaches INSIDE the upload, not just between them. The check at
                # the top of this loop only runs between objects, so a stop landing during
                # a PUT that has no budget is not observed until that PUT ends on its own
                # -- and this cycle must return before the final one may begin, whose
                # deadline is measured from when the stop was observed rather than from
                # then. Every second spent finishing this object is a second taken from
                # uploading the turns the backend flushed on its way out.
                cancel=yield_when,
            )
        finally:
            snapshot.close()


def _already_durable(key: str, snapshot: Snapshot, state: dict[str, Fingerprint]) -> bool:
    """Whether the bucket already holds exactly these bytes under *key*.

    One definition, because two callers ask it for different reasons and must not disagree:
    :func:`_commit_one` asks so it does not re-send an object, and :func:`_upload_phase` asks
    BEFORE its deadline gate so an object needing no PUT is never recorded as refused.
    """
    return state.get(key) == snapshot.fingerprint


def _commit_one(
    key: str,
    snapshot: Snapshot,
    *,
    name: str,
    settings: Settings,
    store: ObjectStore,
    state: dict[str, Fingerprint],
    result: CycleResult,
    budget: float | None = None,
    cancel: Callable[[], bool] | None = None,
) -> None:
    """Send ONE open snapshot, or record why it was not sent. Never closes the descriptor.

    The caller owns the descriptor, because the two phases acquire it at different times:
    the transcript phase opens one per object as it goes, and the authority pair was opened
    by the plan before anything was enumerated.

    *budget* is the seconds the caller's deadline gate set aside for this object, handed to
    the store so the PUT is bounded by the same number the gate reserved. Without it the
    gate reserves a window the transmission is free to overrun, which is the one way an
    object is lost rather than reported: the drain SIGKILLs the process in the middle of a
    PUT, so that object and every object after it go with no record of which.

    *cancel* is what an interval cycle hands over instead. It has no window to reserve, so
    it has no budget to pass; the predicate is how its upload is still ended the moment the
    stop arrives, rather than after a transmission that has no bound at all. Neither is a
    permanent failure, so a cut upload is recorded as this cycle's refusal and re-attempted.
    """
    if _already_durable(key, snapshot, state):
        result.record_unchanged(key)
        return
    if snapshot.size > MAX_OBJECT_BYTES:
        # Uploaded anyway: skipping it is the data loss this design exists to
        # prevent. The warning is the point -- the front refuses to restore an
        # object this large, so the pair is honest but incomplete for this one
        # conversation, and an operator has to hear that from the writer rather
        # than from a customer's failed turn.
        log.warning(
            "backup: %s is %d B, above the %d B ceiling the restore side will "
            "read. It is uploaded, and a turn continuing this conversation on a "
            "replaced task will be refused rather than served an empty history.",
            name,
            snapshot.size,
            MAX_OBJECT_BYTES,
        )
        result.record_above_ceiling(key)
    try:
        store.put(key, snapshot.fh, snapshot.size, budget=budget, cancel=cancel)
    except StoreUnusable:
        # Not recorded as this object's refusal and not retried: the bucket
        # itself cannot be written, so every remaining object in this cycle and
        # every later cycle meets the same answer. It leaves here whole so the
        # process can end on it.
        raise
    except UploadCancelled as exc:
        # The interval cycle's counterpart to the window refusal below, and the reason it
        # is worth recording rather than silently dropping: this object is left to the
        # final cycle DELIBERATELY, and the pair is withheld so the index never names a
        # transcript whose upload was cut. What was given up is one object the final cycle
        # sends anyway; what was bought is the window it sends everything else in.
        log.info("backup: PUT for %s was cut by the stop -- %s", name, exc)
        result.record_refused(name, f"the upload was cut by the stop ({exc})")
        return
    except UploadDeadlineExceeded as exc:
        # A refusal like any other, and deliberately not permanent: the bucket answered
        # and the object is simply larger than this window at the rate the connection is
        # managing. Recording it is the whole gain over being killed mid-PUT -- the pair
        # is withheld, the cycle exits non-zero, and the name of what is missing is in
        # the log instead of nowhere.
        log.error("backup: PUT for %s did not fit its window -- %s", name, exc)
        result.record_refused(name, f"the upload did not fit its window ({exc})")
        return
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        log.error("backup: PUT failed for %s -- %s", name, exc)
        result.record_refused(name, f"the upload failed ({exc})")
        return
    _record_durable(settings, state, key, snapshot.fingerprint)
    result.record_uploaded(key)
