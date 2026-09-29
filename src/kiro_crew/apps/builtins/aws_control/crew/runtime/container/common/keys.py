"""One derivation of every object key this task reads or writes.

Two processes address the same bucket for opposite reasons. The sidecar PUTs a
conversation's transcript and the two authority files; the front GETs one
transcript on the turn that continues it. They must agree on the key exactly, and
drift between them is invisible in both directions: a GET simply misses, and a
customer whose history was not found is indistinguishable from a new customer.

So the derivation lives here and neither process keeps its own copy. The first
live deployment doubled the crew name in every key (``crews/<crew>/<crew>/``)
because two places each decided one prefix, and twelve green gates passed it
because writer and reader agreed with each other while both disagreed with the
contract. A shared definition is what makes that class of mistake unrepresentable
rather than tested for.

## The layout

``<backup_prefix>/<crew_name>/data/<tail>``, where ``tail`` is the object's path
inside the data home. Transcripts are ``data/sessions/<stem>.jsonl``, archived
segments keep their own path under ``data/sessions/archive/``, and the authority
files sit at the data home's root, so theirs are ``data/session_map.json`` and
``data/open_slots.json``.

The ``data/`` namespace is a segment of its own so that anything the owner's
control plane later files beside the data (a manifest, a label) cannot collide with
a conversation whose slot id happens to match.

## One function does the work

Every key comes from :func:`data_key`, which is the only place a path becomes a
key. :func:`transcript_key` and :func:`authority_key` are the two named cases and
both call it, so a transcript addressed by its slot stem and the same file
addressed by its path cannot produce two different keys. Writing either case out
separately is how the two ways of naming one object start to disagree.
"""

from __future__ import annotations

import re
import secrets
import time
from pathlib import Path

from .config import Settings

__all__ = [
    "NAMESPACE",
    "TRANSCRIPT_SUFFIX",
    "AUTHORITY_NAMES",
    "GENERATION_PREFIX",
    "AUTHORITY_POINTER_NAME",
    "OutsideDataHome",
    "object_prefix",
    "full_key",
    "data_key",
    "transcript_key",
    "authority_key",
    "new_generation_id",
    "is_generation_id",
    "generations_prefix",
    "authority_generation_key",
    "authority_pointer_key",
]

#: The segment every data object sits under. See the module docstring.
NAMESPACE = "data/"

#: A conversation transcript's file extension, which is part of its key.
TRANSCRIPT_SUFFIX = ".jsonl"

#: The files that turn a slot id back into a conversation.
#:
#: A tuple, and the only names :func:`authority_key` will derive a key for. They are
#: the backend's own filenames, so an arbitrary name here would either name an object
#: nothing reads or -- on the restore side, which writes what it fetches -- put bytes
#: at a path the backend never asked for.
AUTHORITY_NAMES: tuple[str, ...] = ("session_map.json", "open_slots.json")

#: The segment every generation's authority pair sits under: ``gen/<id>/<name>``.
#:
#: A generation id is minted fresh by each cycle (:func:`new_generation_id`) and its two
#: files are written into ``gen/<id>/`` and NEVER rewritten. Two writers racing in the
#: task-replacement window therefore each write a DISTINCT generation rather than both
#: writing the same slot, so neither can clobber the other's pair and no committed pair is
#: ever a cross-writer tear. The pointer names the one generation that is committed, and
#: which one wins is settled by a compare-and-swap on the pointer alone.
#:
#: This replaces an earlier two-slot scheme where the pair alternated between a fixed
#: ``a``/``b`` and the shared slot key was mutable: two concurrent writers targeting the
#: same slot could interleave their PUTs into it and commit a torn pair. Immutable
#: per-writer keys remove the shared mutable object the tear needed.
GENERATION_PREFIX = "gen/"

#: The grammar a generation id must match to be turned into a key or read from a pointer.
#:
#: Lexically sortable and writer-unique: a zero-padded nanosecond timestamp, a hyphen, and
#: random hex. Sortable so "the newest complete generation" is well defined when a boot has
#: to adopt one whose pointer did not land; random-suffixed so two writers minting in the
#: same nanosecond still get distinct ids. Validated by SHAPE rather than an allowlist,
#: because the set of live ids is unbounded and dynamic -- but the shape forbids a slash or
#: any path character, so a pointer cannot steer a fetched object outside ``gen/<id>/`` any
#: more than the old slot allowlist could.
_GENERATION_ID_RE = re.compile(r"\A[0-9]{20}-[0-9a-f]{16}\Z")


def new_generation_id() -> str:
    """A fresh, writer-unique, lexically-sortable generation id.

    ``<20-digit ns timestamp>-<16 hex>``. The timestamp orders generations by creation so a
    boot with no committed pointer can adopt the NEWEST complete pair; the random suffix
    keeps two writers minting in the same nanosecond distinct. The whole is immutable once a
    cycle writes its pair under it.
    """
    return f"{time.time_ns():020d}-{secrets.token_hex(8)}"


def is_generation_id(value: object) -> bool:
    """Whether *value* is a well-formed generation id.

    A shape check, not an allowlist. The pointer names a generation id and the restore
    writes the fetched pair to the matching local authority names, so the id must not be
    able to carry a path -- the grammar admits only digits, one hyphen and hex, no slash.
    """
    return isinstance(value, str) and _GENERATION_ID_RE.match(value) is not None


#: The object naming the generation id whose pair is COMMITTED.
#:
#: It sits BESIDE the ``data/`` namespace rather than inside it, which is what that
#: segment exists for: a control-plane object under ``data/`` would be indistinguishable
#: from a conversation whose slot id happens to match this name.
#:
#: Its absence is meaningful and is not an error. A bucket written before this protocol
#: has authority objects at their ``data/`` keys and no pointer, and those are read as
#: GENERATION 0 -- they are never deleted, moved or rewritten, so adopting the protocol
#: costs no migration and a bucket stays readable by the writer that made it.
AUTHORITY_POINTER_NAME = "authority_generation.json"


class OutsideDataHome(ValueError):
    """The path is not inside the data home, so it has no key.

    Raised rather than folded to the path's leaf name. A key is this task's claim
    about its own state; deriving one for a path outside the data home would upload
    something the task does not own, or -- on the way back -- write a fetched object
    outside it.
    """


def object_prefix(settings: Settings) -> str:
    """Everything before the namespace: the configured prefix and the crew.

    Empty when neither is set, which is the local-test shape. Each part is stripped of
    its own slashes before joining, so a prefix given as ``crews``, ``/crews`` or
    ``crews/`` produces one key rather than three.
    """
    parts = [p for p in (settings.backup_prefix.strip("/"), settings.crew_name.strip("/")) if p]
    return ("/".join(parts) + "/") if parts else ""


def full_key(settings: Settings, rel_key: str) -> str:
    """A namespace-relative key resolved to the object's full key."""
    return object_prefix(settings) + rel_key


def data_key(settings: Settings, path: Path) -> str:
    """The full key of the data-home file at *path*.

    The one place a path becomes a key. ``as_posix`` is deliberate: an object key uses
    forward slashes whatever the host's separator is, and this tree's own host is Linux
    either way.
    """
    try:
        tail = path.relative_to(settings.data_home).as_posix()
    except ValueError as exc:
        raise OutsideDataHome(
            f"{path} is not inside the data home ({settings.data_home}), so it has no "
            "object key. A key names this task's own state, in both directions."
        ) from exc
    if tail in ("", "."):
        raise OutsideDataHome(
            f"{path} is the data home itself, which is a directory and not an object."
        )
    return full_key(settings, f"{NAMESPACE}{tail}")


def transcript_key(settings: Settings, stem: str) -> str:
    """The full key of the transcript whose filename stem is *stem*.

    The stem carries the transport prefix the backend folds into it, so this is
    ``dashboard_<slot>`` and not ``<slot>``. Deriving that stem from a turn's ``id``
    belongs to the front, which is the only process that sees an id at all; this
    function takes the stem it produces.
    """
    return data_key(settings, settings.sessions_dir / f"{stem}{TRANSCRIPT_SUFFIX}")


def is_archive_key(settings: Settings, key: str) -> bool:
    """Whether *key* names a file under the archive directory.

    The archive is the one part of the set whose keys accumulate without bound -- rotation
    nests it and retention-off never prunes it -- and its segments are IMMUTABLE once
    written. A caller separates them out on that basis: their durability is a bounded
    remembered set, not an entry in the lifetime fingerprint map that would then grow with
    the archive. Derived from the same ``data_key`` construction so it cannot drift from how
    the keys are actually formed.
    """
    prefix = data_key(settings, settings.archive_dir) + "/"
    return key.startswith(prefix)


def authority_key(settings: Settings, name: str) -> str:
    """The full key of authority file *name* at GENERATION 0.

    Generation 0 is the layout a bucket written before the generation protocol has, and
    those objects are never rewritten. It is also what a bucket with no pointer is read
    as, so this derivation stays exactly as it was rather than moving under a slot.

    Refuses a name that is not one of :data:`AUTHORITY_NAMES`. The restore side writes
    the bytes it fetches to the local path of the same name, so an unconstrained name
    would be a key derivation that also decides where a file lands.
    """
    if name not in AUTHORITY_NAMES:
        raise ValueError(
            f"{name!r} is not an authority file. The authority files are "
            f"{', '.join(AUTHORITY_NAMES)}, and a key is derived only for those: the "
            "restore side writes what it fetches to the matching local name, so any "
            "name accepted here is also a path this task would write to."
        )
    return data_key(settings, settings.config_dir / name)


def generations_prefix(settings: Settings) -> str:
    """The full key prefix every generation directory sits under.

    Ends in the ``gen/`` segment, so a key under it is a generation object and not the
    pointer or a transcript. It is the shape that tells an authority-pair key apart from
    the rest of the bucket.
    """
    return full_key(settings, GENERATION_PREFIX)


def authority_generation_key(settings: Settings, generation_id: str, name: str) -> str:
    """The full key of authority file *name* inside generation *generation_id*.

    Refuses an id whose SHAPE is not a generation id, for the reason ``authority_key``
    refuses a foreign name: the restore writes what it fetches to the local authority name,
    so a key derivation that accepted an arbitrary id could let a pointer place a fetched
    object under a path this task did not choose. The shape check admits only the minted
    grammar -- digits, one hyphen, hex -- so the id can carry no slash and no traversal.
    """
    if not is_generation_id(generation_id):
        raise ValueError(
            f"{generation_id!r} is not a generation id. A generation id is minted by "
            "new_generation_id() as '<20-digit ns>-<16 hex>', and a pointer naming anything "
            "else names objects no cycle of this writer published."
        )
    if name not in AUTHORITY_NAMES:
        raise ValueError(
            f"{name!r} is not an authority file. The authority files are "
            f"{', '.join(AUTHORITY_NAMES)}."
        )
    return full_key(settings, f"{GENERATION_PREFIX}{generation_id}/{name}")


def authority_pointer_key(settings: Settings) -> str:
    """The full key of the object naming the committed generation slot.

    Deliberately not built through :func:`data_key`: this object is not a file in the
    data home, and giving it a ``data/`` key would put a control-plane object in the
    namespace a conversation's own key comes from. It has no local path either -- the
    restore reads it to decide which generation to boot from and never writes it to disk.
    """
    return full_key(settings, AUTHORITY_POINTER_NAME)
