"""Exclusive state directories for hosts whose on-disk state must not be shared.

Some hosts keep SQLite databases that tolerate only one live process at a time.
``codex app-server`` is one: every app-server on a host opens the same
``$CODEX_HOME/*.sqlite`` files, so a Codex Desktop daemon plus two Crew runtimes
fail new sessions with ``database is locked``.

A slot is a numbered directory under a root the harness names, held by an
exclusive advisory lock on a file inside it for as long as one runtime owns it.
The lowest free slot wins, so a restarted runtime reuses a directory whose
databases are already built instead of paying a fresh backfill each spawn. The
lock is released by closing its file, so a crashed gateway frees its slots with
no cleanup step.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import stat
from pathlib import Path

from kiro_crew import platform_compat
from kiro_crew.executors import subprocess_executor

logger = logging.getLogger(__name__)

__all__ = ["MAX_STATE_SLOTS", "StateSlot", "acquire_state_slot", "release_slot_off_loop"]

#: More live runtimes than this on one root means something leaks runtimes; the
#: caller falls back to the host's shared default rather than grow without bound.
MAX_STATE_SLOTS = 64

_LOCK_NAME = ".kirocrew-slot.lock"


class StateSlot:
    """One held slot: its directory, and the lock that keeps it exclusive."""

    def __init__(self, root: Path, path: Path, stack: contextlib.ExitStack) -> None:
        self.root = root
        self.path = path
        self._stack = stack

    def release(self) -> None:
        """Drop the lock. Safe to call twice."""
        self._stack.close()


def _release_logging_errors(slot: StateSlot) -> None:
    """``slot.release()`` for a future nobody awaits: an error is logged, never carried."""
    try:
        slot.release()
    except OSError as exc:
        logger.warning("state slot %s release failed: %s", slot.path, exc)


def release_slot_off_loop(slot: StateSlot) -> asyncio.Future[None] | None:
    """Release *slot* without blocking the caller's loop. Fire-and-forget.

    Releasing is an unlock and an ``os.close``, which can block in the kernel, so
    on a running loop it runs on the subprocess/teardown pool -- the pool for
    closes that may wedge -- and the returned future is for a caller that wants to
    wait, not one it must. Off the loop (a worker thread, or a process with no
    loop) the release is inline and ``None`` is returned. The lock stays the
    authority either way: while a release is in flight the slot reads as held, so
    a concurrent acquire takes the next one and no slot ever has two holders.
    """
    try:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(subprocess_executor(), _release_logging_errors, slot)
    except RuntimeError:
        # No running loop, or the pool is already shut down at interpreter exit.
        slot.release()
        return None


def _mkdir_no_follow(path: Path) -> None:
    """Create *path* as a directory, or accept a REAL directory already at the name.

    ``Path.mkdir(exist_ok=True)`` answers "does it exist" with ``is_dir()``, which
    traverses a link or a junction at the name -- on Windows, into a UNC target.
    ``os.mkdir`` creates without resolving the final component and ``lstat`` reads
    the name itself, so nothing here follows what may sit there. The parent must
    already exist.
    """
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    if platform_compat.is_link_or_junction(path):
        raise OSError(f"{path} is a link")
    if not stat.S_ISDIR(os.lstat(path).st_mode):
        raise NotADirectoryError(f"{path} is not a directory")


def acquire_state_slot(root: Path) -> StateSlot:
    """Take the lowest free slot under *root*.

    A mkdir, an open and one non-blocking lock per slot tried, up to
    ``MAX_STATE_SLOTS`` of them: filesystem work, so callers on an event loop run
    it in a thread. Raises ``OSError`` when the root cannot be created, is a link
    or has a linked ancestor, or no slot can be taken: every slot is held, or
    every one is unusable.

    A slot that cannot be used is skipped, never repaired: ``slot-N`` being a file
    or a link, or its lock being a link or a hard-linked file, is the shape a
    same-UID process leaves behind on purpose, and a lock taken on a file that
    aliases another path is no lock at all. No name is resolved before it is
    screened: directories are created and inspected without following a link at
    the name, and the lock is opened by ``platform_compat.create_file_no_reparse_rw``,
    which refuses a link or reparse point inside the open itself on both platforms
    (``O_NOFOLLOW`` alone is 0 on Windows). The descriptor is then checked for a
    regular, single-link inode before the lock is taken, as ``agent_state._locked``
    does for the model-state sidecar.
    """
    if platform_compat.first_linked_ancestor(root) is not None:
        raise OSError(f"an ancestor of state slot root {root} is a link")
    _mkdir_no_follow(root)
    unusable: OSError | None = None
    for index in range(MAX_STATE_SLOTS):
        path = root / f"slot-{index}"
        stack = contextlib.ExitStack()
        try:
            _mkdir_no_follow(path)
            fd = platform_compat.create_file_no_reparse_rw(path / _LOCK_NAME, 0o600)
            stack.callback(os.close, fd)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise OSError(f"{path / _LOCK_NAME} is not a plain file")
            stack.enter_context(platform_compat.file_lock(fd, exclusive=True, wait=False))
        except BlockingIOError:
            # Held by a live runtime, here or in another gateway process.
            stack.close()
            continue
        except OSError as exc:
            stack.close()
            logger.warning("state slot %s skipped: %s", path, exc)
            unusable = exc
            continue
        except BaseException:
            stack.close()
            raise
        return StateSlot(root, path, stack)
    if unusable is not None:
        raise OSError(f"no usable state slot under {root} (last: {unusable})")
    raise OSError(f"all {MAX_STATE_SLOTS} state slots under {root} are held")
