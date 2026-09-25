"""The persisted identity of every spawned backend: ``app_backends.pids.json``.

App backends run in their OWN session (``start_new_session=True``) and are NOT in the
gateway's process group, so when the liveness probe SIGKILLs a wedged gateway (no
``on_cleanup`` runs) they orphan, reparent to PID 1, and accumulate across restarts.
Each spawned backend's ``(pid, start_time, port, spawn_instance)`` is therefore
persisted here and the next clean start reaps the survivors of a PRIOR generation
(:mod:`~kiro_crew.apps.backend_runtime.stale_reap`). ``start_time`` is the PID-reuse
guard: a recorded pid whose live start time does not match names another process now.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from kiro_crew import platform_compat
from kiro_crew.apps.backend_runtime import _FACADE
from kiro_crew.apps.backend_runtime.tracking import AppProcess, _lock, _processes
from kiro_crew.atomic_write import atomic_write
from kiro_crew.config.loader import config_dir
from kiro_crew.session_pid import (
    group_vouching_available,
    process_spawn_instance,
)

logger = logging.getLogger(_FACADE)


# Serializes the pidfile read-modify-write. _record_app_pid runs on the
# to_thread worker that spawns a backend (both the runtime app-enable path and
# the startup reconcile offload start_app_backend via asyncio.to_thread) while
# _forget_app_pid runs on the to_thread worker that stops one — distinct OS
# threads, so without this lock their non-atomic read-modify-writes of the
# whole JSON dict lose each other's entries.
_pidfile_lock = threading.Lock()


def _pidfile_path() -> Path:
    # The app-backend spawn record: pid, start instant and per-spawn instance
    # token for each app whose backend the gateway launched. The adoption path
    # attributes a captured owner against this record to tell an app's own
    # backend from a survivor of a previous install.
    return config_dir() / "app_backends.pids.json"


def _proc_start_time(pid: int) -> str | None:
    """Stable per-process start time, or None if unavailable.

    PID-reuse guard: a recorded pid whose live start_time does not match has
    been recycled to an unrelated process and MUST NOT be killed. The value must
    be stable across gateway restarts (the reap compares a string recorded by a
    prior generation against one read now), so it cannot use ``hash()`` — that
    is salted per interpreter by ``PYTHONHASHSEED``.

    Per-platform sources live in ``platform_compat.process_start_time``: Linux
    reads ``/proc/<pid>/stat`` field 22, Windows the process creation FILETIME
    through a query-only handle, and other POSIX ``ps -o lstart=``. Resolving it
    there is what keeps the guard alive on Windows — a ``/proc``-or-``ps`` probe
    answers None for every pid there, and a recorded None makes the reap decline
    to confirm ANY backend, so nothing is ever reaped and the entries accumulate.
    """
    return platform_compat.process_start_time(pid)


def _read_pidfile() -> dict[str, dict[str, Any]]:
    try:
        with open(_pidfile_path()) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        # A corrupt/half-written pidfile (e.g. a SIGKILL mid-write before atomic
        # writes landed, or a leftover from an older build) silently disabling
        # the reap is exactly the leak this feature exists to prevent — log it.
        logger.warning("App-backend pidfile unreadable (%s); stale-reap skipped this start", exc)
        return {}


def _write_pidfile(data: dict[str, dict[str, Any]]) -> None:
    # Atomic temp-file + rename (fsync): the whole point of the pidfile is to
    # survive a gateway SIGKILL, so a non-atomic open("w") that truncates first
    # would leave an empty/partial file if the kill lands mid-write.
    try:
        atomic_write(_pidfile_path(), json.dumps(data), fsync=True)
    except OSError as exc:
        logger.debug("Could not write app-backend pidfile: %s", exc)


def _record_app_pid(
    app_name: str, pid: int, port: int, spawn_instance: str | None = None
) -> str | None:
    """Persist a spawned backend's identity for the startup stale-reap. Never raises.

    *spawn_instance* is the per-spawn ``KIROCREW_SPAWN_INSTANCE`` stamped on the
    backend's environment and inherited by its whole tree. It is what lets the
    reap vouch the group's MEMBERS once the leader itself is gone; a row written
    by an older build carries none, and the reap then declines to touch that
    group rather than aim a signal at a bare (possibly recycled) group number.
    """
    if pid <= 0:
        return None
    start_time: str | None = None
    try:
        # Compute start_time BEFORE taking the lock: the probe is slow on the
        # platforms that cannot answer from memory (a `ps` spawn on macOS, an
        # OpenProcess round trip on Windows), and holding _pidfile_lock across
        # that IO would serialize concurrent enable/stop/uninstall ops behind
        # it. Mirrors the reap path's validate-lock-free / store-under-lock
        # discipline.
        start_time = _proc_start_time(pid)
        with _pidfile_lock:
            data = _read_pidfile()
            entry: dict[str, Any] = {"pid": pid, "start_time": start_time, "port": port}
            if spawn_instance:
                entry["spawn_instance"] = spawn_instance
            data[app_name] = entry
            _write_pidfile(data)
    except Exception as exc:  # noqa: BLE001 — persistence must never break a spawn
        logger.debug("Could not record app pid for %s: %s", app_name, exc)
    return start_time


def _forget_app_pid(app_name: str) -> dict[str, Any] | None:
    """Drop an app's pidfile entry and return it (called when no process identity is tracked).

    The removed row is handed BACK so a caller that must undo the removal can. A stop
    drops the row inside its lifecycle transition, before it signals anything, and it
    can then REFUSE and restore tracking; the row is where an adopted backend's
    provenance is read from, so leaving it dropped makes every retry unable to
    attribute the listener it is trying to stop.
    """
    try:
        with _pidfile_lock:
            data = _read_pidfile()
            removed = data.pop(app_name, None)
            if removed is not None:
                _write_pidfile(data)
            return removed if isinstance(removed, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not forget app pid for %s: %s", app_name, exc)
        return None


def _forget_app_pid_if(app_name: str, pid: int, start_time: str | None) -> dict[str, Any] | None:
    """Drop a pidfile row only if it still identifies the expected process.

    Returns the removed row, or ``None`` when the row stayed, for the same
    reversibility reason as :func:`_forget_app_pid`.
    """
    try:
        with _pidfile_lock:
            data = _read_pidfile()
            entry = data.get(app_name)
            if (
                isinstance(entry, dict)
                and entry.get("pid") == pid
                and entry.get("start_time") == start_time
            ):
                data.pop(app_name, None)
                _write_pidfile(data)
                return entry
            return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not conditionally forget app pid for %s: %s", app_name, exc)
        return None


def _restore_app_pid(app_name: str, row: dict[str, Any]) -> None:
    """Put back a row a refused stop removed, unless the name has been re-recorded.

    Deliberately ``setdefault`` and not an overwrite: between the removal and the
    restore a fresh spawn can have recorded its own identity, and replacing that with
    the older row would aim both the stale-reap and adoption provenance at a process
    that is gone. Never raises -- a restore that cannot happen leaves the retry no
    worse off than before this function existed.
    """
    try:
        with _pidfile_lock:
            data = _read_pidfile()
            if app_name in data:
                return
            data[app_name] = row
            _write_pidfile(data)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not restore app pid for %s: %s", app_name, exc)


def _forget_exited_leader_row(app_name: str, ap: AppProcess) -> None:
    """Drop an exited leader's pidfile row unless an adopted instance rests on it.

    Called once the whole restart sequence is OVER, never between its attempts. The
    row carries the spawn tree's instance token, and that token is the only thing
    that attributes a pre-fork worker or a detached child still holding the app's
    declared port, so a removal taken while the loop can still retry leaves
    :func:`_adoption_provenance` with no record on every later attempt: adoption
    refuses, the respawn returns nothing, and the loop retries forever against an
    orphan only an operator can clear.

    An adopted instance is tracked with no handle of ours (``proc is None``) and its
    ownership is re-read from this row on every re-bind, so the row stays while that
    instance does. Any other outcome leaves the row stale and it goes: a fresh spawn
    has recorded its own identity, or nothing is tracked at all. The drop stays
    conditional on the exited leader's own pid and start instant, so a successor's row
    is never the one removed.
    """
    with _lock:
        tracked = _processes.get(app_name)
    if tracked is not None and tracked.proc is None:
        return
    _forget_app_pid_if(app_name, ap.pid, ap.pid_start_time)


def _adoption_provenance(app_name: str, owners: list[int]) -> tuple[bool, str]:
    """Whether EVERY pid in *owners* is attributable to this gateway's spawn for *app_name*.

    Adoption otherwise keys on two facts that say nothing about the code the
    listener runs: the port the manifest declares, and a health answer on it. A
    backend that outlives its app's uninstall and rebinds that port satisfies both,
    so the next install that happens to use the same app name adopts it -- the
    gateway then addresses a process the install did not place as that app's
    backend, reports its health as the app's, and aims stop at its PIDs.

    The record that closes it already exists: every backend this gateway spawns is
    written to the app pidfile as ``pid`` + ``start_time`` + per-spawn
    ``spawn_instance``. Attribution asks whether every captured owner matches that
    record; a survivor of an earlier install that rebound the port is not in it, so
    it is refused and the misattribution the issue describes does not happen.

    Two routes attribute ONE owner, and either is enough for that owner:

    ``leader`` -- the owner IS the recorded pid and its live start instant still
    equals the recorded one. A pid plus a start instant names one process for good,
    so this route answers on every platform, and it attributes that pid alone.

    ``tree`` -- the owner's exec-time environment carries the recorded
    ``spawn_instance``. The whole spawn tree inherits that token, so this route
    reaches a pre-fork worker or a detached child still holding the port after its
    leader has exited. It reads ``/proc/<pid>/environ``, which exists on Linux
    alone; :func:`group_vouching_available` reports that, and the reason names it
    so an operator can tell "not ours" from "this host cannot see".

    EVERY owner must clear a route, because every owner the caller captured enters
    the managed set and is signalled at stop. A same-UID ``SO_REUSEPORT`` co-binder
    lands in the same dispatch tier as the real backend, so attributing the set from
    one match would hand stop an unrelated process to terminate -- and the start-time
    token stop re-checks was captured at the same moment, so it confirms that
    bystander rather than excluding it.

    FAILS CLOSED, uniformly: no recorded spawn, an unreadable pidfile, a row
    carrying neither usable identity, an empty owner set, and any owner no route
    attributes all return ``False``. Refusing costs a start on a port the gateway
    does not own, which the caller reports; adopting on an unproven listener is the
    defect itself.

    Returns ``(attributed, reason)``. *reason* is a short phrase for the log and the
    audit trail on both verdicts.
    """
    if not owners:
        return False, "no owning pid to attribute"
    try:
        with _pidfile_lock:
            row = _read_pidfile().get(app_name)
    except Exception as exc:  # noqa: BLE001 — an unreadable record must refuse, not raise
        return False, f"app pidfile unreadable ({exc})"
    if not isinstance(row, dict):
        return False, "no spawn recorded for this app"
    try:
        recorded_pid = int(row.get("pid", 0))
    except (TypeError, ValueError):
        recorded_pid = 0
    recorded_start = row.get("start_time")
    instance = row.get("spawn_instance")
    if not isinstance(instance, str) or not instance:
        instance = ""
    vouchable = bool(instance) and group_vouching_available()
    unattributed: list[int] = []
    for pid in owners:
        if (
            recorded_pid > 0
            and recorded_start
            and pid == recorded_pid
            and _proc_start_time(pid) == recorded_start
        ):
            continue
        if vouchable and process_spawn_instance(pid) == instance:
            continue
        unattributed.append(pid)
    if not unattributed:
        return True, f"every owner {owners} belongs to the recorded spawn"
    if not instance:
        why = "the row carries no spawn instance to vouch its tree with"
    elif not vouchable:
        why = "this host cannot read a process's spawn instance"
    else:
        why = "they were not placed by this gateway for this app"
    return False, (
        f"owner pid(s) {unattributed} are not the recorded spawn (pid {recorded_pid}): {why}"
    )


def retire_windows_app_tracking(pid: int, creation: int) -> None:
    """Retire only this incarnation's app rows, while its cleanup pin is held.

    This mandatory writer does not use the best-effort readers/writers: an
    unreadable file or failed atomic write must leave the cleanup receipt owed.
    No app name or caller callback is retained by the cleanup registry.
    """
    with _pidfile_lock:
        try:
            with open(_pidfile_path(), encoding="utf-8") as stream:
                data = json.load(stream)
        except FileNotFoundError:
            return
        if not isinstance(data, dict):
            raise OSError("Windows app tracking file is malformed")
        remove = [
            name
            for name, entry in data.items()
            if isinstance(entry, dict)
            and entry.get("pid") == pid
            and entry.get("start_time") == str(creation)
        ]
        if remove:
            for name in remove:
                del data[name]
            atomic_write(_pidfile_path(), json.dumps(data), fsync=True)
