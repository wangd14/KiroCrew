"""Private SQLite slots for codex runtimes.

Every ``codex app-server`` on a host opened the same ``$CODEX_HOME/*.sqlite``
files, so a Codex Desktop daemon plus Crew runtimes failed new sessions with
``database is locked``. Each Crew codex runtime now holds its own slot and points
``CODEX_SQLITE_HOME`` at it.

Every slot a test takes is released in a ``finally``: a leaked lock fd keeps the
slot directory open, and on Windows an open handle makes pytest's tmp cleanup
fail for the whole run.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import sys
import threading
from pathlib import Path

import pytest

from kiro_crew import platform_compat
from kiro_crew.acp.harness.codex import sqlite_slot_root
from kiro_crew.acp.runtime import AcpRuntime
from kiro_crew.acp.state_slots import MAX_STATE_SLOTS, StateSlot, acquire_state_slot

#: A pid no live process can own on any supported platform (above every pid_max),
#: so a fake process that reaches a kill or a signal reaches nothing on the runner.
_UNALLOCATABLE_PID = 99_999_999_999

_POSIX_LINKS = pytest.mark.skipif(
    sys.platform == "win32", reason="symlink/hardlink creation needs privileges on Windows"
)


@pytest.fixture
def released():
    """Track every slot a test takes; release them all at teardown."""
    slots: list[StateSlot] = []

    def _track(slot: StateSlot) -> StateSlot:
        slots.append(slot)
        return slot

    yield _track
    for slot in slots:
        slot.release()


@pytest.fixture
def releases(monkeypatch: pytest.MonkeyPatch):
    """Every release the runtime hands off the loop, so a test can wait for it.

    Release is fire-and-forget on a worker thread; a test that asserts the freed
    slot is reused has to wait for the release to land first, or it races it.
    """
    import kiro_crew.acp.runtime as runtime_mod

    real = runtime_mod.release_slot_off_loop
    futures: list[asyncio.Future[None]] = []

    def recording(slot: StateSlot):
        future = real(slot)
        if future is not None:
            futures.append(future)
        return future

    monkeypatch.setattr(runtime_mod, "release_slot_off_loop", recording)

    async def settled() -> None:
        while futures:
            await futures.pop()

    return settled


async def _released(future: "asyncio.Future[None] | None") -> None:
    """Wait for one ``_release_state_slot`` hand-off to land."""
    assert future is not None, "no slot was held"
    await future


def test_concurrent_holders_get_distinct_slots_and_a_freed_slot_is_reused(
    tmp_path: Path, released
) -> None:
    first = released(acquire_state_slot(tmp_path))
    second = released(acquire_state_slot(tmp_path))
    assert first.path != second.path
    assert {first.path.name, second.path.name} == {"slot-0", "slot-1"}

    first.release()
    first.release()  # a second release is harmless
    again = released(acquire_state_slot(tmp_path))
    # Lowest free slot wins, so a restarted runtime keeps its built databases.
    assert again.path == tmp_path / "slot-0"


def test_every_slot_held_raises_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, released
) -> None:
    monkeypatch.setattr("kiro_crew.acp.state_slots.MAX_STATE_SLOTS", 2)
    released(acquire_state_slot(tmp_path))
    released(acquire_state_slot(tmp_path))
    with pytest.raises(OSError, match="are held"):
        acquire_state_slot(tmp_path)
    assert MAX_STATE_SLOTS == 64


def test_a_slot_whose_directory_is_a_file_is_skipped(tmp_path: Path, released) -> None:
    # Left behind by a same-UID process, or by hand: not repaired, not fatal.
    (tmp_path / "slot-0").write_text("")
    assert released(acquire_state_slot(tmp_path)).path == tmp_path / "slot-1"


def test_every_slot_unusable_raises_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kiro_crew.acp.state_slots.MAX_STATE_SLOTS", 2)
    (tmp_path / "slot-0").write_text("")
    (tmp_path / "slot-1").write_text("")
    with pytest.raises(OSError, match="no usable state slot"):
        acquire_state_slot(tmp_path)


@_POSIX_LINKS
@pytest.mark.parametrize("shape", ["symlink", "hardlink"])
def test_a_lock_file_that_aliases_another_path_is_refused(
    tmp_path: Path, released, shape: str
) -> None:
    # A lock on a file that is also reachable by another name is no lock: the other
    # name's holder and this one never contend. Mirrors ``agent_state._locked``.
    slot = tmp_path / "slot-0"
    slot.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("")
    if shape == "symlink":
        os.symlink(elsewhere, slot / ".kirocrew-slot.lock")
    else:
        os.link(elsewhere, slot / ".kirocrew-slot.lock")
    assert released(acquire_state_slot(tmp_path)).path == tmp_path / "slot-1"


def test_a_linked_slot_directory_is_skipped_without_being_traversed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, released
) -> None:
    # A junction (or symlink) at slot-N is skipped, and never RESOLVED on the way
    # to that decision: on Windows resolving one aimed at a UNC share is an
    # outbound SMB authentication, and ``Path.mkdir(exist_ok=True)`` resolves it
    # with the ``is_dir()`` it answers the existence question with.
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "slot-0"
    platform_compat.symlink_or_junction(target, link)
    real_stat = os.stat
    traversals: list[str] = []

    def recording_stat(path, *args, **kwargs):
        if os.fspath(path) == os.fspath(link) and kwargs.get("follow_symlinks", True):
            traversals.append(os.fspath(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", recording_stat)
    assert released(acquire_state_slot(tmp_path)).path == tmp_path / "slot-1"
    assert not (target / ".kirocrew-slot.lock").exists()
    assert traversals == []


def test_a_linked_root_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    platform_compat.symlink_or_junction(target, tmp_path / "root")
    with pytest.raises(OSError, match="is a link"):
        acquire_state_slot(tmp_path / "root")
    assert not (target / "slot-0").exists()


def test_a_linked_ancestor_of_the_root_is_refused_before_any_mkdir(tmp_path: Path) -> None:
    # The root itself is not a link, so a check on the root alone passes; the
    # ancestor is, and creating the root would resolve it. Refused first.
    target = tmp_path / "target"
    target.mkdir()
    platform_compat.symlink_or_junction(target, tmp_path / "link")
    with pytest.raises(OSError, match="is a link"):
        acquire_state_slot(tmp_path / "link" / "root")
    assert list(target.iterdir()) == []


@_POSIX_LINKS
def test_without_o_nofollow_a_linked_lock_file_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, released
) -> None:
    """The Windows branch, run from a machine that is not Windows.

    ``os.O_NOFOLLOW`` does not exist on Windows, so an ``os.open`` with
    ``getattr(os, "O_NOFOLLOW", 0)`` follows a link at the lock's name and takes
    the lock on the TARGET. The lock has to be opened by the no-reparse seam, whose
    Windows arm (``CreateFileW`` + ``FILE_FLAG_OPEN_REPARSE_POINT``) refuses the
    link inside the open. Deleting the attribute reproduces the platform
    difference; the seam is patched with that arm's contract, because its POSIX
    arm is the very ``O_NOFOLLOW`` being removed.
    """
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    slot = tmp_path / "slot-0"
    slot.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("")
    os.symlink(elsewhere, slot / ".kirocrew-slot.lock")
    real_open = platform_compat.create_file_no_reparse_rw
    refused: list[str] = []

    def windows_arm(path, mode=0o600):
        # What the Windows arm asserts about the object it opened: a reparse point
        # at the name is ELOOP, the target untouched.
        if os.path.islink(path):
            refused.append(os.fspath(path))
            raise OSError(errno.ELOOP, "reparse point at the final component", os.fspath(path))
        return real_open(path, mode)

    monkeypatch.setattr(platform_compat, "create_file_no_reparse_rw", windows_arm)
    assert released(acquire_state_slot(tmp_path)).path == tmp_path / "slot-1"
    assert refused == [str(slot / ".kirocrew-slot.lock")]
    # Nobody holds a lock on the target: taking one ourselves succeeds.
    fd = os.open(elsewhere, os.O_RDWR)
    try:
        with platform_compat.file_lock(fd, exclusive=True, wait=False):
            pass
    finally:
        os.close(fd)


def test_slot_root_is_crews_own_unless_the_operator_chose_a_sqlite_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "crew"))
    assert sqlite_slot_root({}) == tmp_path / "crew" / "codex-sqlite"
    # Moving CODEX_HOME moves codex's own files, not Crew's slots.
    assert (
        sqlite_slot_root({"CODEX_HOME": str(tmp_path / "c")}) == tmp_path / "crew" / "codex-sqlite"
    )
    # The operator's own SQLite home is theirs: no slot.
    assert sqlite_slot_root({"CODEX_SQLITE_HOME": str(tmp_path / "s")}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operator_set", [False, True])
async def test_the_codex_plan_takes_no_slot_when_the_operator_set_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operator_set: bool
) -> None:
    from unittest.mock import AsyncMock

    import kiro_crew.acp.harness.codex as codex_mod
    from kiro_crew.acp import client as client_mod
    from kiro_crew.acp.harness.base import SpawnContext
    from kiro_crew.acp.harness.codex import CodexHarness

    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "crew"))
    monkeypatch.setattr(
        client_mod, "_resolve_codex_acp_bin", lambda: (["/opt/codex-acp"], ["/opt"])
    )
    monkeypatch.setattr(codex_mod, "resolve_spawn_masks", AsyncMock(return_value=((), ())))
    monkeypatch.setattr(codex_mod, "_sandbox_wrapper_generations", lambda mode: 0)
    environ = {"CODEX_SQLITE_HOME": str(tmp_path / "theirs")} if operator_set else {}

    plan = await CodexHarness().resolve_spawn(
        SpawnContext(
            agent="kirocrew",
            work_dir=tmp_path / "workspace",
            model=None,
            environ=environ,
            home=tmp_path / "home",
            sandbox_mode="standard",
            member_context=None,
        )
    )

    if operator_set:
        assert plan.private_state_dir is None
    else:
        assert plan.private_state_dir == (
            "CODEX_SQLITE_HOME",
            str(tmp_path / "crew" / "codex-sqlite"),
        )


def _bare_runtime() -> AcpRuntime:
    # The binder reads only its own slot field, so no process is needed.
    return object.__new__(AcpRuntime)


@pytest.mark.asyncio
async def test_runtime_binds_one_slot_and_keeps_it_across_respawns(tmp_path: Path) -> None:
    one, two, three = _bare_runtime(), _bare_runtime(), _bare_runtime()
    request = ("CODEX_SQLITE_HOME", str(tmp_path))
    try:
        env_one: dict[str, str] = {}
        env_two: dict[str, str] = {}
        await one._bind_state_slot(env_one, request)
        await two._bind_state_slot(env_two, request)
        assert env_one["CODEX_SQLITE_HOME"] != env_two["CODEX_SQLITE_HOME"]

        respawn: dict[str, str] = {}
        await one._bind_state_slot(respawn, request)
        assert respawn == env_one

        await _released(one._release_state_slot())
        env_three: dict[str, str] = {}
        await three._bind_state_slot(env_three, request)
        assert env_three == env_one
    finally:
        for runtime in (one, two, three):
            future = runtime._release_state_slot()
            if future is not None:
                await future


@pytest.mark.asyncio
async def test_runtime_without_a_request_leaves_env_alone(tmp_path: Path) -> None:
    env = {"CODEX_SQLITE_HOME": "operator"}
    await _bare_runtime()._bind_state_slot(env, None)
    assert env == {"CODEX_SQLITE_HOME": "operator"}


@pytest.mark.asyncio
async def test_a_callers_own_value_is_kept_and_takes_no_slot(tmp_path: Path, released) -> None:
    # extra_env from a cron or a workflow named the location: theirs, like the
    # operator's own variable is at the harness.
    runtime = _bare_runtime()
    env = {"CODEX_SQLITE_HOME": str(tmp_path / "theirs")}
    await runtime._bind_state_slot(env, ("CODEX_SQLITE_HOME", str(tmp_path / "slots")))
    assert env == {"CODEX_SQLITE_HOME": str(tmp_path / "theirs")}
    assert getattr(runtime, "_state_slot", None) is None
    assert released(acquire_state_slot(tmp_path / "slots")).path == tmp_path / "slots" / "slot-0"


@pytest.mark.asyncio
async def test_unusable_root_falls_back_to_the_shared_default_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    env: dict[str, str] = {}
    with caplog.at_level(logging.WARNING, logger="kiro_crew.acp.runtime"):
        await _bare_runtime()._bind_state_slot(env, ("CODEX_SQLITE_HOME", str(blocker / "root")))
    assert "CODEX_SQLITE_HOME" not in env
    assert "no private CODEX_SQLITE_HOME" in caplog.text


@pytest.mark.asyncio
async def test_acquisition_runs_off_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, released
) -> None:
    import threading

    import kiro_crew.acp.runtime as runtime_mod

    loop_thread = threading.current_thread()
    seen: list[threading.Thread] = []
    real = runtime_mod.acquire_state_slot

    def recording(root: Path) -> StateSlot:
        seen.append(threading.current_thread())
        return real(root)

    monkeypatch.setattr(runtime_mod, "acquire_state_slot", recording)
    runtime = _bare_runtime()
    try:
        await runtime._bind_state_slot({}, ("CODEX_SQLITE_HOME", str(tmp_path)))
    finally:
        await _released(runtime._release_state_slot())
    # A slot walk is up to MAX_STATE_SLOTS mkdir+open+lock triples: not loop work.
    assert seen and seen[0] is not loop_thread


@pytest.mark.asyncio
async def test_release_runs_off_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, released
) -> None:
    loop_thread = threading.current_thread()
    seen: list[threading.Thread] = []
    real = StateSlot.release

    def recording(self: StateSlot) -> None:
        seen.append(threading.current_thread())
        real(self)

    monkeypatch.setattr(StateSlot, "release", recording)
    runtime = _bare_runtime()
    await runtime._bind_state_slot({}, ("CODEX_SQLITE_HOME", str(tmp_path)))
    released(runtime._state_slot)
    await _released(runtime._release_state_slot())
    # An unlock and an os.close can block in the kernel: not loop work either.
    assert seen and seen[0] is not loop_thread


@pytest.mark.asyncio
async def test_a_release_that_fails_is_logged_and_never_an_unretrieved_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real = StateSlot.release

    def failing(self: StateSlot) -> None:
        real(self)  # the lock does go, so the test leaves nothing held
        raise OSError(errno.EIO, "close failed")

    monkeypatch.setattr(StateSlot, "release", failing)
    runtime = _bare_runtime()
    await runtime._bind_state_slot({}, ("CODEX_SQLITE_HOME", str(tmp_path)))
    with caplog.at_level(logging.WARNING, logger="kiro_crew.acp.state_slots"):
        future = runtime._release_state_slot()
        assert future is not None
        # Nobody awaits a fire-and-forget release, so it must not carry the error.
        assert await future is None
    assert future.exception() is None
    assert "release failed" in caplog.text and "close failed" in caplog.text


@pytest.mark.asyncio
async def test_a_slot_taken_after_the_awaiter_was_cancelled_is_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, released
) -> None:
    import kiro_crew.acp.runtime as runtime_mod

    loop_thread = threading.current_thread()
    entered = threading.Event()
    proceed = threading.Event()
    releases: list[StateSlot] = []
    release_threads: list[threading.Thread] = []
    real = runtime_mod.acquire_state_slot

    class _Observed(StateSlot):
        # Release is what the runtime owes; a dropped slot's lock going away by
        # garbage collection is not that, so the call itself is what is asserted.
        def release(self) -> None:
            releases.append(self)
            release_threads.append(threading.current_thread())
            super().release()

    def slow(root: Path) -> StateSlot:
        entered.set()
        assert proceed.wait(10)
        taken = real(root)
        return released(_Observed(taken.root, taken.path, taken._stack))

    monkeypatch.setattr(runtime_mod, "acquire_state_slot", slow)
    runtime = _bare_runtime()
    binding = asyncio.ensure_future(
        runtime._bind_state_slot({}, ("CODEX_SQLITE_HOME", str(tmp_path)))
    )
    await asyncio.to_thread(entered.wait, 10)
    binding.cancel()
    with pytest.raises(asyncio.CancelledError):
        await binding
    # The thread cannot be stopped; whatever it takes now has no owner.
    proceed.set()
    deadline = asyncio.get_running_loop().time() + 10
    while not releases:
        assert asyncio.get_running_loop().time() < deadline, "the late slot was never released"
        await asyncio.sleep(0.02)
    assert getattr(runtime, "_state_slot", None) is None
    # The done-callback runs on the loop; the release it owes must not.
    assert release_threads[0] is not loop_thread
    assert released(acquire_state_slot(tmp_path)).path == tmp_path / "slot-0"


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed_dead", [True, False])
async def test_kill_frees_the_slot_only_once_the_tree_is_confirmed_dead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirmed_dead: bool, released, releases
) -> None:
    import kiro_crew.acp.runtime as runtime_mod

    request = ("CODEX_SQLITE_HOME", str(tmp_path / "slots"))
    runtime = AcpRuntime(work_dir=tmp_path / "workspace")
    await runtime._bind_state_slot({}, request)
    held = runtime._state_slot
    assert held is not None
    released(held)
    runtime._pid = _UNALLOCATABLE_PID  # a process was created against this slot

    async def fake_kill_inner(self, *, expected=False, reason=""):
        # A survivor (an unsignalled root or a retained descendant) leaves the
        # flag False, and it still has the slot's databases open.
        self._process_tree_confirmed_dead = confirmed_dead

    monkeypatch.setattr(runtime_mod, "authorize_runtime_kill", lambda *a, **k: True)
    monkeypatch.setattr(AcpRuntime, "_kill_inner", fake_kill_inner)
    await runtime.kill(expected=True, reason="test")
    await releases()

    assert runtime._state_slot is None
    probe = released(acquire_state_slot(tmp_path / "slots"))
    if confirmed_dead:
        assert runtime._retired_state_slots == []
        assert probe.path == held.path
    else:
        # Retired: still held against the next runtime ...
        assert runtime._retired_state_slots == [held]
        assert probe.path != held.path
        probe.release()
        # ... and not reused by this runtime's own respawn, whose process would
        # otherwise share databases with the survivor.
        respawn: dict[str, str] = {}
        await runtime._bind_state_slot(respawn, request)
        try:
            assert respawn["CODEX_SQLITE_HOME"] != str(held.path)
        finally:
            await _released(runtime._release_state_slot())


@pytest.mark.asyncio
async def test_a_kill_that_raises_still_retires_the_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, released
) -> None:
    """The slot is settled even when the kill itself blows up.

    A kill that raises (or is cancelled) never confirmed the tree dead, so the
    process may still hold the slot's databases open. Settling must not sit
    behind the kill on the success path: left unsettled, the slot stays bound to
    this runtime and its respawn would share databases with the survivor.
    """
    import kiro_crew.acp.runtime as runtime_mod

    request = ("CODEX_SQLITE_HOME", str(tmp_path / "slots"))
    runtime = AcpRuntime(work_dir=tmp_path / "workspace")
    await runtime._bind_state_slot({}, request)
    held = runtime._state_slot
    assert held is not None
    released(held)
    runtime._pid = _UNALLOCATABLE_PID  # a process was created against this slot

    async def failing_kill_inner(self, *, expected=False, reason=""):
        raise RuntimeError("signal delivery failed")

    monkeypatch.setattr(runtime_mod, "authorize_runtime_kill", lambda *a, **k: True)
    monkeypatch.setattr(AcpRuntime, "_kill_inner", failing_kill_inner)
    with pytest.raises(RuntimeError, match="signal delivery failed"):
        await runtime.kill(expected=True, reason="test")

    # Retired, not released: dropped from this runtime ...
    assert runtime._state_slot is None
    assert runtime._retired_state_slots == [held]
    # ... still held against the next runtime ...
    probe = released(acquire_state_slot(tmp_path / "slots"))
    assert probe.path != held.path
    probe.release()
    # ... and not reused by this runtime's own respawn.
    respawn: dict[str, str] = {}
    await runtime._bind_state_slot(respawn, request)
    try:
        assert respawn["CODEX_SQLITE_HOME"] != str(held.path)
    finally:
        await _released(runtime._release_state_slot())


@pytest.mark.asyncio
async def test_kill_of_a_runtime_that_never_spawned_frees_its_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, released, releases
) -> None:
    import kiro_crew.acp.runtime as runtime_mod

    runtime = AcpRuntime(work_dir=tmp_path / "workspace")
    await runtime._bind_state_slot({}, ("CODEX_SQLITE_HOME", str(tmp_path / "slots")))
    held = runtime._state_slot
    assert held is not None
    released(held)

    async def fake_kill_inner(self, *, expected=False, reason=""):
        pass  # no process: the flag stays False, but nothing ever used the slot

    monkeypatch.setattr(runtime_mod, "authorize_runtime_kill", lambda *a, **k: True)
    monkeypatch.setattr(AcpRuntime, "_kill_inner", fake_kill_inner)
    await runtime.kill(expected=True, reason="test")
    await releases()

    assert runtime._state_slot is None
    assert runtime._retired_state_slots == []
    assert released(acquire_state_slot(tmp_path / "slots")).path == held.path


def _patch_spawn_prelude(monkeypatch: pytest.MonkeyPatch, slots: Path):
    """Drive ``spawn`` with a fake plan up to process creation, with no host effects.

    Returns the runtime module. Windows cleanup admission is passed through, as
    every fake-process spawn test does: a fake process has no handle to pin, and
    the admission path would otherwise record it as a tree needing manual
    handling -- process-global state that fails every later spawn test in the
    worker.
    """
    import kiro_crew.acp.runtime as runtime_mod
    from kiro_crew.acp.harness.base import SpawnPlan

    async def plan(self):
        return SpawnPlan(argv=["/bin/true"], private_state_dir=("CODEX_SQLITE_HOME", str(slots)))

    async def unbound(work_dir):
        return work_dir, None

    async def admit(factory):
        return await factory()

    monkeypatch.setenv("KIROCREW_NATIVE_SKILL_PROJECTION", "0")
    monkeypatch.setattr(AcpRuntime, "_resolve_spawn_plan", plan)
    monkeypatch.setattr(runtime_mod, "wrap_argv", lambda argv, mode, **k: (list(argv), None))
    monkeypatch.setattr(runtime_mod, "cgroup_scope_argv", lambda argv: list(argv))
    monkeypatch.setattr(
        runtime_mod, "assert_voice_runtime_outside_agent_workspace", lambda *a: None
    )
    monkeypatch.setattr(runtime_mod, "bind_voice_safe_agent_workspace_async", unbound)
    monkeypatch.setattr(runtime_mod.platform_compat, "create_windows_cleanup_owned_process", admit)
    return runtime_mod


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at", ["before_process_creation", "process_creation"])
async def test_a_spawn_that_never_creates_a_process_frees_its_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: str, released, releases
) -> None:
    slots = tmp_path / "slots"
    runtime_mod = _patch_spawn_prelude(monkeypatch, slots)
    seen: dict[str, str] = {}

    class _StopSpawn(Exception):
        pass

    async def stop_spawn(*args, **kwargs):
        seen.update(kwargs["env"])
        raise _StopSpawn()

    monkeypatch.setattr(runtime_mod, "create_subprocess_limited", stop_spawn)
    if fail_at == "before_process_creation":
        # An await after the slot is taken but before the subprocess call.
        def stop_early(env):
            seen.update(env)
            raise _StopSpawn()

        monkeypatch.setattr(runtime_mod, "inject_xdist_auto_cap", stop_early)

    runtime = AcpRuntime(work_dir=tmp_path / "workspace")
    with pytest.raises(_StopSpawn):
        await runtime.spawn()
    await releases()

    assert seen["CODEX_SQLITE_HOME"] == str(slots / "slot-0")
    assert runtime._state_slot is None
    assert released(acquire_state_slot(slots)).path == slots / "slot-0"


@pytest.mark.asyncio
async def test_a_spawn_cancelled_before_process_creation_frees_its_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, released, releases
) -> None:
    slots = tmp_path / "slots"
    runtime_mod = _patch_spawn_prelude(monkeypatch, slots)
    reached = asyncio.Event()

    async def hang(work_dir):
        # Slot taken, sandbox file live, no process yet: the widest cancel window.
        reached.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(runtime_mod, "bind_voice_safe_agent_workspace_async", hang)

    runtime = AcpRuntime(work_dir=tmp_path / "workspace")
    spawning = asyncio.ensure_future(runtime.spawn())
    await asyncio.wait_for(reached.wait(), 10)
    assert runtime._state_slot is not None
    spawning.cancel()
    with pytest.raises(asyncio.CancelledError):
        await spawning
    await releases()

    assert runtime._state_slot is None
    assert released(acquire_state_slot(slots)).path == slots / "slot-0"


# Windows wraps the child in a real-process pin (create_windows_cleanup_owned_process);
# a fake process fails that pin and parks a pending tree cleanup in module state,
# which then refuses every later spawn in the worker. The slot rule under test is
# platform-neutral, so POSIX coverage is enough.
@pytest.mark.skipif(
    sys.platform == "win32", reason="fake process cannot pass the Windows child pin"
)
@pytest.mark.asyncio
async def test_a_spawn_that_fails_after_a_reap_left_a_survivor_retires_its_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, released
) -> None:
    slots = tmp_path / "slots"
    runtime_mod = _patch_spawn_prelude(monkeypatch, slots)

    class _StopSpawn(Exception):
        pass

    class _FakeProcess:
        pid = _UNALLOCATABLE_PID
        returncode: int | None = None

    async def fake_process(*args, **kwargs):
        return _FakeProcess()

    def stop_after_process(pid):
        # The first thing spawn does once the process is live and its pid recorded.
        raise _StopSpawn()

    async def kill_leaving_a_survivor(self, *, expected=False, reason=""):
        # The reap drops the handle, but a descendant it could not signal still
        # has the slot's databases open: the flag stays False.
        self._process = None

    monkeypatch.setattr(runtime_mod, "create_subprocess_limited", fake_process)
    monkeypatch.setattr(runtime_mod.platform_compat, "get_process_start_id", stop_after_process)
    monkeypatch.setattr(runtime_mod, "authorize_runtime_kill", lambda *a, **k: True)
    monkeypatch.setattr(AcpRuntime, "_kill_inner", kill_leaving_a_survivor)

    runtime = AcpRuntime(work_dir=tmp_path / "workspace")
    try:
        with pytest.raises(_StopSpawn):
            await runtime.spawn()

        assert runtime._pid == _UNALLOCATABLE_PID
        assert runtime._process is None
        assert runtime._process_tree_confirmed_dead is False
        assert runtime._state_slot is None
        (held,) = runtime._retired_state_slots
        assert held.path == slots / "slot-0"
        # The survivor's slot must not be handed to the next runtime.
        assert released(acquire_state_slot(slots)).path == slots / "slot-1"
    finally:
        for slot in runtime._retired_state_slots:
            slot.release()
