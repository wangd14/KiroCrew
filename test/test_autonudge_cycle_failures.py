"""A loop whose own delivered cycles keep FAILING stands down.

The prompt loop stood down on three narrow signals -- a structurally rejected
payload, an unanswered tool approval, a cycle that never got a model session --
but not on the generic case: a cycle that reached a session and dispatched, then
DIED (a backend error after retries were spent, a persistent tool error, a
prompt timeout). Such a loop fired every interval, spent a turn, produced
nothing, and stopped only when ``max_cycles`` happened to run out. These tests
pin the streak, the terminal stand-down, and the clearing on a landed turn --
all driven by recorded evidence, so a loop that recovers is never held back.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from kiro_crew import autonudge as _an
from kiro_crew.autonudge import (
    CONSECUTIVE_FAILURE_REASON,
    AutoNudgeService,
    NudgeLoop,
)

SLOT = "chat-1-654"


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setenv("KIROCREW_AUTONUDGE", "1")


@pytest.fixture(autouse=True)
def _no_published_service_outlives_the_test():
    """Unpublish the singleton and bound leftover work after every test.

    ``start()`` publishes the service as the module singleton and only ``stop()``
    clears it, so a service left published hands a later test one bound to a store
    it is finished with. Sync on purpose: this suite's pytest-asyncio pin errors
    on async-generator fixtures at setup.
    """
    yield
    svc = _an.get_instance()
    if svc is None:
        return
    try:
        inflight = getattr(svc, "_inflight_adds", None)
        if inflight is not None:
            for task in list(inflight):
                task.cancel()
            inflight.clear()
        svc.stop()
    finally:
        _an._INSTANCE = None


@pytest.fixture
def store_dir(tmp_path_factory):
    """A store directory owned by the SESSION, not by one test.

    ``_persist_locked`` hands ``_write_state`` to a thread and a thread cannot be
    cancelled, so a write that lands late against a per-test ``tmp_path``
    re-creates a directory pytest already removed.
    """
    return tmp_path_factory.mktemp("autonudge-cycle-failures")


@pytest.fixture
def svc(store_dir):
    return AutoNudgeService(base_dir=store_dir)


@pytest.fixture
def _nosleep(monkeypatch):
    """Collapse the timer's idle wait so ``_timer`` runs synchronously."""

    async def _noop(_secs):
        return None

    monkeypatch.setattr(_an.asyncio, "sleep", _noop)


async def _armed(svc, **kwargs) -> NudgeLoop:
    await svc.start()
    loop = await svc.add(slot_key=SLOT, message="go", idle_secs=600, **kwargs)
    await svc._timers[loop.id]
    return loop


async def _stop_and_drain(svc: AutoNudgeService) -> None:
    timers = list(svc._timers.values())
    svc.stop()
    if timers:
        await asyncio.gather(*timers, return_exceptions=True)
    inflight = list(svc._inflight_adds)
    if inflight:
        await asyncio.gather(*inflight, return_exceptions=True)


@pytest.mark.asyncio
async def test_the_hook_records_a_streak_without_stopping_the_loop(svc, _nosleep):
    loop = await _armed(svc)

    svc.notify_cycle_failed(SLOT)
    svc.notify_cycle_failed(SLOT)

    assert svc._loops[loop.id].consecutive_failed_cycles == 2
    assert svc._loops[loop.id].active is True
    await _stop_and_drain(svc)


@pytest.mark.asyncio
async def test_a_landed_turn_clears_the_streak(svc, _nosleep):
    """A turn that completed proves the session can make progress.

    Any landed turn counts, a human's as much as a cycle's: the streak only ever
    stops a loop, so clearing it on broader evidence can only keep a working loop
    running.
    """
    loop = await _armed(svc)
    svc.notify_cycle_failed(SLOT)
    svc.notify_cycle_failed(SLOT)

    svc.notify_cycle_landed(SLOT)

    assert svc._loops[loop.id].consecutive_failed_cycles == 0
    await _stop_and_drain(svc)


@pytest.mark.asyncio
async def test_a_short_streak_still_fires(svc, _nosleep):
    """Below the stand-down threshold nothing changes: a couple of errors are
    weather, and the loop keeps firing."""
    fired: list[NudgeLoop] = []

    async def on_fire(loop, *_args, **_kwargs):
        fired.append(loop)
        return True

    loop = await _armed(svc)
    svc._on_fire = on_fire
    svc._loops[loop.id].consecutive_failed_cycles = 4

    svc._cancel_timer(loop.id)
    await svc._timer(svc._loops[loop.id])

    assert len(fired) == 1
    assert svc._loops[loop.id].active is True
    await _stop_and_drain(svc)


@pytest.mark.asyncio
async def test_five_failures_stand_the_loop_down(svc, _nosleep):
    """Terminal, in the same shape as the other bounds: deactivate + ``expired``
    and NO fire."""
    fired: list[NudgeLoop] = []

    async def on_fire(loop, *_args, **_kwargs):
        fired.append(loop)
        return True

    events: list[tuple[str, str]] = []
    svc.subscribe(lambda ev, lp: events.append((ev, lp.id if lp else "")))
    loop = await _armed(svc)
    svc._on_fire = on_fire
    svc._loops[loop.id].consecutive_failed_cycles = 5

    svc._cancel_timer(loop.id)
    await svc._timer(svc._loops[loop.id])

    refreshed = svc._loops[loop.id]
    assert refreshed.active is False
    assert refreshed.stopped_reason == CONSECUTIVE_FAILURE_REASON
    assert ("expired", loop.id) in events, f"the stop must be user-visible; got {events}"
    assert fired == []
    await _stop_and_drain(svc)


@pytest.mark.asyncio
async def test_the_cycle_cap_still_wins(svc, _nosleep):
    """The failure bound must not relabel an existing terminal outcome."""
    loop = await _armed(svc, max_cycles=1)
    svc._loops[loop.id].cycle_count = 1
    svc._loops[loop.id].consecutive_failed_cycles = 9

    svc._cancel_timer(loop.id)
    await svc._timer(svc._loops[loop.id])

    assert svc._loops[loop.id].stopped_reason == "cycle_cap"
    await _stop_and_drain(svc)


@pytest.mark.asyncio
async def test_the_stand_down_is_re_armable(svc, _nosleep):
    """The remedy is external (the backend or tool recovering), so a directive may
    revive the loop -- and the revival starts a fresh run, so the old streak must
    not survive it."""
    loop = await _armed(svc)
    svc._loops[loop.id].consecutive_failed_cycles = 5
    svc._cancel_timer(loop.id)
    await svc._timer(svc._loops[loop.id])
    assert svc._loops[loop.id].active is False

    revived = await svc.update(loop.id, active=True)

    assert revived is not None and revived.active is True
    assert svc._loops[loop.id].consecutive_failed_cycles == 0
    assert svc._loops[loop.id].stopped_reason == ""
    await _stop_and_drain(svc)


@pytest.mark.asyncio
async def test_a_manual_pause_is_not_relabelled_by_the_bound(svc, _nosleep):
    """A user pause that lands first must not be overwritten by the failure bound
    firing on an in-flight cycle -- it is in ``_TERMINAL_BOUND_REASONS`` for
    exactly that no-op protection."""
    loop = await _armed(svc)
    svc._loops[loop.id].consecutive_failed_cycles = 5
    await svc.update(loop.id, active=False)  # manual pause, records "manual"

    svc._cancel_timer(loop.id)
    await svc._timer(svc._loops[loop.id])

    assert svc._loops[loop.id].stopped_reason == "manual"
    await _stop_and_drain(svc)


@pytest.mark.asyncio
async def test_the_hook_ignores_an_inactive_loop(svc, _nosleep):
    """A paused loop is not accruing evidence; recording would stale-stop it."""
    loop = await _armed(svc)
    await svc.update(loop.id, active=False)

    svc.notify_cycle_failed(SLOT)

    assert svc._loops[loop.id].consecutive_failed_cycles == 0
    await _stop_and_drain(svc)


@pytest.mark.asyncio
async def test_the_hook_is_silent_for_an_unknown_slot(svc, _nosleep):
    await _armed(svc)

    svc.notify_cycle_failed("chat-9-nobody")  # must not raise

    await _stop_and_drain(svc)


@pytest.mark.asyncio
async def test_the_streak_survives_a_restart(svc, store_dir, _nosleep):
    """The cause that fails a cycle (a wedged backend, a broken tool) routinely
    outlives a restart, so a reload that dropped the streak would restart the
    doomed cycles."""
    loop = await _armed(svc)
    svc.notify_cycle_failed(SLOT)
    svc.notify_cycle_failed(SLOT)
    await svc._persist_locked()
    await _stop_and_drain(svc)
    _an._INSTANCE = None

    reloaded = AutoNudgeService(base_dir=store_dir)
    await reloaded.start()
    try:
        assert reloaded._loops[loop.id].consecutive_failed_cycles == 2
    finally:
        await _stop_and_drain(reloaded)


@pytest.mark.parametrize("stored", ["3", None, -5, 2.9, float("nan"), 10**400])
@pytest.mark.asyncio
async def test_a_malformed_persisted_streak_is_normalised_at_load(svc, store_dir, stored, _nosleep):
    """The store is agent-writable and the streak is compared with ``>=`` on every
    wake, so a string or ``null`` would raise TypeError inside ``_timer`` and the
    automation would silently never fire again -- surviving every reload."""
    loop = await _armed(svc)
    await svc._persist_locked()
    await _stop_and_drain(svc)
    _an._INSTANCE = None

    raw = json.loads((store_dir / "autonudge.json").read_text(encoding="utf-8"))
    for row in raw["loops"]:
        row["consecutive_failed_cycles"] = stored
    (store_dir / "autonudge.json").write_text(json.dumps(raw), encoding="utf-8")

    reloaded = AutoNudgeService(base_dir=store_dir)
    await reloaded.start()
    try:
        value = reloaded._loops[loop.id].consecutive_failed_cycles
        assert isinstance(value, int) and value >= 0
        # The comparison the guard named must not raise.
        reloaded._cancel_timer(loop.id)
        await reloaded._timer(reloaded._loops[loop.id])
    finally:
        await _stop_and_drain(reloaded)
