"""Slack Reconnect: ``GatewayOrchestrator.reconnect_slack`` + ``POST /api/slack/reconnect``.

Slack credentials are hoisted once, in ``GatewayOrchestrator.__init__``, and
nothing reassigned them afterwards, so a token or owner ID saved from the
dashboard could take effect only through ``POST /api/restart``. These tests
pin the in-place alternative:

* the orchestrator re-reads the store, recomputes ``_slack_enabled`` from the
  tokens now on disk (a stale ``False`` would make ``init_socket_mode`` a
  silent no-op), tears the old socket client down, re-awaits
  ``init_socket_mode`` on the running loop and records the outcome where the
  settings badge reads it;
* a store that cannot be read leaves the live connection untouched;
* the dashboard's Slack client mirror is cleared with the old socket and
  published again only behind a connected one, and the dashboard's
  ``owner_id`` follows the saved owner -- so a rejected workspace or a former
  owner never keeps dashboard access through a reconnect;
* concurrent callers share one handshake;
* the route carries the PUT's direct-local gate and answers the
  ``connected`` / ``connect_error`` shape ``GET /api/slack/config`` documents.

The orchestrator is built through ``__new__`` (its ``__init__`` boots the
world); ``init_socket_mode`` and ``_connect_slack`` are replaced by recorders
because a real handshake needs Slack.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from kiro_crew.config.loader import CRED_OWNER_ID, CRED_SLACK_APP_TOKEN, CRED_SLACK_BOT_TOKEN

# Obvious placeholders: nothing here reaches Slack.
NEW_CREDS = {
    CRED_SLACK_APP_TOKEN: "xapp-new-not-a-real-value",
    CRED_SLACK_BOT_TOKEN: "xoxb-new-not-a-real-value",
    CRED_OWNER_ID: "U0NEWOWNER",
}


def _orch(creds: dict[str, str] | Exception = NEW_CREDS, *, old_client: Any = None) -> Any:
    """A GatewayOrchestrator in the state a failed boot leaves it in.

    ``_slack_enabled`` is False and the tokens are stale, exactly the state the
    issue describes: ``init_socket_mode`` early-returns on that flag, so a
    reconnect that does not recompute it is a no-op.
    """
    from kiro_crew.slack.gateway import GatewayOrchestrator

    orch = GatewayOrchestrator.__new__(GatewayOrchestrator)
    orch._cfg = MagicMock()
    if isinstance(creds, Exception):
        orch._cfg.load_credentials.side_effect = creds
    else:
        orch._cfg.load_credentials.return_value = dict(creds)
    orch._app_token = "xapp-stale"
    orch._bot_token = "xoxb-stale"
    orch._owner_id = ""
    orch._allowed_users = set()
    orch._slack_enabled = False
    orch._slack_connect_error = "invalid_auth"
    orch.slack = None
    orch._socket_client = old_client
    orch._slack_seen = MagicMock(name="boot-seen-cache")
    orch._slack_reconnect_task = None
    orch.dashboard_state = MagicMock()
    orch.dashboard_state.slack_socket_connected = False
    orch.dashboard_state.slack_connect_error = "invalid_auth"
    orch.dashboard_state.slack_client = None
    orch.dashboard_state.owner_id = "U0FORMEROWNER"
    orch._tracking_channels = set()
    orch._background_tasks = set()
    return orch


class _Recorder:
    """Stand-in for ``init_socket_mode`` that behaves like the real one's edges.

    On ``owner_missing`` it mirrors the real early return (flag off, no
    client); otherwise it installs a fresh client and records where it ran.
    """

    def __init__(self, *, owner_missing: bool = False, hold: asyncio.Event | None = None):
        self.calls: list[tuple[Any, Any]] = []
        self.loops: list[asyncio.AbstractEventLoop] = []
        self.threads: list[threading.Thread] = []
        self.owner_missing = owner_missing
        self.hold = hold
        self.client = MagicMock(name="new-socket-client")

    async def __call__(self, orch: Any, seen: Any) -> None:
        self.calls.append((orch, seen))
        self.loops.append(asyncio.get_running_loop())
        self.threads.append(threading.current_thread())
        if self.hold is not None:
            await self.hold.wait()
        if not orch._slack_enabled:
            return
        if self.owner_missing or not orch._owner_id:
            orch._slack_enabled = False
            orch.slack = None
            return
        orch._socket_client = self.client


def _patches(recorder: _Recorder, connect: Any = None, client_cls: Any = None):
    """Patch the handshake seams; ``connect`` replaces ``_connect_slack``."""
    connect_mock = connect if connect is not None else AsyncMock(return_value=True)
    return (
        patch("kiro_crew.slack.events.init_socket_mode", recorder),
        patch("kiro_crew.slack.gateway.RealSlackClient", client_cls or MagicMock()),
        patch("kiro_crew.slack.gateway.GatewayOrchestrator._connect_slack", connect_mock),
    )


async def _run(
    orch: Any, recorder: _Recorder, connect: Any = None, client_cls: Any = None
) -> dict[str, object]:
    p_init, p_client, p_connect = _patches(recorder, connect, client_cls)
    with p_init, p_client, p_connect:
        return await orch.reconnect_slack()


# ── orchestrator ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reconnect_recomputes_enabled_and_hoists_new_credentials() -> None:
    old = MagicMock(name="old-socket-client")
    old.close = AsyncMock()
    orch = _orch(old_client=old)
    rec = _Recorder()
    client_cls = MagicMock(name="RealSlackClient")

    result = await _run(orch, rec, client_cls=client_cls)

    # The stale False is recomputed from the tokens now on disk -- the whole
    # point: without it init_socket_mode returns before doing anything.
    assert orch._slack_enabled is True
    assert orch._app_token == NEW_CREDS[CRED_SLACK_APP_TOKEN]
    assert orch._bot_token == NEW_CREDS[CRED_SLACK_BOT_TOKEN]
    assert orch._owner_id == NEW_CREDS[CRED_OWNER_ID]
    assert orch._allowed_users == {NEW_CREDS[CRED_OWNER_ID]}
    # Old client torn down before the new handshake; new one installed.
    old.close.assert_awaited_once()
    assert orch._socket_client is rec.client
    # The boot-time dedup cache is reused, so redelivered envelopes stay deduped.
    assert rec.calls == [(orch, orch._slack_seen)]
    # Web API client rebuilt on the new bot token and, the socket being
    # connected, mirrored to the dashboard; the dashboard's owner follows.
    client_cls.assert_called_once_with(NEW_CREDS[CRED_SLACK_BOT_TOKEN])
    assert orch.slack is client_cls.return_value
    assert orch.dashboard_state.slack_client is orch.slack
    assert orch.dashboard_state.owner_id == NEW_CREDS[CRED_OWNER_ID]
    # Outcome recorded where GET /api/slack/config reads it, and returned.
    assert orch.dashboard_state.slack_socket_connected is True
    assert orch.dashboard_state.slack_connect_error == ""
    assert result == {"connected": True, "connect_error": ""}


@pytest.mark.asyncio
async def test_reconnect_awaits_init_socket_mode_on_the_running_loop() -> None:
    """WSSocketModeClient needs a current loop in the constructing thread."""
    orch = _orch()
    rec = _Recorder()

    await _run(orch, rec)

    assert rec.loops == [asyncio.get_running_loop()]
    assert rec.threads == [threading.current_thread()]


@pytest.mark.asyncio
async def test_load_failure_leaves_the_live_connection_untouched() -> None:
    old = MagicMock(name="old-socket-client")
    old.close = AsyncMock()
    orch = _orch(OSError("store unreadable"), old_client=old)
    rec = _Recorder()

    with pytest.raises(OSError):
        await _run(orch, rec)

    old.close.assert_not_awaited()
    assert orch._socket_client is old
    assert orch._app_token == "xapp-stale"
    assert orch._slack_enabled is False
    assert rec.calls == []
    # Nothing was recorded either: the badge still describes the live state.
    assert orch.dashboard_state.slack_connect_error == "invalid_auth"


@pytest.mark.asyncio
async def test_missing_tokens_disable_slack_without_a_handshake() -> None:
    old = MagicMock(name="old-socket-client")
    old.close = AsyncMock()
    orch = _orch({CRED_OWNER_ID: "U0NEWOWNER"}, old_client=old)
    rec = _Recorder()

    result = await _run(orch, rec)

    old.close.assert_awaited_once()  # a cleared token must not keep the old socket alive
    assert orch._socket_client is None
    assert orch._slack_enabled is False
    assert orch.slack is None
    assert orch.dashboard_state.slack_client is None
    assert rec.calls == []
    assert result == {"connected": False, "connect_error": "tokens_missing"}
    assert orch.dashboard_state.slack_socket_connected is False
    assert orch.dashboard_state.slack_connect_error == "tokens_missing"


@pytest.mark.asyncio
async def test_missing_owner_is_named_for_the_badge() -> None:
    creds = {k: v for k, v in NEW_CREDS.items() if k != CRED_OWNER_ID}
    orch = _orch(creds)
    rec = _Recorder()

    result = await _run(orch, rec)

    assert len(rec.calls) == 1  # init_socket_mode ran (the flag was recomputed True)
    assert orch._socket_client is None
    assert result == {"connected": False, "connect_error": "owner_id_missing"}


@pytest.mark.asyncio
async def test_connect_failure_reason_is_surfaced() -> None:
    orch = _orch()
    rec = _Recorder()

    async def _connect(self: Any) -> bool:
        self._slack_connect_error = "invalid_auth"
        return False

    result = await _run(orch, rec, connect=_connect)

    assert result == {"connected": False, "connect_error": "invalid_auth"}
    assert orch.dashboard_state.slack_socket_connected is False
    assert orch.dashboard_state.slack_connect_error == "invalid_auth"


@pytest.mark.asyncio
async def test_old_client_close_failure_does_not_block_the_new_handshake() -> None:
    old = MagicMock(name="old-socket-client")
    old.close = AsyncMock(side_effect=RuntimeError("websocket already gone"))
    orch = _orch(old_client=old)
    rec = _Recorder()

    result = await _run(orch, rec)

    assert orch._socket_client is rec.client
    assert result["connected"] is True


@pytest.mark.asyncio
async def test_rejected_workspace_leaves_no_client_in_the_dashboard() -> None:
    """The enterprise gate's decline must not leave a sendable client behind.

    ``init_socket_mode`` clears ``orch.slack`` on that path but never touched
    the dashboard mirror, so a reconnect that published the mirror before the
    handshake would hand every dashboard Slack sender a client on a workspace
    the gate rejected.
    """
    old_web = MagicMock(name="old-web-client")
    orch = _orch()
    orch.slack = old_web
    orch.dashboard_state.slack_client = old_web
    seen: list[Any] = []

    class _Rejecting(_Recorder):
        async def __call__(self, orch: Any, seen_cache: Any) -> None:
            # What the dashboard held while the handshake ran.
            seen.append(orch.dashboard_state.slack_client)
            orch._slack_enabled = False
            orch.slack = None  # the real early return does this too

    result = await _run(orch, _Rejecting())

    assert seen == [None]  # the old client was gone BEFORE validation
    assert result == {"connected": False, "connect_error": "enterprise_validation_failed"}
    assert orch.dashboard_state.slack_client is None
    assert orch.dashboard_state.slack_socket_connected is False


@pytest.mark.asyncio
async def test_failed_handshake_publishes_no_client() -> None:
    """A client whose socket did not connect is not offered to the dashboard."""
    orch = _orch()

    async def _connect(self: Any) -> bool:
        self._slack_connect_error = "invalid_auth"
        return False

    await _run(orch, _Recorder(), connect=_connect)

    assert orch.slack is not None  # the orchestrator keeps its own handle, as at boot
    assert orch.dashboard_state.slack_client is None


@pytest.mark.asyncio
async def test_owner_change_moves_the_dashboard_owner() -> None:
    """``DashboardState.owner_id`` is the owner-only handlers' subject.

    It is set once from the boot-time owner; a reconnect that hoists a new
    owner without moving it would leave the former owner authorised.
    """
    orch = _orch()
    assert orch.dashboard_state.owner_id == "U0FORMEROWNER"

    await _run(orch, _Recorder())

    assert orch.dashboard_state.owner_id == NEW_CREDS[CRED_OWNER_ID]


@pytest.mark.asyncio
async def test_cleared_owner_clears_the_dashboard_owner() -> None:
    creds = {k: v for k, v in NEW_CREDS.items() if k != CRED_OWNER_ID}
    orch = _orch(creds)

    await _run(orch, _Recorder())

    assert orch.dashboard_state.owner_id == ""
    assert orch.dashboard_state.slack_client is None


@pytest.mark.asyncio
async def test_connected_reconnect_runs_the_tracked_channel_probe() -> None:
    """Boot warns about a tracked private channel the install cannot read; a
    reconnect that connects is the same moment and must warn the same way."""
    orch = _orch()
    orch._tracking_channels = {"C0TRACKED"}
    probe = AsyncMock()

    with patch("kiro_crew.slack.gateway.warn_unreadable_tracked_channels", probe):
        await _run(orch, _Recorder())
        await asyncio.sleep(0)  # let the fire-and-forget task start

    probe.assert_awaited_once()
    args, kwargs = probe.await_args
    assert args[0] is orch.slack
    assert args[1] == {"C0TRACKED"}
    assert kwargs["notify"] is orch.dashboard_state.notify


@pytest.mark.asyncio
async def test_failed_reconnect_skips_the_tracked_channel_probe() -> None:
    orch = _orch()
    orch._tracking_channels = {"C0TRACKED"}
    probe = AsyncMock()

    async def _connect(self: Any) -> bool:
        self._slack_connect_error = "invalid_auth"
        return False

    with patch("kiro_crew.slack.gateway.warn_unreadable_tracked_channels", probe):
        await _run(orch, _Recorder(), connect=_connect)
        await asyncio.sleep(0)

    probe.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_handshake() -> None:
    """A double-click must not race two Socket Mode handshakes."""
    orch = _orch()
    gate = asyncio.Event()
    rec = _Recorder(hold=gate)
    p_init, p_client, p_connect = _patches(rec)
    with p_init, p_client, p_connect:
        first = asyncio.create_task(orch.reconnect_slack())
        await asyncio.sleep(0)  # first attempt is now parked inside init_socket_mode
        second = asyncio.create_task(orch.reconnect_slack())
        await asyncio.sleep(0)
        gate.set()
        results = await asyncio.gather(first, second)

    assert len(rec.calls) == 1
    assert results[0] == results[1] == {"connected": True, "connect_error": ""}
    assert orch._cfg.load_credentials.call_count == 1
    # The slot is released, so a later click starts a fresh attempt.
    assert orch._slack_reconnect_task is None


@pytest.mark.asyncio
async def test_cancelled_caller_does_not_cancel_the_shared_attempt() -> None:
    orch = _orch()
    gate = asyncio.Event()
    rec = _Recorder(hold=gate)
    p_init, p_client, p_connect = _patches(rec)
    with p_init, p_client, p_connect:
        first = asyncio.create_task(orch.reconnect_slack())
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        gate.set()
        second = await orch.reconnect_slack()

    assert len(rec.calls) == 1  # the in-flight attempt finished and was reused
    assert second["connected"] is True


def test_gateway_wires_the_callback_onto_dashboard_state() -> None:
    """Source pin: the dashboard init publishes reconnect_slack for the route."""
    import inspect

    from kiro_crew.slack.gateway import GatewayOrchestrator

    src = inspect.getsource(GatewayOrchestrator._init_dashboard)
    assert "self.dashboard_state._slack_reconnect = self.reconnect_slack" in src


# ── route ────────────────────────────────────────────────────────────────────


def _request(state: Any) -> web.Request:
    app = web.Application()
    app["state"] = state
    return make_mocked_request("POST", "/api/slack/reconnect", app=app)


@pytest.mark.asyncio
async def test_route_answers_the_config_get_shape() -> None:
    import kiro_crew.dashboard.handlers.messaging as mod

    state = MagicMock()
    state._slack_reconnect = AsyncMock(
        return_value={"connected": False, "connect_error": "invalid_auth"}
    )
    sel = MagicMock()
    with (
        patch.object(mod, "is_direct_local_request", lambda req: True),
        patch.object(mod, "_sel", lambda: sel),
    ):
        resp = await mod.api_slack_reconnect(_request(state))

    assert resp.status == 200
    assert resp.text is not None
    import json

    assert json.loads(resp.text) == {"connected": False, "connect_error": "invalid_auth"}
    state._slack_reconnect.assert_awaited_once_with()
    kw = sel.log_api_access.call_args.kwargs
    assert kw["operation"] == "slack.reconnect"
    assert kw["outcome"] == "failed"
    assert kw["error"] == "invalid_auth"


@pytest.mark.asyncio
async def test_route_denies_remote_sessions_like_the_put() -> None:
    import kiro_crew.dashboard.handlers.messaging as mod

    state = MagicMock()
    state._slack_reconnect = AsyncMock()
    with (
        patch.object(mod, "is_direct_local_request", lambda req: False),
        patch.object(mod, "_sel", lambda: MagicMock()),
    ):
        resp = await mod.api_slack_reconnect(_request(state))

    assert resp.status == 403
    state._slack_reconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_route_is_503_when_no_gateway_owns_a_socket() -> None:
    import kiro_crew.dashboard.handlers.messaging as mod

    state = MagicMock()
    state._slack_reconnect = None
    with (
        patch.object(mod, "is_direct_local_request", lambda req: True),
        patch.object(mod, "_sel", lambda: MagicMock()),
    ):
        resp = await mod.api_slack_reconnect(_request(state))

    assert resp.status == 503


@pytest.mark.asyncio
async def test_route_is_500_when_the_store_cannot_be_read() -> None:
    import kiro_crew.dashboard.handlers.messaging as mod

    state = MagicMock()
    state._slack_reconnect = AsyncMock(side_effect=OSError("store unreadable"))
    sel = MagicMock()
    with (
        patch.object(mod, "is_direct_local_request", lambda req: True),
        patch.object(mod, "_sel", lambda: sel),
    ):
        resp = await mod.api_slack_reconnect(_request(state))

    assert resp.status == 500
    assert sel.log_api_access.call_args.kwargs["outcome"] == "denied"


def test_route_is_registered_as_post_beside_the_put() -> None:
    from kiro_crew.dashboard import handlers
    from kiro_crew.dashboard.routes import messaging as routes

    app = web.Application()
    routes.register(app)
    reconnect = [
        r
        for r in app.router.routes()
        if r.resource is not None and r.resource.canonical == "/api/slack/reconnect"
    ]
    assert [r.method for r in reconnect] == ["POST"]
    assert reconnect[0].handler is handlers.api_slack_reconnect
