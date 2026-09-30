"""An approved launch that pins the gateway's own interpreter survives an upgrade.

``apps/bridges.py`` rewrites a manifest's bare ``python3`` (and the ``kirocrew``
host CLI) to ``sys.executable``, and inserts the ``deps_boot`` shim by absolute
path. Both live inside the install prefix, which the desktop and CLI installers
lay out under a VERSIONED directory -- so before this rule the approval hash
recorded ``.../0.8.0.2/payload/.../python3.12`` and every upgrade produced a
``changed_needs_reapproval`` refusal for a program that had not changed.

``hash_command`` now encodes a token inside ``sys.prefix`` install-relative.
These tests pin: the same hash across two prefixes, a real change still
changing it, no effect on paths outside the prefix (byte-for-byte the old
digest), the path-boundary and root-prefix guards, and that no literal token
sequence can collide with the encoding.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from kiro_crew.mcp_gateway import hashing, launch_approval
from kiro_crew.mcp_gateway.hashing import hash_command

launch_token_bytes = getattr(hashing, "launch_token_bytes", None)


def _pin_prefix(monkeypatch: pytest.MonkeyPatch, prefix: Path) -> None:
    monkeypatch.setattr(hashing.sys, "prefix", str(prefix))
    monkeypatch.setattr(hashing.sys, "exec_prefix", str(prefix))


def _legacy_hash(command: str, args: list[str]) -> str:
    """The pre-rule digest: UTF-8 tokens joined by ``\\0``."""
    h = hashlib.sha256()
    h.update(command.encode("utf-8"))
    h.update(b"\0")
    for a in args:
        h.update(a.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def test_interpreter_hash_is_stable_across_versioned_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = tmp_path / "kirocrew" / "0.8.0.2" / "payload"
    new = tmp_path / "kirocrew" / "0.8.0.3" / "payload"
    args = ["-s", str(tmp_path / "apps" / "demo" / "server.py")]

    _pin_prefix(monkeypatch, old)
    before = hash_command(str(old / "bin" / "python3.12"), args)
    _pin_prefix(monkeypatch, new)
    after = hash_command(str(new / "bin" / "python3.12"), args)

    assert before == after
    # A different interpreter under the SAME prefix is still a different launch.
    assert hash_command(str(new / "bin" / "python3.13"), args) != after
    # And an interpreter outside the install never matches the pinned one.
    assert hash_command(str(tmp_path / "other" / "bin" / "python3.12"), args) != after


def test_deps_boot_shim_argv_is_install_relative_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shim ``bridges.py`` inserts by absolute path moves with the prefix as well."""
    server = str(tmp_path / "apps" / "demo" / "server.py")
    deps = str(tmp_path / "apps" / "demo" / ".deps")
    digests = []
    for version in ("0.8.0.2", "0.8.0.3"):
        prefix = tmp_path / version
        _pin_prefix(monkeypatch, prefix)
        shim = (
            prefix / "lib" / "python3.12" / "site-packages" / "kiro_crew" / "apps" / "deps_boot.py"
        )
        digests.append(hash_command(str(prefix / "bin" / "python3.12"), [str(shim), deps, server]))
    assert digests[0] == digests[1]


def test_paths_outside_the_prefix_keep_the_legacy_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch that names nothing under the install hashes byte-for-byte as before."""
    _pin_prefix(monkeypatch, tmp_path / "prefix")
    for command, args in (
        ("/usr/bin/node", ["server.js", "--stdio"]),
        (str(tmp_path / "apps" / "x" / ".venv" / "bin" / "python"), ["-m", "srv"]),
        ("npx", ["-y", "@scope/pkg"]),
        ("", []),
    ):
        assert hash_command(command, args) == _legacy_hash(command, args)


def test_prefix_match_is_on_a_path_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/opt/kc2/bin/python`` is not inside ``/opt/kc``."""
    root = tmp_path / "kc"
    _pin_prefix(monkeypatch, root)
    sibling = str(tmp_path / "kc2" / "bin" / "python")
    assert launch_token_bytes(sibling) == sibling.encode("utf-8")
    inside = str(root / "bin" / "python")
    assert launch_token_bytes(inside).startswith(b"\xffkirocrew-install:prefix\xff")
    # The prefix itself, and a trailing-separator spelling of it, both match.
    assert launch_token_bytes(str(root)) == b"\xffkirocrew-install:prefix\xff"
    monkeypatch.setattr(hashing.sys, "prefix", str(root) + os.sep)
    assert launch_token_bytes(inside).startswith(b"\xffkirocrew-install:prefix\xff")


def test_split_prefix_and_exec_prefix_are_distinct_launches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``<prefix>/bin/python`` and ``<exec_prefix>/bin/python`` never share a digest."""
    monkeypatch.setattr(hashing.sys, "prefix", str(tmp_path / "share"))
    monkeypatch.setattr(hashing.sys, "exec_prefix", str(tmp_path / "lib"))
    args = ["-m", "srv"]
    under_prefix = hash_command(str(tmp_path / "share" / "bin" / "python3"), args)
    under_exec = hash_command(str(tmp_path / "lib" / "bin" / "python3"), args)
    assert under_prefix != under_exec
    assert launch_token_bytes(str(tmp_path / "lib" / "bin" / "python3")) == (
        b"\xffkirocrew-install:exec-prefix\xffbin" + os.sep.encode() + b"python3"
    )
    # Both roots still fold their own version directory away.
    monkeypatch.setattr(hashing.sys, "exec_prefix", str(tmp_path / "lib2"))
    assert hash_command(str(tmp_path / "lib2" / "bin" / "python3"), args) == under_exec
    # Equal roots collapse to one role, so a venv encodes exactly one way.
    monkeypatch.setattr(hashing.sys, "exec_prefix", str(tmp_path / "share"))
    assert hashing._install_roots() == ((b"prefix", os.path.normcase(str(tmp_path / "share"))),)


def test_filesystem_root_prefix_canonicalises_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    drive = os.path.splitdrive(os.path.abspath(os.sep))[0]
    for root in (os.sep, drive + os.sep if drive else os.sep, ""):
        monkeypatch.setattr(hashing.sys, "prefix", root)
        monkeypatch.setattr(hashing.sys, "exec_prefix", root)
        assert hashing._install_roots() == ()
        assert launch_token_bytes("/usr/bin/python3") == b"/usr/bin/python3"


def test_no_literal_argv_can_spell_the_install_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``0xFF`` is not UTF-8, so no typed token sequence reproduces the digest."""
    prefix = tmp_path / "prefix"
    _pin_prefix(monkeypatch, prefix)
    approved = hash_command(str(prefix / "bin" / "python3"), ["-m", "srv"])
    # The obvious forgeries: the tag spelled as a token, split across the
    # separator, or carried in the command slot with the rest as args.
    forgeries = (
        ("\xffkirocrew-install:prefix\xffbin/python3", ["-m", "srv"]),
        ("", ["kirocrew-install", "bin/python3", "-m", "srv"]),
        ("kirocrew-install:prefix", ["bin/python3", "-m", "srv"]),
        ("\x00kirocrew-install:prefix\x00bin/python3", ["-m", "srv"]),
    )
    for command, args in forgeries:
        assert hash_command(command, args) != approved
    # And the encoding itself is not a spawnable spelling of anything.
    token = launch_token_bytes(str(prefix / "bin" / "python3"))
    with pytest.raises(UnicodeDecodeError):
        token.decode("utf-8")


def test_approval_recorded_under_one_release_admits_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the approval store: approve on 0.8.0.2, resolve on 0.8.0.3."""
    old = tmp_path / "kirocrew" / "0.8.0.2" / "payload"
    new = tmp_path / "kirocrew" / "0.8.0.3" / "payload"
    args = [str(tmp_path / "apps" / "demo" / "server.py")]
    env_hash = launch_approval.env_fingerprint({})
    store = tmp_path / "approvals.json"

    _pin_prefix(monkeypatch, old)
    old_cmd = str(old / "bin" / "python3.12")
    launch_approval.approve(
        {
            "demo": [
                launch_approval.ResolvedLaunch(
                    launch_approval.launch_fingerprint(old_cmd, args, {}),
                    old_cmd,
                    tuple(args),
                    frozenset({env_hash}),
                )
            ]
        },
        path=store,
    )

    _pin_prefix(monkeypatch, new)
    new_cmd = str(new / "bin" / "python3.12")
    approvals = launch_approval.load_approvals(store)
    assert approvals.admits_command("DEMO", hash_command(new_cmd, args))
    assert approvals.admits_launch("DEMO", hash_command(new_cmd, args), env_hash)
    # A launch that really changed is still refused.
    assert not approvals.admits_command("DEMO", hash_command(new_cmd, args + ["--debug"]))
    assert not approvals.admits_command(
        "DEMO", hash_command(str(tmp_path / "elsewhere" / "python3.12"), args)
    )
