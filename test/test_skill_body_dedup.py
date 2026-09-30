"""A triggered skill's body is injected at most once per provider session.

With ``skills.max_triggered`` above 0, a matched skill's full body is injected
at most once per provider session: the first match sends the body, and a later
match in the same session sends the cheap pointer line ``inject_on_trigger:
false`` skills already use. A monitor loop that re-sends the same message
therefore does not re-send the same 8k–34k-char body every cycle — the copies
would add nothing, because the provider replays native history.

The record fails SAFE: whenever it is reset — a fresh session, the first turn
after a compaction, an agent switch, an edited skill body, or simply no
session key at all — the body re-injects rather than falling silent. Every test
here fails if a reset stops re-injecting, or if a repeat stops being demoted.
"""

from pathlib import Path
from unittest.mock import MagicMock

from kiro_crew.config.loader import KiroCrewConfig, SkillsConfig
from kiro_crew.context import ContextBuilder
from kiro_crew.memory import MemoryStore
from kiro_crew.skills import SkillsLoader

BODY_SENTINEL = "STEP ONE: pour the concrete before the rebar."
HINT_HEADER = "[Relevant skills for this message]"
SKILL_BLOCK = "[Skill: foundation]"
SESSION = "chat-session-abc"


def _write_skill(root: Path, name: str, *, body: str = BODY_SENTINEL) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Lay a foundation\ntriggers: zebra quokka\n---\n{body}",
        encoding="utf-8",
    )
    return d / "SKILL.md"


def _loader(skills_path: Path, *, cap: int = 3) -> SkillsLoader:
    return SkillsLoader(
        skills_path=skills_path,
        install_builtins=False,
        config=KiroCrewConfig(skills=SkillsConfig(max_triggered=cap)),
    )


def _builder(tmp_path: Path, skills: SkillsLoader) -> ContextBuilder:
    return ContextBuilder(memory=MemoryStore(workspace=tmp_path / "ws"), skills=skills)


def _send(builder: ContextBuilder, **kw: object) -> str:
    """Build one turn's message. Dedup records at build time, so a second call
    for the same session demotes without any separate confirmation step."""
    defaults: dict = {"is_new_session": False, "session_key": SESSION}
    defaults.update(kw)
    msg, _ = builder.build_message("zebra quokka", **defaults)  # type: ignore[arg-type]
    return msg


class TestSecondMatchInSessionDemotesToPointer:
    def test_first_send_injects_body_second_sends_pointer(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        path = _write_skill(skills, "foundation")
        builder = _builder(tmp_path, _loader(skills))

        first = _send(builder)
        assert BODY_SENTINEL in first
        assert SKILL_BLOCK in first
        assert HINT_HEADER not in first

        second = _send(builder)
        assert BODY_SENTINEL not in second
        assert SKILL_BLOCK not in second
        assert HINT_HEADER in second
        assert str(path) in second

    def test_third_and_later_matches_also_stay_demoted(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        _write_skill(skills, "foundation")
        builder = _builder(tmp_path, _loader(skills))

        _send(builder)
        for _ in range(4):
            assert BODY_SENTINEL not in _send(builder)


class TestNoSessionKeyNeverDedups:
    """A None session key keeps no record — today's always-send behaviour."""

    def test_every_match_injects_the_body(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        _write_skill(skills, "foundation")
        builder = _builder(tmp_path, _loader(skills))

        for _ in range(3):
            assert BODY_SENTINEL in _send(builder, session_key=None)


class TestUnlandedTurnFallsBackToPointerNotSilence:
    """A rare turn that builds but never lands demotes to a pointer, not silence.

    Recording at build time means such a turn's body is recorded, so the next
    match sends the pointer line instead of the body. That still tells the agent
    the skill applies — the same fail-safe an ``inject_on_trigger: false`` skill
    already relies on — rather than dropping the skill entirely.
    """

    def test_repeat_after_any_build_sends_the_pointer(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        path = _write_skill(skills, "foundation")
        builder = _builder(tmp_path, _loader(skills))

        assert BODY_SENTINEL in _send(builder)
        second = _send(builder)
        assert BODY_SENTINEL not in second
        assert HINT_HEADER in second  # the agent still learns the skill applies
        assert str(path) in second


class TestResetReinjects:
    def test_new_session_reinjects(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        _write_skill(skills, "foundation")
        builder = _builder(tmp_path, _loader(skills))

        _send(builder)
        assert BODY_SENTINEL not in _send(builder)
        # A fresh provider session does not hold the earlier body.
        assert BODY_SENTINEL in _send(builder, is_new_session=True)

    def test_first_turn_after_compaction_reinjects(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        _write_skill(skills, "foundation")
        builder = _builder(tmp_path, _loader(skills))

        _send(builder)
        assert BODY_SENTINEL not in _send(builder)
        # needs_reinjection is the post-compaction turn: the window was rebuilt.
        assert BODY_SENTINEL in _send(builder, needs_reinjection=True)

    def test_flag_on_a_no_match_turn_still_resets(self, tmp_path: Path) -> None:
        """A window-rebuild flag is one-shot and can land on a turn where no
        skill matches. The reset must fire on that turn anyway, so a later
        matching turn re-injects rather than demoting a body the rebuilt window
        never received."""
        skills = tmp_path / "skills"
        _write_skill(skills, "foundation")
        builder = _builder(tmp_path, _loader(skills))

        assert BODY_SENTINEL in _send(builder)
        assert BODY_SENTINEL not in _send(builder)

        # The compaction turn triggers no skill (its text does not match the
        # trigger), so it consumes needs_reinjection without touching the
        # skill-match branch. The record must still be cleared here.
        no_match, _ = builder.build_message(
            "unrelated message", is_new_session=False, session_key=SESSION, needs_reinjection=True
        )
        assert BODY_SENTINEL not in no_match  # nothing matched, so no body anyway

        # The next matching turn must re-inject: the window was rebuilt.
        assert BODY_SENTINEL in _send(builder)

    def test_flag_on_a_no_match_turn_clears_the_record(self, tmp_path: Path) -> None:
        """Same guarantee, asserted at the record: the no-match rebuild turn
        empties the session's stored hashes."""
        skills = tmp_path / "skills"
        _write_skill(skills, "foundation")
        builder = _builder(tmp_path, _loader(skills))

        _send(builder)
        key = builder._cap_memo_key(SESSION)
        assert builder._sent_skill_bodies.get(key)  # recorded

        builder.build_message(
            "unrelated message", is_new_session=False, session_key=SESSION, needs_reinjection=True
        )
        # The rebuild turn cleared the record even though no skill matched.
        assert not builder._sent_skill_bodies.get(key)

    def test_agent_switch_reinjects(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        # The default agent takes the body path; a custom agent skips triggered
        # skills, so switch between two names that both inject: None and the
        # canonical crew agent are both non-custom.
        _write_skill(skills, "foundation")
        builder = _builder(tmp_path, _loader(skills))

        assert BODY_SENTINEL in _send(builder, agent=None)
        assert BODY_SENTINEL not in _send(builder, agent=None)
        # Same session key, different agent: the window is a different one.
        assert BODY_SENTINEL in _send(builder, agent="kirocrew")

    def test_edited_body_reinjects_on_hash_change(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        path = _write_skill(skills, "foundation")
        builder = _builder(tmp_path, _loader(skills))

        assert BODY_SENTINEL in _send(builder)
        assert BODY_SENTINEL not in _send(builder)

        edited = "STEP ONE: cure the slab a full week."
        path.write_text(
            path.read_text(encoding="utf-8").replace(BODY_SENTINEL, edited),
            encoding="utf-8",
        )
        # A new hash means the agent has never seen this body: send it.
        assert edited in _send(builder)


class TestSessionsAreIndependent:
    def test_one_session_dedup_does_not_silence_another(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        _write_skill(skills, "foundation")
        builder = _builder(tmp_path, _loader(skills))

        assert BODY_SENTINEL in _send(builder, session_key="chat-a")
        assert BODY_SENTINEL not in _send(builder, session_key="chat-a")
        # A never-seen session still gets the body.
        assert BODY_SENTINEL in _send(builder, session_key="chat-b")


class TestConfinedSkillNeverDemoted:
    """A confined project skill has no pointer form, so it always re-injects."""

    def test_confined_skill_body_repeats_every_turn(self, tmp_path: Path) -> None:
        loader = MagicMock()
        loader.get_triggered_skills.return_value = ["proj-skill"]
        loader.split_triggered.return_value = (["proj-skill"], [])
        # Confinement is what forbids the pointer path; mark it confined.
        loader.confined_triggered.return_value = {"proj-skill"}
        loader.load_skill.return_value = "confined body ONE"
        loader.strip_frontmatter.return_value = "confined body ONE"
        loader.trigger_hint.return_value = ""
        builder = ContextBuilder(memory=MemoryStore(workspace=tmp_path / "ws"), skills=loader)

        first, _ = builder.build_message("trigger", is_new_session=False, session_key=SESSION)
        second, _ = builder.build_message("trigger", is_new_session=False, session_key=SESSION)

        # Demoting it would drop it entirely (no body, no pointer); instead it
        # re-injects both times.
        assert "confined body ONE" in first
        assert "confined body ONE" in second
        assert "[Skill: proj-skill]" in second


class TestRecordIsBounded:
    def test_session_record_evicts_oldest_past_the_bound(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        _write_skill(skills, "foundation")
        builder = _builder(tmp_path, _loader(skills))
        builder._SENT_SKILL_BODY_SESSIONS = 3  # type: ignore[misc]

        # First session sends the body and is recorded.
        assert BODY_SENTINEL in _send(builder, session_key="s0")
        assert BODY_SENTINEL not in _send(builder, session_key="s0")

        # Fill past the bound with newer sessions; s0 is the oldest and evicts.
        for i in range(1, 5):
            _send(builder, session_key=f"s{i}")

        assert len(builder._sent_skill_bodies) <= 3
        # Keys are digests of the session key, so check the digest is gone.
        assert builder._cap_memo_key("s0") not in builder._sent_skill_bodies
        assert len(builder._sent_skill_agents) <= 3
        # Evicted -> its next match re-injects (fail-safe direction).
        assert BODY_SENTINEL in _send(builder, session_key="s0")

    def test_per_session_entry_count_is_bounded(self, tmp_path: Path) -> None:
        """One session matching many distinct skills cannot grow without limit."""
        skills = tmp_path / "skills"
        # Every skill shares the same trigger word, so one message matches all.
        for i in range(6):
            _write_skill(skills, f"skill{i}", body=f"BODY NUMBER {i}")
        builder = _builder(tmp_path, _loader(skills, cap=6))
        builder._SENT_SKILL_BODY_ENTRIES = 3  # type: ignore[misc]

        _send(builder, session_key="cap-test")
        inner = builder._sent_skill_bodies[builder._cap_memo_key("cap-test")]
        assert len(inner) <= 3

    def test_keys_are_digested_not_raw_session_keys(self, tmp_path: Path) -> None:
        """A caller-supplied key is stored as a fixed-size digest, not verbatim."""
        skills = tmp_path / "skills"
        _write_skill(skills, "foundation")
        builder = _builder(tmp_path, _loader(skills))

        raw = "session/with:arbitrary\nlength-" + "x" * 500
        _send(builder, session_key=raw)

        assert raw not in builder._sent_skill_bodies
        assert builder._cap_memo_key(raw) in builder._sent_skill_bodies

    def test_inner_skill_keys_are_digested_not_raw(self, tmp_path: Path) -> None:
        """A skill key is stored as a fixed-size digest, so a long key cannot
        stay retained at full length and defeat the byte bound."""
        long_name = "s" + "k" * 300
        loader = MagicMock()
        loader.get_triggered_skills.return_value = [long_name]
        loader.split_triggered.return_value = ([long_name], [])
        loader.confined_triggered.return_value = set()
        loader.load_skill.return_value = "some body"
        loader.strip_frontmatter.return_value = "some body"
        loader.trigger_hint.return_value = ""
        builder = ContextBuilder(memory=MemoryStore(workspace=tmp_path / "ws"), skills=loader)

        builder.build_message("trigger", is_new_session=False, session_key="digest-test")
        inner = builder._sent_skill_bodies[builder._cap_memo_key("digest-test")]
        assert long_name not in inner
        assert builder._cap_memo_key(long_name) in inner


class TestDeliveryAuditReflectsDedup:
    """The delivery audit names what the prompt ACTUALLY carries after dedup.

    The matcher's ``skill_trigger`` row records the frontmatter-level split at
    match time, before this dedup runs, so a demoted body is still named a
    delivered body there. A ``skill_delivery`` correction row is emitted after
    dedup — and only when a demotion diverged from that claim — so an auditor is
    never told a demoted body was in the prompt.
    """

    def _delivery_rows(self, sel_mock: MagicMock) -> list[dict]:
        return [
            c.kwargs["metadata"]
            for c in sel_mock.log_tool_invocation.call_args_list
            if c.kwargs.get("tool_name") == "skill_delivery"
        ]

    def test_no_correction_row_on_first_send(self, tmp_path: Path, monkeypatch) -> None:
        skills = tmp_path / "skills"
        _write_skill(skills, "foundation")
        builder = _builder(tmp_path, _loader(skills))
        sel_mock = MagicMock()
        monkeypatch.setattr("kiro_crew.context.sel", lambda: sel_mock)

        _send(builder)
        # Nothing demoted -> the matcher row is already correct, no extra row.
        assert self._delivery_rows(sel_mock) == []

    def test_correction_row_names_demoted_skill_as_pointer(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        skills = tmp_path / "skills"
        _write_skill(skills, "foundation")
        builder = _builder(tmp_path, _loader(skills))
        sel_mock = MagicMock()
        monkeypatch.setattr("kiro_crew.context.sel", lambda: sel_mock)

        _send(builder)  # body delivered, recorded
        _send(builder)  # foundation demoted to pointer this turn

        rows = self._delivery_rows(sel_mock)
        assert len(rows) == 1
        assert rows[0]["demoted"] == "foundation"
        assert rows[0]["pointers"] == "foundation"
        assert rows[0]["bodies"] == ""
