"""The crew main dashboard's gate: the template and the contract name the same fields.

``mypy src/kiro_crew/`` checks both ends of :func:`build_crew_main` -- a
:class:`~kiro_crew.crew_main_contract.CrewMainReads` in, a
:class:`~kiro_crew.crew_main_contract.CrewMainDerived` out, required keys and all. What it
cannot see is ``crew_main.html``, so the HTML end needs a gate of its own. This is that
gate, built after ``test_pipeline_board_contract_parity``.

Scope, stated honestly, because a reader will assume more than is here. The parity case
asserts a field EXISTS on both sides. It does NOT assert the value is right, or that a
person will see it where the template puts it. The cases after it cover what a text gate
cannot: that no branch of the provider can leave a field unanswered, that absence is
three-state in WORDS rather than a zero, that no value is a percentage, and that a model
cannot reach a derived field.

Two attacks an earlier draft of the extractor allowed, each with a case here: a field
named only inside an HTML COMMENT counting as read (``crew_main.html`` opens with a long
comment that names several field concepts in prose), and a broken extractor returning an
empty set -- which would read as "the contract is over-specified" rather than "the gate is
broken". The precondition cases run first for that reason.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from kiro_crew.crew_main_contract import (
    CONTRACT_VERSION,
    DERIVED_FIELDS,
    EMPTY_JUDGMENT,
    FOLD_UNREADABLE,
    HOST_FIELDS,
    JUDGMENT_FIELDS,
    JUDGMENT_TEXT_LIMIT,
    NO_PUBLISHED_VIEW,
    NOT_RECORDED,
    SENTENCES_OFF,
    SENTENCES_ON,
    SENTENCES_OVER_BUDGET,
    UNREADABLE,
    CrewMainData,
    CrewMainReads,
    build_crew_main,
    card_data_payload,
    merge_crew_main,
    read_crew_main_template,
    template_fields,
    validate_judgment,
)
from kiro_crew.dashboard import card_lifecycle
from kiro_crew.dashboard.card_lifecycle import (
    _JUDGMENT_PROMPT,
    _redact,
    _redact_judgment,
    is_root_session,
)
from kiro_crew.dashboard.dynamic_cards import (
    MAX_DATA_BYTES,
    MAX_HTML_BYTES,
    normalize_card,
)

FOLD_NAMES = ("status", "usage", "approvals", "work", "panel")


async def _settle(lifecycle) -> None:
    """Await the derived worker this lifecycle started, so nothing outlives the case.

    ``notify`` starts a task. A case that returns while it is pending leaves it to be
    destroyed on a loop teardown, and the warning that produces is the polite version of
    the real problem: work belonging to one test running under another's conditions.
    """
    worker = lifecycle._derived_worker
    if worker is not None:
        await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 2)


@pytest.fixture(autouse=True)
def _isolate_session_tree():
    """Drop the process's session-tree projection around every case in this file.

    ``is_root_session`` reads that projection, which is a process singleton keyed to one
    store. Without this, a case here leaves a fold behind that the NEXT file's cases read
    as their own -- which is how this file turned a passing case in
    ``test_dynamic_dashboard_cards.py`` red purely by running before it. Reset on the way
    in AND on the way out, so neither direction of that leak survives.
    """
    from kiro_crew.crew_log import session_tree_projection

    session_tree_projection.reset_for_tests()
    yield
    session_tree_projection.reset_for_tests()


class _Slot:
    """The slot fields the panel's gates and publisher actually read, and no others."""

    def __init__(self, key: str, **kwargs: object) -> None:
        self.key = key
        self.messages: list[dict] = kwargs.pop("messages", [{"role": "user", "content": "hi"}])  # type: ignore[assignment]
        self._created_by = kwargs.pop("created_by", "")
        self.is_remote = kwargs.pop("is_remote", False)
        self.executor = kwargs.pop("executor", "")
        self.memory_mode = kwargs.pop("memory_mode", "persistent")
        self._dashboard_card_identity = kwargs.pop("identity", "id-" + key)
        self.linked_session_key = ""
        for name, value in kwargs.items():
            setattr(self, name, value)


def _slot(key: str, **kwargs: object) -> _Slot:
    return _Slot(key, **kwargs)


class _Log:
    """The transcript reads ``_generate`` makes, and a generation a case can move.

    Only the cases that run a whole generation need one; the rest publish through
    ``_write_card`` and never reach a transcript, which is why ``conversation_log`` is
    ``None`` by default rather than always this.
    """

    def __init__(self, slot: _Slot) -> None:
        self._slot = slot
        self.generation = 0

    @contextmanager
    def publication_hold(self, key: str):
        yield

    def session_mtime(self, key: str) -> int:
        return 1

    def rotation_generation(self, key: str) -> int:
        return self.generation

    def chained_keys(self, key: str) -> list[str]:
        return [key]

    def derive_recent(self, key: str, max_messages: int, roles: object = None) -> list[dict]:
        return self._slot.messages[-max_messages:]


class _State:
    """Only what CardLifecycle touches: the slot table and a broadcast sink."""

    def __init__(self, *slots: _Slot, log: object = None) -> None:
        self._slots = {slot.key: slot for slot in slots}
        self._background_tasks: set = set()
        self.conversation_log = log
        self.sessions = object()
        self.frames: list[tuple[str, object]] = []

    def flush_slot_now(self, slot: object) -> None:
        return None

    def broadcast_ws_owners(self, kind: str, payload: object) -> None:
        self.frames.append((kind, payload))


def _lifecycle(*slots: _Slot, enabled: bool = False, log: object = None):
    from kiro_crew.dashboard.card_lifecycle import CardLifecycle

    return CardLifecycle(_State(*slots, log=log), enabled=enabled)


def _reads(**overrides: object) -> CrewMainReads:
    """A board where every fold read and every value is present."""
    base: dict[str, object] = {
        "status": {
            "lifecycle": "open",
            "turn_open": True,
            "turns_completed": 12,
            "turns_refused": 1,
            "entries": 480,
            "agent": "kirocrew-conductor",
            "model": "a-model",
            "last_time": int(time.time() * 1000),
        },
        "usage": {
            "credits": 3.5,
            "credits_by_source": {"subagent": {"credits": 1.25, "reported": 4}},
            "tokens": {"total": 1215},
        },
        "approvals": {
            "requested": 9,
            "decided": 7,
            "pending": 2,
            "by_decision": {"allow": 6, "deny": 1},
        },
        "work": {
            "items": [
                {"state": "open", "status": "progress"},
                {"state": "open", "status": "blocked"},
                {"state": "open", "status": "question"},
                {"state": "accepted", "status": "done"},
                {"state": "rejected", "status": None},
            ],
            "omitted": 2,
        },
        "panel": {"template": "a-template", "title": "Fleet board", "publishes": 7},
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


# --------------------------------------------------------------------------
# preconditions -- a broken extractor must not read as a clean contract
# --------------------------------------------------------------------------


def test_the_extractor_finds_fields_at_all() -> None:
    """An empty set would make every parity case below pass vacuously."""
    assert len(template_fields(read_crew_main_template())) == len(CrewMainData.__annotations__)
    assert len(CrewMainData.__annotations__) > 1


def test_the_extractor_ignores_a_field_named_only_in_a_comment() -> None:
    """The real template opens with a comment that names field concepts in prose."""
    markup = '<!-- data-dashboard-field="ghost" --><span data-dashboard-field="real"></span>'
    assert template_fields(markup) == {"real"}


def test_the_extractor_ignores_a_field_with_no_name() -> None:
    assert template_fields("<span data-dashboard-field></span>") == frozenset()


# --------------------------------------------------------------------------
# the parity gate
# --------------------------------------------------------------------------


def test_the_template_and_the_contract_name_the_same_fields() -> None:
    fields = template_fields(read_crew_main_template())
    declared = frozenset(CrewMainData.__annotations__)
    assert fields - declared == frozenset(), "the template paints a field no type declares"
    assert declared - fields == frozenset(), "the contract declares a field nothing paints"


def test_every_contract_field_has_exactly_one_writer() -> None:
    """Three writers, disjoint, and together exactly the contract."""
    halves = (DERIVED_FIELDS, HOST_FIELDS, JUDGMENT_FIELDS)
    for left in range(len(halves)):
        for right in range(left + 1, len(halves)):
            assert halves[left] & halves[right] == frozenset()
    assert DERIVED_FIELDS | HOST_FIELDS | JUDGMENT_FIELDS == frozenset(CrewMainData.__annotations__)
    assert JUDGMENT_FIELDS == {"lede", "you", "notes"}
    # The host writes about the CARD (are the sentences on), never about the session.
    assert HOST_FIELDS == {"sentences"}
    assert CONTRACT_VERSION == 1


def test_the_template_is_inert() -> None:
    """No script, no control, no navigation, no remote byte -- read off the markup.

    The host strips these too, but a template that needs stripping is a template one
    render-path change away from shipping them. This asserts the file itself is clean.
    """
    # Comments STRIPPED first, the same precaution the parity extractor takes: the
    # template's opening comment explains in prose which CSS the host removes, and prose
    # naming a forbidden token is documentation rather than a declaration.
    markup = re.sub(r"<!--.*?-->", " ", read_crew_main_template(), flags=re.S).lower()
    assert "<div" in markup, "comment stripping ate the markup"
    for token in ("<script", "<form", "<button", "<input", "<select", "<textarea", "<iframe"):
        assert token not in markup, f"the template carries {token}"
    for token in ("href=", "src=", "srcset=", "onclick", "javascript:", "@import", "@font-face"):
        assert token not in markup, f"the template carries {token}"
    # CSS generated text is stripped by the host in card mode, so a label declared that
    # way would vanish from the page AND from the backend's text scan of it.
    for token in ("content:", "list-style", "quotes:", "text-emphasis"):
        assert token not in markup, f"the template declares {token}, which the host strips"


def test_the_template_says_nothing_on_it_is_a_link() -> None:
    """A ruling: if something reads as a link, the card says in words that it is not."""
    assert "link" in read_crew_main_template().lower()


# --------------------------------------------------------------------------
# the provider is TOTAL: no branch can leave a field unanswered
# --------------------------------------------------------------------------


@pytest.mark.parametrize("unreadable", [(), *[(name,) for name in FOLD_NAMES], FOLD_NAMES])
def test_every_field_is_answered_whatever_could_not_be_read(unreadable: tuple[str, ...]) -> None:
    """Each fold failing alone, all of them failing, and none -- every field present.

    This is the case that would have caught a section helper whose early return dropped
    a key: an absent key binds nothing and the template paints an EMPTY cell, which a
    reader cannot tell from a recorded zero.
    """
    derived = build_crew_main(_reads(**{name: FOLD_UNREADABLE for name in unreadable}))
    assert frozenset(derived) == DERIVED_FIELDS
    for field, value in derived.items():
        assert isinstance(value, str) and value.strip(), f"{field} answered nothing"


@pytest.mark.parametrize(
    "reads",
    [
        pytest.param(_reads(**{name: {} for name in FOLD_NAMES}), id="folds-read-but-empty"),
        pytest.param(_reads(work={"items": []}), id="board-with-no-items"),
        pytest.param(_reads(work={}), id="board-with-no-item-list"),
        pytest.param(_reads(usage={"credits_by_source": {}}), id="no-credit-split"),
        pytest.param(_reads(panel={"template": ""}), id="nothing-published"),
        pytest.param(_reads(status={"lifecycle": "unknown"}), id="log-no-longer-says"),
    ],
)
def test_every_field_is_answered_for_a_thin_fold(reads: CrewMainReads) -> None:
    derived = build_crew_main(reads)
    assert frozenset(derived) == DERIVED_FIELDS
    for field, value in derived.items():
        assert isinstance(value, str) and value.strip(), f"{field} answered nothing"


def test_a_damaged_value_costs_one_field_and_not_the_panel() -> None:
    """Bytes off a file the reader does not control must not raise.

    Each of these is a value a damaged or planted line can carry: a non-finite credit
    total, a bool where a count belongs, and a negative count. The line stays on disk
    and nothing rewrites it, so a raise here would be permanent for that crew.
    """
    derived = build_crew_main(
        _reads(
            status={"lifecycle": "open", "turns_completed": True},
            usage={"credits": float("nan"), "tokens": {"total": -5}},
            approvals={"pending": True, "requested": 9},
        )
    )
    assert derived["turns"] == NOT_RECORDED
    assert derived["tokens"] == NOT_RECORDED
    assert derived["approvals_open"] == NOT_RECORDED
    assert "nan" not in derived["credits"].lower()


@pytest.mark.parametrize(
    ("stamp", "raises"),
    [
        pytest.param(10**20, "OSError", id="errno-value-too-large"),
        pytest.param(-(10**20), "OSError", id="negative-value-too-large"),
        pytest.param(10**30, "OverflowError", id="past-platform-time_t"),
        pytest.param(float("inf"), "OverflowError", id="infinite"),
    ],
)
def test_a_damaged_stamp_costs_one_field_whichever_way_it_breaks(stamp: float, raises: str) -> None:
    """One case per exception ``fromtimestamp`` raises, because they are not one branch.

    A single value exercises ONE of them: ``10**20`` raises ``OSError`` here and
    ``10**30`` raises ``OverflowError``, so a catch narrowed to either alone still passes
    a suite that only tried the other. (``ValueError`` is the third spelling, raised for a
    year out of range on platforms whose ``fromtimestamp`` checks that instead; it is kept
    in the catch and is not reproducible on this one.)
    """
    derived = build_crew_main(_reads(status={"lifecycle": "open", "last_time": stamp}))
    assert derived["last_activity"] == NOT_RECORDED


# --------------------------------------------------------------------------
# absence is three-state, in words
# --------------------------------------------------------------------------


def test_an_unread_fold_and_an_empty_one_read_differently() -> None:
    """The distinction the whole contract rests on: unknown is not the same as none."""
    unread = build_crew_main(_reads(**{name: FOLD_UNREADABLE for name in FOLD_NAMES}))
    empty = build_crew_main(_reads(**{name: {} for name in FOLD_NAMES}))
    assert set(unread.values()) == {UNREADABLE}
    assert UNREADABLE not in set(empty.values())
    assert empty["state"] == NOT_RECORDED
    assert empty["published_view"] == NO_PUBLISHED_VIEW
    for field in DERIVED_FIELDS:
        assert unread[field] != empty[field], f"{field} cannot tell unknown from none"


def test_no_value_is_ever_a_bare_number_or_a_zero() -> None:
    """A bare count invites the reader to supply the total, and they supply a wrong one."""
    for reads in (_reads(), _reads(**{name: {} for name in FOLD_NAMES})):
        for field, value in build_crew_main(reads).items():
            assert not value.strip().isdigit(), f"{field} is a bare number"
            assert any(ch.isalpha() for ch in value), f"{field} carries no words"


def test_no_value_is_a_percentage() -> None:
    values = list(build_crew_main(_reads()).values())
    assert not [v for v in values if "%" in v or "percent" in v.lower()]


def test_every_count_with_a_denominator_states_it() -> None:
    """A ruling, asserted: the total is on the page, not in the reader's head."""
    derived = build_crew_main(_reads())
    for field in (
        "items_open",
        "items_progress",
        "items_blocked",
        "items_done",
        "items_question",
        "approvals_open",
    ):
        assert " of " in derived[field], f"{field} states a count with no denominator"


def test_a_worker_claim_is_not_reported_as_an_acceptance() -> None:
    """``items_done`` counts the conductor's ruling, and says the word.

    Two items report ``done`` and neither is accepted, so a provider that counted the
    worker's own status would say two.
    """
    derived = build_crew_main(
        _reads(
            work={
                "items": [
                    {"state": "open", "status": "done"},
                    {"state": "open", "status": "done"},
                    {"state": "accepted", "status": "done"},
                ],
                "omitted": 0,
            }
        )
    )
    assert derived["items_done"] == "1 of 3 items accepted"


def test_dropped_log_entries_are_never_added_into_a_board_total() -> None:
    """One dropped ENTRY is not one missing ITEM, so the two counts stay apart."""
    derived = build_crew_main(_reads(work={"items": [], "omitted": 4}))
    assert "4" in derived["board_omitted"]
    assert "4" not in derived["items_open"]


def test_the_floor_warning_is_derived_and_not_left_to_a_sentence() -> None:
    """The count carries its own consequence, so switching sentences off cannot drop it.

    A warning carried by a model sentence is one the opt-in can remove while the tally it
    qualifies stays, leaving a reader the confidence and not the reason to doubt it. A bare
    count here has the same effect, because the only place left to explain it is a sentence.
    """
    dropped = build_crew_main(_reads(work={"items": [], "omitted": 4}))["board_omitted"]
    assert "floor" in dropped, "the count does not say what it does to the other counts"
    clean = build_crew_main(_reads(work={"items": [], "omitted": 0}))["board_omitted"]
    # Zero is stated too: silence would make a complete board and an unreadable one alike.
    assert "complete" in clean
    assert clean != dropped


def test_an_unmetered_subagent_bucket_is_not_a_charge_of_zero() -> None:
    derived = build_crew_main(
        _reads(
            usage={
                "credits": 1.0,
                "credits_by_source": {"subagent": {"credits": 0.0, "reported": 0}},
            }
        )
    )
    assert derived["credits_subagents"] == "no sub-agent charge reported"


def test_a_credit_charge_is_shown_to_two_places() -> None:
    derived = build_crew_main(_reads(usage={"credits": 3.499}))
    assert derived["credits"] == "3.50 credits billed to this crew"


def test_a_credit_charge_is_never_shown_in_scientific_notation() -> None:
    derived = build_crew_main(_reads(usage={"credits": 0.00001}))
    assert "e-" not in derived["credits"]


def test_a_charge_too_small_for_two_places_is_described_not_rounded_to_zero() -> None:
    """Two places would print "0.00", and a spend field must never read as no spend.

    The distinction is against a charge of exactly zero, which is a real measurement and
    says so.
    """
    derived = build_crew_main(_reads(usage={"credits": 0.00001}))
    assert derived["credits"] == "less than 0.01 credits billed to this crew"
    zero = build_crew_main(_reads(usage={"credits": 0.0}))
    assert zero["credits"] == "0.00 credits billed to this crew"


# --------------------------------------------------------------------------
# the model cannot reach a number
# --------------------------------------------------------------------------


def test_a_model_field_can_never_overwrite_a_derived_one() -> None:
    """The merge names every field, so a judgment carrying a count cannot land.

    mypy rejects the extra key in a ``CrewMainJudgment`` literal, which is why this
    passes the hostile payload through ``validate_judgment`` -- the runtime door a model
    actually arrives at.
    """
    derived = build_crew_main(_reads())
    hostile = validate_judgment(
        {
            "lede": "ok",
            "credits": "999999 credits billed to this crew",
            "items_open": "0 of 0 items still open",
            "state": "closed",
        }
    )
    assert frozenset(hostile) == JUDGMENT_FIELDS
    merged = merge_crew_main(derived, {"sentences": SENTENCES_ON}, hostile)
    for field in DERIVED_FIELDS:
        assert merged[field] == derived[field], f"a model reached {field}"
    assert merged["lede"] == "ok"


def test_the_merge_ignores_a_derived_key_even_when_one_reaches_it() -> None:
    """The SECOND defence, pinned on its own.

    The case above proves ``validate_judgment`` strips a numeric key at the door, which
    means it never reaches the merge -- so it says nothing about whether the merge would
    honour one. Two defences, and a suite that only exercises the outer one passes a
    merge rewritten as ``{**derived, **judgment}``.

    mypy rejects the extra key in a ``CrewMainJudgment`` literal, which is the third
    defence and the reason for the ignore: this hands the merge the payload the type
    system forbids, to assert the runtime does not honour it either.
    """
    derived = build_crew_main(_reads())
    forged = {
        **EMPTY_JUDGMENT,
        "lede": "ok",
        "credits": "999999 credits billed to this crew",
        "items_open": "0 of 0 items still open",
        "state": "closed",
        "published_view": "a view that was never published",
    }
    merged = merge_crew_main(derived, {"sentences": SENTENCES_ON}, forged)  # type: ignore[arg-type]
    for field in DERIVED_FIELDS:
        assert merged[field] == derived[field], f"the merge honoured a forged {field}"
    assert merged["lede"] == "ok"
    assert frozenset(merged) == frozenset(CrewMainData.__annotations__)


def test_a_judgment_is_bounded_and_single_line() -> None:
    judgment = validate_judgment({"lede": "x" * 500, "you": "two\nlines  here", "notes": 17})
    assert len(judgment["lede"]) == JUDGMENT_TEXT_LIMIT
    assert judgment["you"] == "two lines here"
    assert judgment["notes"] == ""


@pytest.mark.parametrize("raw", [None, [], "a string", 17, {"lede": None}])
def test_a_malformed_model_reply_still_yields_three_empty_sentences(raw: object) -> None:
    """The panel publishes either way: the numbers must never wait on a model."""
    assert validate_judgment(raw) == EMPTY_JUDGMENT


def test_a_credential_in_one_sentence_empties_all_three() -> None:
    """Redacted prose is a sentence written about a value that is now gone.

    And the three sentences are ONE display boundary: they render one under another, so a
    reader shown the surviving two cannot tell whether a rule took the third or the model
    had nothing to say. So a removal takes all three.
    """
    secret = "AKIA" + "Q" * 16
    judgment = _redact_judgment(
        '{"lede": "The key is ' + secret + '", "you": "Rule on it.", "notes": ""}'
    )
    assert judgment == EMPTY_JUDGMENT


def test_the_judgment_prompt_asks_for_three_sentences_and_forbids_numbers() -> None:
    """A sentence carrying a count is a number the merge cannot catch."""
    prompt = _JUDGMENT_PROMPT
    for field in JUDGMENT_FIELDS:
        assert field in prompt
    assert "NO NUMBERS" in prompt
    assert str(JUDGMENT_TEXT_LIMIT) in prompt
    # The layout is not the model's on this path. Asserted on the JSON SHAPE the prompt
    # requires rather than on the word "html" anywhere in it: the prompt also FORBIDS
    # html in prose, and a scan for the word cannot tell a prohibition from a request.
    schema = next(line for line in prompt.splitlines() if line.startswith("Return ONLY JSON"))
    assert schema.count('": "') == len(JUDGMENT_FIELDS)
    for banned in ("html", "data", "field"):
        assert banned not in schema


# --------------------------------------------------------------------------
# the payload the host actually accepts
# --------------------------------------------------------------------------


def test_the_published_card_fits_the_host_caps_at_their_worst() -> None:
    """Every judgment field at its cap, every fold unreadable, and still normalizable."""
    long_judgment = validate_judgment({field: "x" * 400 for field in JUDGMENT_FIELDS})
    for reads in (_reads(), _reads(**{name: FOLD_UNREADABLE for name in FOLD_NAMES})):
        markup = read_crew_main_template()
        data = card_data_payload(
            merge_crew_main(build_crew_main(reads), {"sentences": SENTENCES_ON}, long_judgment)
        )
        assert len(markup.encode()) <= MAX_HTML_BYTES
        size = sum(len(k.encode()) + len(v.encode()) for k, v in data.items())
        assert size <= MAX_DATA_BYTES, f"{size} data bytes"
        assert normalize_card({"html": markup, "data": data}) is not None


def test_every_field_name_is_one_the_host_accepts() -> None:
    """``normalize_card`` refuses a name outside its grammar, silently dropping the card."""
    from kiro_crew.dashboard.dynamic_cards import _FIELD_NAME

    for field in CrewMainData.__annotations__:
        assert _FIELD_NAME.fullmatch(field), field


# --------------------------------------------------------------------------
# whose panel this is
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("created_by", "expected"),
    [
        pytest.param("", True, id="a-persons-own-tab"),
        pytest.param("member-bolin", False, id="dispatched-by-a-crew-member"),
        pytest.param("chat-1875-1790253512", False, id="dispatched-by-a-conductor"),
    ],
)
def test_only_a_root_session_takes_the_derived_panel(created_by: str, expected: bool) -> None:
    """ROOT means no parent, which is an empty ``_created_by`` and nothing else.

    Deliberately keyed on the parent field rather than on the slot KEY's shape. A crew
    member's DM slot is one kind of root session, not the definition of one, so a
    DM-shaped test would withhold the panel from every ordinary root tab -- which is most
    of the rows a person looks at, and the rows whose emptiness started this.
    """
    assert is_root_session(type("S", (), {"_created_by": created_by, "key": "s"})()) is expected


def test_the_panel_gate_reads_the_parent_field_and_not_the_slot_key() -> None:
    """A key-shape test and a parent test disagree on exactly the ordinary root tab."""
    import inspect

    from kiro_crew.dashboard import card_lifecycle

    source = inspect.getsource(card_lifecycle.is_root_session)
    # Both readings of "no parent" are present: the birth-time field and the tree edge.
    assert "_created_by" in source
    assert "parent_slot" in source
    # The key is READ, because the tree is keyed by it -- but its SHAPE is never parsed.
    # A shape test is what would turn this into a crew-DM notion by the back door.
    for token in ("slug_from_dm_slot_key", "startswith", "endswith", "split", "member-"):
        assert token not in source, f"the root gate parses the slot key with {token}"


def test_a_worker_is_not_eligible_for_a_panel_at_all() -> None:
    """The eligibility gate, not just the root predicate, has to refuse a child."""
    lifecycle = _lifecycle()
    assert lifecycle._eligible(_slot("root", created_by="")) is True
    assert lifecycle._eligible(_slot("child", created_by="root")) is False


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        pytest.param({"is_remote": True}, "a peer holds this slot's crew log", id="remote-flag"),
        pytest.param({"executor": "remote"}, "same, by executor", id="remote-executor"),
        pytest.param({"memory_mode": "incognito"}, "no durable record exists", id="incognito"),
    ],
)
def test_a_slot_with_no_local_crew_log_gets_no_panel(kwargs: dict, why: str) -> None:
    """These two exclusions are about the log, not about cost, so they are not toggleable.

    A remote slot's log is on its peer's disk and an incognito one is deliberately never
    written, so there is nothing here to fold either way.
    """
    assert _lifecycle()._eligible(_slot("s", **kwargs)) is False, why


# --------------------------------------------------------------------------
# the toggle and the budget gate the sentences, never the numbers
# --------------------------------------------------------------------------


def _publish(lifecycle, slot: _Slot, derived) -> dict[str, str]:
    """Publish *derived* for *slot* through the real write path, and return the data.

    Goes through ``notify`` first so the entry exists exactly as production makes it,
    then calls the same ``_write_card`` the derived worker calls. What is skipped is only
    the fold READ, which is supplied here instead -- so the card, its host fields and its
    normalization are the product's, not the test's.
    """
    lifecycle.notify(slot, "test")
    entry = lifecycle.publisher.entries[slot.key]
    lifecycle._derived[slot.key] = derived
    lifecycle._write_card(entry, derived, lifecycle._judgment.get(slot.key, EMPTY_JUDGMENT))
    assert entry.payload is not None, "the numbers did not publish"
    return entry.payload["data"]


@pytest.mark.asyncio
async def test_a_root_slot_with_the_toggle_off_gets_numbers_and_empty_sentences() -> None:
    """The case the live page failed: every row read "content generation is unavailable".

    The opt-in is off, so no sentence is written -- and every folded number is still
    there, with a line saying in words why the sentences are absent. Three blank
    sentences on their own are indistinguishable from a broken card.
    """
    slot = _slot("root", created_by="")
    lifecycle = _lifecycle(slot, enabled=False)
    data = _publish(lifecycle, slot, build_crew_main(_reads()))

    assert data["sentences"] == SENTENCES_OFF
    assert (data["lede"], data["you"], data["notes"]) == ("", "", "")
    for field in DERIVED_FIELDS:
        assert data[field], f"{field} lost its number with the toggle off"
    assert data["items_open"] == "3 of 5 items not yet done"
    assert data["credits"].startswith("3.50 credits")
    # And the entry is not left pending: nothing will ask for sentences, so a pending
    # entry would spin the drain worker and report the card as queued for ever.
    assert lifecycle.publisher.entries["root"].pending is False
    # notify QUEUED the derived publish even though the opt-in is off. Asserted because
    # this case supplies the numbers itself, so without it a notify that skipped the
    # derived queue while disabled would still show a card full of numbers here and none
    # at all in production.
    assert "root" in lifecycle._derived_pending
    assert lifecycle._derived_worker is not None


@pytest.mark.asyncio
async def test_a_child_slot_gets_no_derived_panel() -> None:
    """A worker is refused before an entry is ever created, so there is nothing to serve."""
    child = _slot("child", created_by="root")
    lifecycle = _lifecycle(child, enabled=True)
    lifecycle.notify(child, "test")
    assert "child" not in lifecycle.publisher.entries
    assert "child" not in lifecycle._derived
    assert lifecycle._derived_pending == set()


@pytest.mark.asyncio
async def test_the_numbers_publish_with_the_toggle_on_and_the_budget_spent() -> None:
    """The budget is the other sentence gate, and it is also not a number gate."""
    slot = _slot("root", created_by="")
    lifecycle = _lifecycle(slot, enabled=True)
    # Spend the hour against the publisher's own deque, which is the number ``run_ready``
    # consults -- not a second tally that could disagree with it.
    lifecycle.publisher.attempts.extend([0.0] * lifecycle.publisher.budget.per_hour)
    data = _publish(lifecycle, slot, build_crew_main(_reads()))
    assert data["sentences"] == SENTENCES_OVER_BUDGET
    assert data["items_open"] == "3 of 5 items not yet done"


@pytest.mark.asyncio
async def test_turning_the_toggle_off_keeps_the_card_and_drops_the_sentences() -> None:
    """Disabling republishes rather than clearing: the numbers cost nothing to keep.

    Clearing every entry is what the old path did, and it is why a page of rows went
    blank while the logs beside them held the numbers.
    """
    slot = _slot("root", created_by="")
    lifecycle = _lifecycle(slot, enabled=True)
    _publish(lifecycle, slot, build_crew_main(_reads()))
    lifecycle._judgment[slot.key] = validate_judgment({"lede": "Three items are moving."})
    entry = lifecycle.publisher.entries[slot.key]
    lifecycle._write_card(entry, lifecycle._derived[slot.key], lifecycle._judgment[slot.key])
    assert entry.payload is not None and entry.payload["data"]["lede"] == "Three items are moving."

    lifecycle.set_enabled(False)

    assert "root" in lifecycle.publisher.entries, "the card was cleared"
    data = lifecycle.publisher.entries["root"].payload
    assert data is not None
    assert data["data"]["items_open"] == "3 of 5 items not yet done"
    # The stale sentence is dropped, not frozen: prose written while the sentences were
    # on, sitting beside a line saying they are off, is the contradiction to avoid.
    assert data["data"]["lede"] == ""
    assert data["data"]["sentences"] == SENTENCES_OFF

    # And it is dropped from the STORE, not merely omitted from this one publish. The
    # derived publisher reads ``_judgment.get(key, EMPTY_JUDGMENT)`` on every later fold
    # change, so a retained judgment would put the stale sentence back on the next
    # number update -- while the line beside it still said the sentences were off.
    assert "root" not in lifecycle._judgment
    lifecycle._write_card(
        entry, lifecycle._derived["root"], lifecycle._judgment.get("root", EMPTY_JUDGMENT)
    )
    republished = lifecycle.publisher.entries["root"].payload
    assert republished is not None
    assert republished["data"]["lede"] == ""
    assert republished["data"]["sentences"] == SENTENCES_OFF


@pytest.mark.asyncio
async def test_the_sentences_field_is_never_empty_in_any_state() -> None:
    """A blank here and three blank sentences are the same page, meaning different things."""
    slot = _slot("root", created_by="")
    for enabled, spend in ((False, 0), (True, 0), (True, 60)):
        lifecycle = _lifecycle(slot, enabled=enabled)
        lifecycle.publisher.attempts.extend([0.0] * spend)
        data = _publish(lifecycle, slot, build_crew_main(_reads()))
        assert data["sentences"].strip(), f"enabled={enabled} spend={spend} said nothing"
    assert {SENTENCES_ON, SENTENCES_OFF, SENTENCES_OVER_BUDGET} == {
        SENTENCES_ON,
        SENTENCES_OFF,
        SENTENCES_OVER_BUDGET,
    }


def test_the_progress_label_names_items_and_not_sessions() -> None:
    """The count is of work items reporting progress; a session count would include workers."""
    markup = read_crew_main_template()
    assert "Reporting progress" in markup
    assert ">Running<" not in markup


# --------------------------------------------------------------------------
# root means no parent edge, and there are two places an edge can live
# --------------------------------------------------------------------------


class _Node:
    """A session-tree node, as ``fold_tree`` returns it: slot, parent, cycle flag."""

    def __init__(self, slot: str, parent_slot: str | None) -> None:
        self.slot = slot
        self.parent_slot = parent_slot
        self.cycle = False


def _with_tree(monkeypatch, nodes: dict[str, _Node]) -> None:
    """Stand in for the in-memory session-tree projection ``is_root_session`` reads."""
    from kiro_crew.crew_log import session_tree_projection

    monkeypatch.setattr(
        session_tree_projection, "projection", lambda: SimpleNamespace(nodes=lambda: nodes)
    )


def test_an_adopted_slot_is_not_root_even_with_an_empty_created_by(monkeypatch) -> None:
    """The case ``_created_by`` alone cannot see, and the reason the tree is read.

    The adopt verb records a parent edge in the crew log and never touches
    ``_created_by``, so a slot born as a person's own tab and later taken over still
    reads as parentless by that field. Without the tree reading it would be handed a
    panel of its own while its parent already summarises it.
    """
    adopted = _slot("adopted", created_by="")
    _with_tree(monkeypatch, {"adopted": _Node("adopted", "owner")})
    assert is_root_session(adopted) is False
    assert _lifecycle(adopted)._eligible(adopted) is False


def test_a_released_slot_is_root_again(monkeypatch) -> None:
    """A parent edge can be taken away, and the tree is what records that."""
    released = _slot("released", created_by="")
    _with_tree(monkeypatch, {"released": _Node("released", None)})
    assert is_root_session(released) is True


def test_a_dispatched_worker_is_refused_before_the_tree_is_consulted(monkeypatch) -> None:
    """``_created_by`` decides on its own, so a flag-off gateway still refuses a worker.

    The tree is empty here, which is what "the crew log is off" looks like. Read alone it
    would call every slot root, so this pins that the cheap field is required too.
    """
    worker = _slot("worker", created_by="root")
    _with_tree(monkeypatch, {})
    assert is_root_session(worker) is False


def test_an_empty_tree_leaves_an_ordinary_tab_as_root(monkeypatch) -> None:
    """An empty tree is "nothing cites a creator", not "nothing is known"."""
    tab = _slot("tab", created_by="")
    _with_tree(monkeypatch, {})
    assert is_root_session(tab) is True


def test_an_unreadable_tree_falls_back_to_the_cheap_reading(monkeypatch) -> None:
    """A broken projection must cost the wider reading, never the panel or the turn."""
    from kiro_crew.crew_log import session_tree_projection

    def boom():
        raise RuntimeError("tree unavailable")

    monkeypatch.setattr(session_tree_projection, "projection", boom)
    assert is_root_session(_slot("tab", created_by="")) is True
    assert is_root_session(_slot("worker", created_by="root")) is False


def test_the_root_test_does_no_file_io(monkeypatch) -> None:
    """It runs on the gateway serving loop, so a file read here would stall every task.

    Asserted by making ``open`` raise for the duration: the projection's own ``nodes()``
    documents that it never reads a file, and this pins that the predicate around it
    does not either.
    """
    import builtins

    _with_tree(monkeypatch, {"tab": _Node("tab", None)})

    def no_open(*args: object, **kwargs: object):
        raise AssertionError("the root test opened a file")

    monkeypatch.setattr(builtins, "open", no_open)
    assert is_root_session(_slot("tab", created_by="")) is True


# --------------------------------------------------------------------------
# which path publishes is a property of the LOG, never of who won a race
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_slot_whose_every_fold_is_unreadable_gets_no_derived_numbers(monkeypatch) -> None:
    """No numbers means no card from the derived path, and the model keeps the slot.

    A template painted from five unreadable folds would read "could not be read" nineteen
    times, which tells a reader nothing they can act on.

    This is also what makes the choice of path DETERMINISTIC, and that is why the case
    exists. Keying it on whether ``_derived`` happened to be populated made it a question
    of which task won -- the derived worker, or a model generation that runs up to 45
    seconds -- so the same session could publish either kind of card depending on timing.
    That shape passed locally and failed in CI, which is the signature of a race and not
    of a fallback. Keyed on whether the log could be read at all, the answer is a property
    of the session and is identical however the two tasks interleave.
    """
    slot = _slot("root", created_by="")
    lifecycle = _lifecycle(slot, enabled=False)
    lifecycle.notify(slot, "test")
    entry = lifecycle.publisher.entries[slot.key]

    async def no_folds(fn, *args):
        return {name: FOLD_UNREADABLE for name in FOLD_NAMES}

    monkeypatch.setattr(card_lifecycle.asyncio, "to_thread", no_folds)
    await lifecycle._publish_derived(slot.key)
    await _settle(lifecycle)

    assert slot.key not in lifecycle._derived, "unreadable folds stored as numbers"
    assert entry.payload is None, "a card of nineteen refusals was published"


@pytest.mark.asyncio
async def test_one_readable_fold_is_enough_for_a_derived_card(monkeypatch) -> None:
    """The refusal is for ALL five failing, not for any one of them.

    A partial read is exactly what the three-state wording exists for: the folds that
    answered carry their numbers, and the ones that did not say "could not be read" beside
    them. Refusing the card here would throw away the half that IS known.
    """
    slot = _slot("root", created_by="")
    lifecycle = _lifecycle(slot, enabled=False)
    lifecycle.notify(slot, "test")
    entry = lifecycle.publisher.entries[slot.key]

    partial = {name: FOLD_UNREADABLE for name in FOLD_NAMES}
    partial["status"] = {"lifecycle": "open", "turns_completed": 4}

    async def one_fold(fn, *args):
        return partial

    monkeypatch.setattr(card_lifecycle.asyncio, "to_thread", one_fold)
    await lifecycle._publish_derived(slot.key)
    await _settle(lifecycle)

    assert slot.key in lifecycle._derived
    assert entry.payload is not None
    data = entry.payload["data"]
    assert data["turns"] == "4 turns finished"
    assert data["credits"] == UNREADABLE


# --------------------------------------------------------------------------
# numbers that stop being supportable are dropped, not left on screen
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_folds_going_unreadable_drops_the_numbers_already_published(monkeypatch) -> None:
    """A slot that HAD readable folds and loses them must not keep showing the old ones.

    Leaving them is the worst of the three outcomes. The figures are ones a reader would
    act on, nothing can refresh them, and the model path stays unreachable because it is
    chosen by this cache being empty -- so the panel would sit on numbers from the last
    successful read for as long as the entry lives.
    """
    slot = _slot("root", created_by="")
    lifecycle = _lifecycle(slot, enabled=False)
    lifecycle.notify(slot, "test")
    entry = lifecycle.publisher.entries[slot.key]

    async def reads(payload):
        async def _run(fn, *args):
            return payload

        monkeypatch.setattr(card_lifecycle.asyncio, "to_thread", _run)
        await lifecycle._publish_derived(slot.key)
        await _settle(lifecycle)

    await reads(_reads())
    assert entry.payload is not None
    assert "3 of 5 items not yet done" in entry.payload["data"].values()

    await reads({name: FOLD_UNREADABLE for name in FOLD_NAMES})

    assert slot.key not in lifecycle._derived, "stale numbers kept in the cache"
    assert entry.payload is None, "a card of unsupportable numbers stayed on screen"
    assert entry.published_at is None


# --------------------------------------------------------------------------
# a digit in a sentence is a number this panel did not fold
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        pytest.param("40 turns completed so far.", id="leading-count"),
        pytest.param("About 3 items are blocked.", id="hedged-count"),
        pytest.param("Spend is near 12.5 credits.", id="decimal"),
        pytest.param("See ticket 4172 for context.", id="reference"),
    ],
)
def test_a_sentence_carrying_a_digit_is_dropped(sentence: str) -> None:
    """The merge cannot catch a figure inside the field the model may write, so the door
    is at :func:`validate_judgment`.

    The reference case is included deliberately: this refuses a legitimate ticket
    reference too. That trade is the point -- a false refusal costs one sentence, a false acceptance
    puts a guessed figure on a panel whose whole premise is that its figures come from the
    log.

    The drop takes all three sentences with it, for the display-boundary reason: what a
    reader would otherwise see is the two survivors and no way to read the gap.
    """
    judgment = validate_judgment({"lede": sentence, "you": "Rule on it.", "notes": ""})
    assert judgment == EMPTY_JUDGMENT


def test_a_sentence_with_no_digit_survives() -> None:
    """The control. Without it, a rule that dropped everything would read as working.

    Also the other half of the display boundary: ``notes`` is empty here because the MODEL
    left it empty, which is its documented choice and not a removal, so the two sentences
    it did write publish.
    """
    judgment = validate_judgment(
        {
            "lede": "Three of six items are still open and one worker is blocked.",
            "you": "Rule on the blocked item.",
            "notes": "",
        }
    )
    assert judgment["lede"].startswith("Three of six")
    assert judgment["you"] == "Rule on the blocked item."


def test_the_prompt_tells_the_model_the_digit_rule() -> None:
    """A door the writer cannot see produces empty fields it cannot explain."""
    assert "no digits at all" in _JUDGMENT_PROMPT


# --------------------------------------------------------------------------
# the three sentences are one display boundary, and one credential boundary
# --------------------------------------------------------------------------


def test_a_credential_split_across_two_sentences_empties_all_three() -> None:
    """The case a per-field scan cannot see at all.

    Several catalogue rules identify a credential by a neighbouring LABEL rather than by
    the value alone. Put the label in one sentence and the value in the next and each
    field is clean read on its own, while the page -- where the three render one under
    another -- carries the whole thing.

    The pair below is the one the free-form suite already pins as label-dependent: the
    forty characters alone are not a credential to the catalogue, and the label alone is
    just a word. Together, with any whitespace between them, they are.
    """
    label, value = "aws_secret_access_key:", "A" * 40
    assert _redact(label) == label, "the label alone must not be a credential"
    assert _redact(value) == value, "the value alone must not be a credential"

    judgment = _redact_judgment(json.dumps({"lede": label, "you": value, "notes": "Rule on it."}))
    assert judgment == EMPTY_JUDGMENT


def test_a_clean_three_survives_the_joined_scan() -> None:
    """The control for the joined scan. Without it, a scan that bit on everything passes."""
    judgment = _redact_judgment(
        json.dumps(
            {
                "lede": "The board is moving and nothing is stuck.",
                "you": "Nothing needs you.",
                "notes": "",
            }
        )
    )
    assert judgment["lede"].startswith("The board is moving")
    assert judgment["you"] == "Nothing needs you."


@pytest.mark.asyncio
async def test_a_judgment_the_source_check_rejects_is_never_cached(monkeypatch) -> None:
    """A rejected payload must not leave its prose behind to be republished.

    The source re-read after the model returns is what discards a result the slot moved
    on from. Cached before that check, the sentences survived the discard -- and the
    derived path republishes this cache on the slot's next fold event, so prose the check
    had just rejected reappeared under numbers that had moved on. Nothing clears the cache
    for a live entry, so it never corrected itself either.
    """
    slot = _slot("root", created_by="")
    log = _Log(slot)
    lifecycle = _lifecycle(slot, enabled=True, log=log)
    lifecycle.notify(slot, "test")
    entry = lifecycle.publisher.entries[slot.key]
    lifecycle._derived[slot.key] = build_crew_main(_reads())

    async def _to_thread(fn, *args):
        # Run inline -- every callee here is synchronous -- and move the transcript on the
        # moment the run has taken its snapshot. That is the race: the generation the run
        # started from is gone by the time the model answers.
        result = fn(*args)
        if getattr(fn, "__name__", "") == "source_snapshot":
            log.generation += 1
        return result

    monkeypatch.setattr(card_lifecycle.asyncio, "to_thread", _to_thread)

    async def _oneliner(*args, **kwargs):
        return json.dumps({"lede": "Stale prose.", "you": "Rule on it.", "notes": ""})

    monkeypatch.setattr(card_lifecycle, "run_bg_oneliner", _oneliner)
    monkeypatch.setattr(
        card_lifecycle.KiroCrewConfig,
        "load",
        staticmethod(
            lambda: SimpleNamespace(
                agent=SimpleNamespace(resolve_model=lambda role: "auto"),
                dashboard=SimpleNamespace(dynamic_dashboard_cards=True),
            )
        ),
    )

    payload = await lifecycle._generate(entry)
    await _settle(lifecycle)

    assert payload is None, "the stale result published"
    assert slot.key not in lifecycle._judgment, "prose the source check rejected was cached"


@pytest.mark.asyncio
async def test_switching_the_cards_off_takes_a_model_authored_card_off_the_page() -> None:
    """The switch is the operator's, so a card it cannot reach is the switch not working.

    A slot with derived numbers keeps them and loses its sentences -- that is the whole
    point of the split. A slot with NO derived numbers has only the model's card, and
    leaving that served would show an operator a model-authored card after they switched
    model-authored cards off. Broadcasting the change alone left exactly that, because the
    correction rides the slot's next crew-log event and an idle slot never has one.
    """
    derived_slot = _slot("root", created_by="")
    freeform_slot = _slot("plain", created_by="")
    lifecycle = _lifecycle(derived_slot, freeform_slot, enabled=True)
    data = _publish(lifecycle, derived_slot, build_crew_main(_reads()))
    assert data["items_open"] == "3 of 5 items not yet done"

    lifecycle.notify(freeform_slot, "test")
    plain = lifecycle.publisher.entries[freeform_slot.key]
    plain.payload = normalize_card({"html": "<p>from a model</p>", "data": {}}, None)
    lifecycle._derived.pop(freeform_slot.key, None)
    lifecycle._judgment[freeform_slot.key] = {"lede": "x", "you": "y", "notes": ""}

    # Asserted before settling: ``set_enabled`` is synchronous and this is its whole
    # effect. Settling afterwards lets the derived worker refold from the real store,
    # which legitimately rewrites the numbers and would say nothing about the switch.
    lifecycle.set_enabled(False)

    assert freeform_slot.key not in lifecycle.publisher.entries, "a model card stayed served"
    kept = lifecycle.publisher.entries[derived_slot.key]
    assert kept.payload is not None
    assert kept.payload["data"]["items_open"] == "3 of 5 items not yet done"
    assert kept.payload["data"]["sentences"] == SENTENCES_OFF
    assert (kept.payload["data"]["lede"], kept.payload["data"]["you"]) == ("", "")
    await _settle(lifecycle)


@pytest.mark.asyncio
async def test_a_slot_that_never_had_numbers_keeps_its_model_card(monkeypatch) -> None:
    """The derived worker must not delete a card it did not publish.

    Both producers run on one slot. On a slot with no readable crew log the derived worker
    has nothing to publish and the model's own free-form card is what is on screen -- so a
    worker that cleared the payload on finding no fold deleted the OTHER producer's card,
    and which of the two finished second decided whether the card survived. That is why
    this passed in every local ordering and failed in a CI shard.

    The distinction is the cache: a key in it was published from a fold, and only then is
    there anything of the derived worker's to take away.
    """
    slot = _slot("root", created_by="")
    lifecycle = _lifecycle(slot, enabled=True)
    lifecycle.notify(slot, "test")
    entry = lifecycle.publisher.entries[slot.key]
    model_card = normalize_card({"html": "<p>from a model</p>", "data": {}}, None)
    entry.payload = model_card
    assert slot.key not in lifecycle._derived, "this slot must start with no folded numbers"

    async def no_folds(fn, *args):
        return {name: FOLD_UNREADABLE for name in FOLD_NAMES}

    monkeypatch.setattr(card_lifecycle.asyncio, "to_thread", no_folds)
    await lifecycle._publish_derived(slot.key)
    await _settle(lifecycle)

    assert entry.payload == model_card, "the derived worker deleted the model's card"


@pytest.mark.asyncio
async def test_shutdown_settles_the_derived_worker_and_not_only_the_model_one(monkeypatch) -> None:
    """This producer owns TWO tasks, and a teardown that knows one leaves the other running.

    Held at its fold read on purpose. A derived worker that has already finished says
    nothing about whether shutdown waited for it, which is how a teardown missing it reads
    as working: the task was simply done by the time anyone looked.
    """
    slot = _slot("root", created_by="")
    lifecycle = _lifecycle(slot, enabled=False)
    release = asyncio.Event()

    async def held(fn, *args):
        await release.wait()
        return {name: FOLD_UNREADABLE for name in FOLD_NAMES}

    monkeypatch.setattr(card_lifecycle.asyncio, "to_thread", held)
    lifecycle.notify(slot, "test")
    await asyncio.sleep(0)
    task = lifecycle._derived_worker
    assert task is not None and not task.done(), "the derived worker is not at its read"

    # ``set`` does not yield, so the worker is still pending when shutdown is entered.
    release.set()
    await asyncio.wait_for(lifecycle.shutdown(), 2)

    assert task.done(), "shutdown left the derived worker running"
    assert not lifecycle.state._background_tasks


@pytest.mark.parametrize(
    ("lifecycle", "expected"),
    [
        pytest.param("open", "session open", id="open"),
        pytest.param("closed", "session closed", id="closed"),
        pytest.param("unknown", "the log no longer says", id="retention-took-it"),
    ],
)
def test_the_state_pill_names_the_session_it_is_about(lifecycle: str, expected: str) -> None:
    """ "open" carries two senses on this one panel, and the pill must not be the bare word.

    The work tile beside it counts items that are open in the sense of UNRESOLVED. A pill
    reading just "open" invites a reader to carry one sense onto the other field, so the
    pill says which thing is open.
    """
    status = {**_reads()["status"], "lifecycle": lifecycle}  # type: ignore[dict-item]
    assert build_crew_main(_reads(status=status))["state"] == expected
