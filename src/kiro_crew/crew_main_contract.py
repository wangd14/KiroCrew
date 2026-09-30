"""The crew main dashboard's contract: one template, one type, one provider.

The crew side panel is today drawn by React arithmetic over a host inventory. The tiles,
the progress bar, the segmented-control counts and the per-session rows are all computed
in the browser from four REST reads plus the slot list in the store, and the card inside
the panel is written entirely by a background model -- its layout AND every number in it.
Two writers, neither of them the log.

This module replaces both. The chain is ``crew log -> fold -> this provider -> the
template paints``, and a number never passes through a model or through browser
arithmetic. The React shell becomes a plain host: it mounts the template and binds
``data-dashboard-field`` values, and draws no number of its own.

ONE template (:data:`CREW_MAIN_TEMPLATE`), ONE contract type (:class:`CrewMainData`), ONE
provider (:func:`build_crew_main`), ONE :data:`CONTRACT_VERSION`. The provider's return
type IS the template's field set, and ``test_crew_main_contract`` asserts that both ways
round, because mypy cannot see inside HTML. ``mypy src/kiro_crew/`` is blocking in CI
with ``check_untyped_defs`` on, so a missing required key here is a build failure rather
than a convention.

WHOSE PANEL THIS IS. A ROOT session -- one no other session dispatched -- and no other
slot. The gate is :func:`~kiro_crew.dashboard.card_lifecycle.is_root_session`, which is
an empty ``_created_by``: that field holds the slot key of the session that asked for
this one, so empty means "no parent", and it is the same root test the sidebar tree
already draws with.

A worker gets no panel. Its numbers are read by opening that worker's own session, and
it gets no row inside a root session's panel either, because that panel is about the root
session's own work.

A crew member's DM slot is one kind of root session, not the definition of one, so the
gate deliberately is NOT the DM key shape: keying on that would withhold the panel from
every ordinary root tab, which is most of the rows a person looks at.

THREE WRITERS, KEPT APART BY TYPE.

THREE WRITERS, KEPT APART BY TYPE.

* :class:`CrewMainDerived` -- nineteen fields, every one from a fold render. Neither the
  host nor a model can reach them, so the panel's arithmetic cannot disagree with the log
  it summarises.
* :class:`CrewMainHost` -- one field, and it is about the CARD rather than about the
  session: whether the three sentences below are being written at all. No fold can answer
  it, because it is a fact about this gateway's settings and this slot's cost
  eligibility, and a model must not answer it because a model that is switched off
  cannot say so. A host writer is otherwise avoided -- it is a second answer to a
  question the log already answers -- and the one field that looked like it needed one,
  which view the crew has published, turned out to have the ``panel`` fold behind it.
* :class:`CrewMainJudgment` -- the three sentences no fold can produce.

:func:`merge_crew_main` is the only place the three meet, and it copies every field by
NAME rather than updating a dict, so a model field can never land on a derived one.

WHAT THE TOGGLE AND THE BUDGET GATE. The three sentences, and nothing else. The card
publishes for every slot with a local crew log -- workers included -- whatever
``dashboard.dynamic_dashboard_cards`` says and whatever the hourly model budget has left.
When the sentences are withheld the numbers are all still there and
:attr:`CrewMainHost.sentences` says so in words, because a reader who sees three blank
lines cannot tell "switched off" from "the model had nothing to say" from "this card is
broken". The failure this replaces is concrete: with the toggle off, the lifecycle was
never even constructed, so every row on the page read "content generation is unavailable"
while the crew log beside it held every number those rows wanted.

EVERY VALUE IS A STRING, deliberately, and this contract therefore has no
``UNSAID -> null`` exit of the kind :mod:`kiro_crew.pipeline_board_contract` needs. The
data island is bound by ``website/src/pages/chat/command-center/dashboardDocument.ts``,
which sets ``element.textContent`` per field and requires a flat map of strings
(``normalize_card`` refuses anything else). So absence is carried in WORDS, on the page,
where the reader sees it:

* :data:`NOT_RECORDED` -- the fold was read and does not carry this key. The crew
  genuinely has no such fact yet.
* :data:`UNREADABLE` -- the fold could not be read at all. What the value is remains
  unknown, which is a different fact from there being none.

Those two are never collapsed. A fold that failed to read and a crew with nothing
recorded look identical once either becomes ``0``, and a zero standing in for an unknown
is the one failure a reader cannot recover from.

NO PERCENTAGES, and every count carries its denominator in words where one exists. This
is why the panel's ``<progress>`` bar has no field here: a bar is a percentage drawn, so
it states a ratio while hiding both of its terms, and a board of one item then looks
like a board of a hundred. A bare ``3`` has the same defect more quietly -- it invites
the reader to supply the total, and the total they supply is wrong.

NOTHING IN THE TEMPLATE IS A CONTROL. No count is an anchor, nothing is clickable, and
there is no hover affordance. The panel's real controls -- an approval, a question draft,
the link that opens a session -- stay in React OUTSIDE this document, because a control
inside a sandboxed presentation document is a control whose authority nobody can audit.
The host strips ``href``, ``src``, ``alt`` and ``title`` in card mode, so a value that
reads like a link is still only text, and the template says so in words.

WHAT IS DELIBERATELY NOT A FIELD.

* The ask_question inventory. ``api.pendingQuestions`` is a HOST read and no fold carries
  it, so there is no honest derived count of it and this contract does not invent one.
  The two question-shaped numbers that ARE folds appear instead:
  :attr:`CrewMainDerived.approvals_open` from the ``approvals`` fold, which counts TOOL
  approvals raised and not decided, and :attr:`CrewMainDerived.items_question` from the
  ``work`` fold, which counts items whose worker is waiting on the conductor. The
  question CARDS keep being rendered by React, which is where they have to be anyway:
  each one is a control.
* The published view's CONTENT. It is a sandboxed document React mounts, and an iframe
  document cannot nest another framed document's React tree, so composing it here is not
  available at any price. :attr:`CrewMainDerived.published_view` carries its TITLE and
  how many times the crew has published it, both from the ``panel`` fold; the view
  itself stays mounted beside this document in its own sandbox.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal, TypedDict, cast

from kiro_crew.work_vocab import WorkBoardView

__all__ = [
    "CONTRACT_VERSION",
    "CREW_MAIN_TEMPLATE",
    "DERIVED_FIELDS",
    "EMPTY_JUDGMENT",
    "HOST_FIELDS",
    "JUDGMENT_FIELDS",
    "JUDGMENT_TEXT_LIMIT",
    "NOT_RECORDED",
    "NO_PUBLISHED_VIEW",
    "SENTENCES_OFF",
    "SENTENCES_ON",
    "SENTENCES_OVER_BUDGET",
    "UNREADABLE",
    "CrewMainData",
    "CrewMainDerived",
    "CrewMainHost",
    "CrewMainJudgment",
    "CrewMainReads",
    "FoldUnreadable",
    "build_crew_main",
    "card_data_payload",
    "merge_crew_main",
    "read_crew_main_template",
    "template_fields",
    "validate_judgment",
]


# --------------------------------------------------------------------------
# the words absence is spelt with
# --------------------------------------------------------------------------

NOT_RECORDED: Final[str] = "not recorded"
"""The fold was read and carries no such key. Shown to the reader, in words."""

UNREADABLE: Final[str] = "could not be read"
"""The fold could not be read. NOT the same fact as nothing being recorded."""

NO_PUBLISHED_VIEW: Final[str] = "no view published yet"
"""The crew has published no dashboard view. A host fact, not a fold's."""


FoldUnreadable = Literal["__unreadable__"]
"""The type a caller passes INSTEAD of a fold render when the read failed."""

FOLD_UNREADABLE: Final[FoldUnreadable] = "__unreadable__"
"""Pass this for a fold whose read raised. It cannot arrive from a lookup."""


# --------------------------------------------------------------------------
# which template this contract is of
# --------------------------------------------------------------------------

CREW_MAIN_TEMPLATE: Final[str] = "crew_main.html"
"""The one template this contract describes, under ``dashboard_templates/``."""


def _template_dir() -> Path:
    return Path(__file__).resolve().parent / "dashboard_templates"


def read_crew_main_template() -> str:
    """The template's markup, as shipped. Inert: no script, no controls, no links."""
    return (_template_dir() / CREW_MAIN_TEMPLATE).read_text(encoding="utf-8")


def template_fields(markup: str) -> frozenset[str]:
    """Every ``data-dashboard-field`` name in *markup*.

    Parsed rather than pattern-matched, so the gate reads the field set the BROWSER
    binds. ``dashboardDocument.ts`` walks the parsed document's
    ``[data-dashboard-field]`` elements, so a name only a regex can find is a name
    nothing paints -- and a name inside a comment, which a regex would happily count,
    is not a field at all.
    """
    from html.parser import HTMLParser

    class _Fields(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.names: set[str] = set()

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            for name, value in attrs:
                if name == "data-dashboard-field" and value:
                    self.names.add(value)

    parser = _Fields()
    parser.feed(markup)
    parser.close()
    return frozenset(parser.names)


# --------------------------------------------------------------------------
# INPUT -- the fold renders, each readable or not
# --------------------------------------------------------------------------


class CrewMainReads(TypedDict):
    """The four fold renders this panel is built from, or :data:`FOLD_UNREADABLE`.

    Required keys, all four. A caller that could not read one says so with the sentinel
    rather than omitting the key, because an omitted key and a failed read are the same
    absence to a ``dict`` and different facts to a reader.

    The reads are passed IN rather than fetched here. This module then has no store to
    reach, so it cannot decide to read one more thing on a whim, and the provider is a
    pure function a test can drive with a hand-built board.
    """

    #: ``status`` fold, SESSION-keyed: lifecycle, turn, turn counts, last time.
    status: Mapping[str, Any] | FoldUnreadable
    #: ``usage`` fold, SESSION-keyed: credits with the per-source split, and tokens.
    usage: Mapping[str, Any] | FoldUnreadable
    #: ``approvals`` fold, SESSION-keyed: tool approvals raised, decided, still open.
    approvals: Mapping[str, Any] | FoldUnreadable
    #: ``work`` fold, SLOT-keyed and eager: the board's items and their states.
    work: WorkBoardView | FoldUnreadable
    #: ``panel`` fold, SLOT-keyed and eager: the crew's own published view record.
    panel: Mapping[str, Any] | FoldUnreadable


# --------------------------------------------------------------------------
# OUTPUT -- the shape crew_main.html reads
# --------------------------------------------------------------------------


class CrewMainDerived(TypedDict):
    """Everything derived from folds. Neither the host nor a model can write here."""

    #: Whether the crew log has this session open, closed, or cannot say.
    state: str
    #: What it is doing right now: a turn running, or how the last one ended.
    phase: str
    #: When the log last recorded anything for it.
    last_activity: str
    #: Turns the log saw complete, and how many were refused.
    turns: str
    #: Entries in the crew log, which is what every count here was folded from.
    entries: str
    #: The agent the log recorded for this session.
    agent: str
    #: The model the log recorded for it.
    model: str
    #: Work items still open, over the board's total.
    items_open: str
    #: Open items whose worker last reported progress, over the open ones. Labelled
    #: "Reporting progress" rather than "Running", because it counts work ITEMS and not
    #: sessions that are awake: a session count would have to include workers, which this
    #: panel does not show, so the old label named a question this number does not answer.
    items_progress: str
    #: Open items whose worker last reported blocked, over the open ones.
    items_blocked: str
    #: Items the conductor accepted, over the board's total.
    items_done: str
    #: Open items whose worker is waiting on the conductor's own decision.
    items_question: str
    #: Entries the work fold dropped. A fact about the LOG, not about the board, which
    #: is why it is its own count and is never added into the totals above.
    board_omitted: str
    #: Credits the log billed to this session, all sources together.
    credits: str
    #: The sub-agent share of that total, which is otherwise invisible.
    credits_subagents: str
    #: Tokens across every dimension the log measured.
    tokens: str
    #: Tool approvals raised and not yet decided, over those raised.
    approvals_open: str
    #: Approvals already decided, and how they went.
    approvals_decided: str
    #: The title of the crew's published dashboard view, and how many times it has been
    #: published, or :data:`NO_PUBLISHED_VIEW`. From the ``panel`` fold. The view itself
    #: is a sandboxed document React mounts beside this one; only its name is composed
    #: in, because one framed document cannot hold another's React tree.
    published_view: str


class CrewMainHost(TypedDict):
    """One field, about the card rather than about the session."""

    #: Whether the three sentences below are being written, and when they are not, why.
    #: One of the :data:`SENTENCES_ON` / :data:`SENTENCES_OFF` / :data:`SENTENCES_OVER_BUDGET`
    #: wordings. Never empty: a blank here and three blank sentences are the same
    #: page, and they mean different things.
    sentences: str


SENTENCES_ON: Final[str] = (
    "Sentences here come from a background model; every number is computed from the crew log."
)
"""The opt-in is on and this slot is eligible, so the sentences may be present."""

SENTENCES_OFF: Final[str] = (
    "Sentences are off for this dashboard, so it shows only numbers computed from the crew log."
)
"""``dashboard.dynamic_dashboard_cards`` is off. Every number above is unaffected."""

SENTENCES_OVER_BUDGET: Final[str] = (
    "The hourly limit for written sentences is used up, so this shows numbers only."
)
"""The budget is spent. Said in words rather than left as three blank lines."""


class CrewMainJudgment(TypedDict):
    """The model's whole surface: three sentences, and not one number among them."""

    #: One sentence: what this crew is doing.
    lede: str
    #: One sentence: what, if anything, the reader must do. Empty when nothing.
    you: str
    #: One sentence of caveat, or empty.
    notes: str


class CrewMainData(CrewMainDerived, CrewMainHost, CrewMainJudgment):
    """THE contract. Every field ``crew_main.html`` reads, and nothing else.

    The three halves are inherited rather than restated, so a field added to any one of
    them reaches this type without a second edit -- and the parity gate then fails
    against the template until the template gains it too.
    """


CONTRACT_VERSION: Final[int] = 1
""":class:`CrewMainData`'s shape. Bumped when that shape changes."""


DERIVED_FIELDS: Final[frozenset[str]] = frozenset(CrewMainDerived.__annotations__)
"""Fold-derived. Neither the host nor a model may write one."""

HOST_FIELDS: Final[frozenset[str]] = frozenset(CrewMainHost.__annotations__)
"""The host's. Not a fold's, because no log knows this gateway's settings; and not a
model's, because a model that is switched off cannot report that it is switched off."""

JUDGMENT_FIELDS: Final[frozenset[str]] = frozenset(CrewMainJudgment.__annotations__)
"""The fields a model may write, and the only ones."""

JUDGMENT_TEXT_LIMIT: Final[int] = 160
"""Per judgment field. One sentence, and a bound the data cap can always afford."""

_HALVES: Final[tuple[frozenset[str], ...]] = (DERIVED_FIELDS, HOST_FIELDS, JUDGMENT_FIELDS)

for _left in range(len(_HALVES)):  # pragma: no cover - import-time consistency
    for _right in range(_left + 1, len(_HALVES)):
        if _HALVES[_left] & _HALVES[_right]:
            raise RuntimeError(
                "a crew main field has exactly one writer: "
                f"{sorted(_HALVES[_left] & _HALVES[_right])}"
            )

_ALL_FIELDS: Final[frozenset[str]] = DERIVED_FIELDS | HOST_FIELDS | JUDGMENT_FIELDS

if set(CrewMainData.__annotations__) != _ALL_FIELDS:  # pragma: no cover - import-time
    raise RuntimeError(
        "CrewMainData must be exactly its three halves: "
        f"{sorted(set(CrewMainData.__annotations__) ^ _ALL_FIELDS)}"
    )


EMPTY_JUDGMENT: Final[CrewMainJudgment] = {"lede": "", "you": "", "notes": ""}
"""A model that said nothing, or was never called. The numbers publish regardless."""


# --------------------------------------------------------------------------
# the model's half, checked at RUN time
# --------------------------------------------------------------------------
#
# mypy checks the provider because the provider is Python in this tree. The other end is
# a background model returning JSON, so no type checker is anywhere near it. The contract
# is therefore enforced twice, in the two ways the two ends admit of.


def validate_judgment(raw: object) -> CrewMainJudgment:
    """*raw* as a :class:`CrewMainJudgment`. Never raises; never partly trusts.

    A model's output is data, not a contract, so a departure from the shape is read as
    "said nothing about that field" rather than refused wholesale: the panel's numbers
    come from folds and must publish whatever the model did.

    An UNKNOWN key is dropped silently for the same reason, and because the only thing a
    model could put in one is a number this panel takes from a fold instead.
    """
    out: CrewMainJudgment = {"lede": "", "you": "", "notes": ""}
    if not isinstance(raw, Mapping):
        return out
    # A REMOVAL is not a silence, and the difference decides whether the others publish.
    # The three sentences are ONE display boundary: they render one under another, so a
    # reader shown two of them cannot tell which rule took the third or whether the model
    # simply had nothing to say. So anything WE take out takes all three with it, while a
    # field the model left empty -- ``notes`` is documented as "or empty" -- is the model's
    # own choice and leaves its siblings alone.
    removed = False
    for field in ("lede", "you", "notes"):
        value = raw.get(field)
        if not isinstance(value, str):
            continue
        # One line: a model returning a paragraph with newlines would otherwise push a
        # multi-line block into a one-sentence slot.
        text = " ".join(value.split())[:JUDGMENT_TEXT_LIMIT]
        if any(character.isdigit() for character in text):
            # A DIGIT IN A SENTENCE IS A NUMBER THIS PANEL DID NOT FOLD. The merge cannot
            # catch it, because it is inside the field the model is entitled to write, and
            # the prompt forbidding it is an instruction rather than a guarantee -- so the
            # door is here.
            #
            # The whole sentence goes, not the digits: a sentence is dropped for the same
            # reason a redacted one is, in that the prose around the figure was written
            # about the figure, and stripping it leaves a claim with its subject missing.
            #
            # Blunt on purpose. It also refuses a legitimate "item 3" or a ticket reference, and
            # that trade is deliberate: a false refusal costs one sentence, while a false
            # acceptance puts a guessed figure on a panel whose entire premise is that its
            # figures come from the log. ASCII digits only, so a number spelt in words or
            # in another script still passes -- this narrows the hole rather than closing
            # it, and the prompt does the rest.
            removed = True
            continue
        out[field] = text
    if removed:
        return {"lede": "", "you": "", "notes": ""}
    return out


def merge_crew_main(
    derived: CrewMainDerived,
    host: CrewMainHost,
    judgment: CrewMainJudgment,
) -> CrewMainData:
    """The one place the three writers meet.

    Built by NAMING all twenty-three fields, not by ``{**derived, **host, **judgment}``.
    A dict update writes whatever keys the right-hand side happens to hold, so a judgment
    carrying ``credits`` would overwrite the folded value and nothing would notice;
    naming the fields means only a source edit could do that, and mypy would reject the
    derived key in a :class:`CrewMainJudgment` literal.
    """
    return {
        "state": derived["state"],
        "phase": derived["phase"],
        "last_activity": derived["last_activity"],
        "turns": derived["turns"],
        "entries": derived["entries"],
        "agent": derived["agent"],
        "model": derived["model"],
        "items_open": derived["items_open"],
        "items_progress": derived["items_progress"],
        "items_blocked": derived["items_blocked"],
        "items_done": derived["items_done"],
        "items_question": derived["items_question"],
        "board_omitted": derived["board_omitted"],
        "credits": derived["credits"],
        "credits_subagents": derived["credits_subagents"],
        "tokens": derived["tokens"],
        "approvals_open": derived["approvals_open"],
        "approvals_decided": derived["approvals_decided"],
        "published_view": derived["published_view"],
        "sentences": host["sentences"],
        "lede": judgment["lede"],
        "you": judgment["you"],
        "notes": judgment["notes"],
    }


def card_data_payload(data: CrewMainData) -> dict[str, str]:
    """*data* as the flat string map ``normalize_card`` accepts. The single exit."""
    return {key: str(value) for key, value in data.items()}


# --------------------------------------------------------------------------
# the provider -- the one place a fold render becomes a panel field
# --------------------------------------------------------------------------


class _StatusFields(TypedDict):
    """What the ``status`` fold answers for. A TypedDict so mypy checks the branches.

    The four section types below exist for one reason: a helper with an early return
    for an unreadable fold has as many branches as it has outcomes, and a plain
    ``dict[str, str]`` return lets one of those branches forget a key. The forgotten key
    then reaches the template as an absent binding, which paints an EMPTY cell -- the
    one outcome this contract is built to make impossible, because an empty cell and a
    recorded zero are indistinguishable. Typed, a branch that forgets a key fails the
    blocking mypy run instead.
    """

    state: str
    phase: str
    last_activity: str
    turns: str
    entries: str
    agent: str
    model: str


class _WorkFields(TypedDict):
    """What the ``work`` fold answers for."""

    items_open: str
    items_progress: str
    items_blocked: str
    items_done: str
    items_question: str
    board_omitted: str


class _UsageFields(TypedDict):
    """What the ``usage`` fold answers for."""

    credits: str
    credits_subagents: str
    tokens: str


class _ApprovalFields(TypedDict):
    """What the ``approvals`` fold answers for."""

    approvals_open: str
    approvals_decided: str


def build_crew_main(reads: CrewMainReads) -> CrewMainDerived:
    """Map five fold renders onto the panel's derived half.

    The signature is the contract: fold renders in, a checked field set out, and mypy
    checks both ends. Every branch answers in words -- no key here is ever left to a
    caller to fill in, and no absence is ever answered with a zero.

    The nineteen fields are NAMED rather than gathered by ``**`` expansion. Expansion
    reads shorter and checks nothing: mypy cannot verify a ``dict`` spread against a
    TypedDict's required keys, so a section helper that dropped a field would produce a
    value missing that key with no error anywhere. Named, the assignment is checked.
    """
    status = _status_fields(reads["status"])
    work = _work_fields(reads["work"])
    usage = _usage_fields(reads["usage"])
    approvals = _approval_fields(reads["approvals"])
    return {
        "state": status["state"],
        "phase": status["phase"],
        "last_activity": status["last_activity"],
        "turns": status["turns"],
        "entries": status["entries"],
        "agent": status["agent"],
        "model": status["model"],
        "items_open": work["items_open"],
        "items_progress": work["items_progress"],
        "items_blocked": work["items_blocked"],
        "items_done": work["items_done"],
        "items_question": work["items_question"],
        "board_omitted": work["board_omitted"],
        "credits": usage["credits"],
        "credits_subagents": usage["credits_subagents"],
        "tokens": usage["tokens"],
        "approvals_open": approvals["approvals_open"],
        "approvals_decided": approvals["approvals_decided"],
        "published_view": _published_view(reads["panel"]),
    }


def _published_view(panel: Mapping[str, Any] | FoldUnreadable) -> str:
    """The crew's published view, named, or why there is no name for it.

    An empty ``template`` is the fold's OWN way of saying this crew has published
    nothing -- the store refuses a publish naming no template, so no real record has an
    empty one -- which is why that is the key tested rather than the title. A crew that
    published a view and gave it no title has a record with a template and a blank
    title, and that reads as published-but-unnamed rather than as never published.
    """
    if panel == FOLD_UNREADABLE:
        return UNREADABLE
    fold = cast("Mapping[str, Any]", panel)
    template = fold.get("template")
    if not isinstance(template, str) or not template.strip():
        return NO_PUBLISHED_VIEW
    title = fold.get("title")
    named = title.strip()[:60] if isinstance(title, str) and title.strip() else "an untitled view"
    publishes = _count(fold.get("publishes"))
    if publishes is None or publishes <= 0:
        return named
    times = "once" if publishes == 1 else f"{publishes} times"
    return f"{named}, a view this crew published {times}"


def _status_fields(status: Mapping[str, Any] | FoldUnreadable) -> _StatusFields:
    """The ``status`` fold's seven fields, or seven honest refusals."""
    if status == FOLD_UNREADABLE:
        return {
            "state": UNREADABLE,
            "phase": UNREADABLE,
            "last_activity": UNREADABLE,
            "turns": UNREADABLE,
            "entries": UNREADABLE,
            "agent": UNREADABLE,
            "model": UNREADABLE,
        }
    fold = cast("Mapping[str, Any]", status)
    lifecycle = fold.get("lifecycle")
    if lifecycle == "open":
        state = "session open"
    elif lifecycle == "closed":
        state = "session closed"
    elif lifecycle == "unknown":
        # The fold's own third state: retention removed the entry that opened the
        # session, so the log cannot say. Not the same as the key being absent.
        state = "the log no longer says"
    else:
        state = NOT_RECORDED
    completed = _count(fold.get("turns_completed"))
    refused = _count(fold.get("turns_refused"))
    return {
        "state": state,
        "phase": _phase(fold),
        "last_activity": _stamp(fold.get("last_time")),
        "turns": (
            NOT_RECORDED
            if completed is None
            # "refused" alone named nothing a reader could place. These count
            # ``turn/refused`` entries, so what refused is the backend -- and saying
            # "approval" here would point at the approvals tile, a different thing on
            # this same card.
            else f"{completed} turns finished"
            + ("" if not refused else f", {refused} refused by the backend")
        ),
        "entries": _plain(fold.get("entries"), "entries in the log"),
        "agent": _text(fold.get("agent")),
        "model": _text(fold.get("model")),
    }


def _phase(fold: Mapping[str, Any]) -> str:
    """What the crew is doing, from the fold's open turn or its last stop.

    ``turn_open`` is read rather than ``turn``: the fold REPORTS an open turn and never
    closes one, so the flag is the fold's own answer and the dict beside it is the
    detail. A session with a stop reason reads as that reason, which is the nearest
    thing the log has to a phase.
    """
    if fold.get("turn_open") is True:
        return "a turn is running now"
    reason = fold.get("last_stop_reason")
    if isinstance(reason, str) and reason.strip():
        return f"last turn ended: {reason.strip()[:80]}"
    if "turn_open" not in fold and "last_stop_reason" not in fold:
        return NOT_RECORDED
    return "no turn running"


def _work_fields(work: WorkBoardView | FoldUnreadable) -> _WorkFields:
    """The board's six counts, each with its denominator in words.

    ``state`` and ``status`` are two different vocabularies and both are used here,
    because they answer two different questions. ``state`` is the CONDUCTOR's
    disposition (open / accepted / rejected / abandoned); ``status`` is the worker's own
    last report (progress / done / blocked / question). A worker's ``done`` is a claim
    its conductor has not ruled on, so ``items_done`` counts ACCEPTED items -- the
    ruling -- and says the word, rather than promoting a claim to a result.
    """

    def same(text: str, omitted: str) -> _WorkFields:
        """All five item counts reading *text*, with ``board_omitted`` its own answer.

        ``board_omitted`` is never folded into *text*: it counts log entries the fold
        DROPPED, which stays answerable when the item list does not, so a board whose
        items cannot be read can still say how many entries went missing.
        """
        return {
            "items_open": text,
            "items_progress": text,
            "items_blocked": text,
            "items_done": text,
            "items_question": text,
            "board_omitted": omitted,
        }

    if work == FOLD_UNREADABLE:
        return same(UNREADABLE, UNREADABLE)
    board = cast("WorkBoardView", work)
    items = board.get("items")
    omitted = _dropped_entries(board.get("omitted"))
    if not isinstance(items, list):
        return same(NOT_RECORDED, omitted)
    total = len(items)
    if not total:
        # A real answer, not an absence: this crew folded a board and it has no items.
        # Saying "not recorded" here would hide a fact the fold does carry.
        return same("no work items on this board", omitted)
    open_items = [item for item in items if item.get("state") == "open"]
    accepted = sum(1 for item in items if item.get("state") == "accepted")
    open_total = len(open_items)

    def of_open(count: int, tail: str, empty: str) -> str:
        return f"{count} of {open_total} open items {tail}" if open_total else empty

    return {
        "items_open": f"{open_total} of {total} items not yet done",
        "items_progress": of_open(
            sum(1 for item in open_items if item.get("status") == "progress"),
            "reporting progress",
            "no open items to be working",
        ),
        "items_blocked": of_open(
            sum(1 for item in open_items if item.get("status") == "blocked"),
            "blocked",
            "no open items to block",
        ),
        "items_done": f"{accepted} of {total} items accepted",
        "items_question": of_open(
            sum(1 for item in open_items if item.get("status") == "question"),
            "waiting on you",
            "no items waiting on you",
        ),
        "board_omitted": omitted,
    }


def _dropped_entries(value: object) -> str:
    """How many entries the fold dropped, and what that does to every count beside it.

    The qualifier rides HERE rather than in the model's ``notes``, because a reader deciding
    from a tally needs to know the tally is a floor whether or not the sentences are switched
    on. A warning that disappears with the opt-in while the number it qualifies stays is the
    same defect as a number written by a model: the page keeps its confidence and loses its
    caveat.

    A count of zero is stated too. "No entries dropped" is a real assurance, and leaving the
    field silent would make a complete board and an unreadable one look alike.
    """
    count = _count(value)
    if count is None:
        return NOT_RECORDED
    if not count:
        return "no entries dropped, so these counts are complete"
    return f"{count} entries dropped, so these counts are a floor"


def _usage_fields(usage: Mapping[str, Any] | FoldUnreadable) -> _UsageFields:
    """Credits, the sub-agent share of them, and tokens.

    The sub-agent share gets its own field because it is otherwise invisible: it is
    folded into the same total as the crew's own turns, so a reader looking at one number
    cannot tell a crew that spent it all itself from one that fanned out. ``reported``
    beside each bucket is what says how much of the bucket the total covers, so a bucket
    nothing reported reads as unmetered rather than as free.
    """
    if usage == FOLD_UNREADABLE:
        return {"credits": UNREADABLE, "credits_subagents": UNREADABLE, "tokens": UNREADABLE}
    fold = cast("Mapping[str, Any]", usage)
    charge = fold.get("credits")
    by_source = fold.get("credits_by_source")
    subagent = by_source.get("subagent") if isinstance(by_source, Mapping) else None
    tokens = fold.get("tokens")
    total_tokens = tokens.get("total") if isinstance(tokens, Mapping) else None
    return {
        "credits": (
            NOT_RECORDED
            if not isinstance(charge, (int, float)) or isinstance(charge, bool)
            else f"{_credits(charge)} credits billed to this crew"
        ),
        "credits_subagents": _subagent_credits(subagent),
        "tokens": _plain(total_tokens, "tokens measured"),
    }


def _subagent_credits(bucket: object) -> str:
    """The sub-agent bucket, or why there is no number for it."""
    if not isinstance(bucket, Mapping):
        return NOT_RECORDED
    charge = bucket.get("credits")
    reported = bucket.get("reported")
    if not isinstance(charge, (int, float)) or isinstance(charge, bool):
        return NOT_RECORDED
    if not isinstance(reported, int) or isinstance(reported, bool) or reported <= 0:
        # Zero charges reported is not a charge of zero: an unmetered provider writes no
        # credits key at all, and folding that in as 0.0 would state a measurement
        # nobody made.
        return "no sub-agent charge reported"
    return f"{_credits(charge)} of that from {reported} sub-agent charges"


def _approval_fields(approvals: Mapping[str, Any] | FoldUnreadable) -> _ApprovalFields:
    """Tool approvals still open, and how the decided ones went.

    TOOL approvals, and the field says so on the page. The panel's other
    question-shaped number is the ask_question inventory, which is a host read with no
    fold behind it and therefore not a field in this contract at all.
    """
    if approvals == FOLD_UNREADABLE:
        return {"approvals_open": UNREADABLE, "approvals_decided": UNREADABLE}
    fold = cast("Mapping[str, Any]", approvals)
    pending = _count(fold.get("pending"))
    requested = _count(fold.get("requested"))
    decided = _count(fold.get("decided"))
    if pending is None:
        open_text = NOT_RECORDED
    elif requested is None:
        open_text = f"{pending} tool approvals still waiting"
    else:
        open_text = f"{pending} of {requested} tool approvals still waiting"
    return {
        "approvals_open": open_text,
        "approvals_decided": (
            NOT_RECORDED if decided is None else f"{decided} answered{_decisions(fold)}"
        ),
    }


def _decisions(fold: Mapping[str, Any]) -> str:
    """The decided count's own breakdown, in words, or nothing.

    Sorted by DECISION NAME rather than by count, so two reads of the same board put
    the same words in the same order: ordering by count would reshuffle the sentence
    every time one of them moved, which reads as a change that did not happen.
    """
    by_decision = fold.get("by_decision")
    if not isinstance(by_decision, Mapping):
        return ""
    rows = [
        f"{count} {name}"
        for name, count in sorted(by_decision.items())
        if isinstance(name, str) and name and _count(count)
    ]
    return f" ({', '.join(rows)})" if rows else ""


# --------------------------------------------------------------------------
# the small conversions, each with one job
# --------------------------------------------------------------------------


def _count(value: object) -> int | None:
    """*value* as a non-negative count, or ``None`` when it is not one.

    ``bool`` is excluded explicitly: it is an ``int`` subclass, so ``True`` would
    otherwise fold in as the count ``1``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _plain(value: object, noun: str) -> str:
    """``"<n> <noun>"``, or :data:`NOT_RECORDED`. No bare number ever reaches a field."""
    count = _count(value)
    return NOT_RECORDED if count is None else f"{count} {noun}"


def _text(value: object) -> str:
    """A recorded string, or :data:`NOT_RECORDED`. Bounded, because it is displayed."""
    if not isinstance(value, str) or not value.strip():
        return NOT_RECORDED
    return value.strip()[:80]


def _credits(charge: float) -> str:
    """A credit charge as a reader budgets in it, and never in scientific notation.

    ``repr`` of a small float is exponential (``1e-05``), which on a panel reads as a
    different order of magnitude than it is.

    TWO places, because this is a display field and a reader budgeting from it does not
    think in millionths. The fold keeps its six, so nothing downstream loses precision by
    this. Trailing zeros stay: "3.50" reads as money where "3.5" reads as a measurement.
    """
    if charge != charge or charge in (float("inf"), float("-inf")):
        # A non-finite total is a broken fold, not a charge. Say so rather than print
        # "nan" into the field a reader budgets from.
        return "an unreadable"
    if 0 < charge < 0.005:
        # Two places would round this to "0.00", and a charge shown as zero reads as no
        # charge at all -- the one thing a spend field must never say while spending. So
        # the SMALLEST charge is described instead of rounded, still to two places.
        return "less than 0.01"
    return f"{charge:.2f}"


def _stamp(value: object) -> str:
    """A fold's epoch-millisecond time as a local ISO spelling, or a refusal.

    The folds carry ``time`` in epoch milliseconds. A value outside the range a
    ``datetime`` holds answers :data:`NOT_RECORDED` rather than raising: these are bytes
    off a file the reader does not control, so a damaged stamp must cost one field
    rather than the whole panel, and the line stays on disk so a raise would be
    permanent for that crew.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return NOT_RECORDED
    try:
        when = datetime.fromtimestamp(value / 1000, tz=timezone.utc).astimezone()
    except (OSError, OverflowError, ValueError):
        return NOT_RECORDED
    # A human spelling, local, no offset. The raw ISO form is what the render showed under
    # "Last recorded", where an offset is noise: this value is always in the reader's own
    # zone, having just been converted into it.
    return when.strftime("%d %b %Y, %H:%M")
