"""``_SubstitutionDepth``: the case-aware successor to the paren counter.

Every argv window on the floor stops at the first token that ends the argv
while no command substitution is open.  Measured with the bare
``_substitution_depth_delta`` -- a per-token character count with no shell
grammar -- a ``case`` PATTERN's ``)`` (the token ``x)``) scores as a
substitution closer and a pattern's ``|`` (``x|y)``) as a separator, so such a
window closes at the first list operator after ``esac`` and every later clause
is lost.  These tests pin the walker on the token shapes the tokenizer
actually produces (``$(case``, ``x)``, ``(x)``, ``x|y)``, ``:;;``, ``esac;``,
``esac)``), each measured against what bash spans.
"""

from __future__ import annotations

import pytest

from kiro_crew.security import shell_normalizer
from kiro_crew.security.shell_normalizer import (
    _outside_expansions,
    _reduce_expansions,
    _substitution_depth_delta,
    _SubstitutionDepth,
    normalize_shell_command,
)

_NAME = "kiro" + "crew"


def _window(cmd: str, *, command_position: bool = False, skip: int = 1) -> list[str]:
    """The tokens a consumer window anchored after the first word would scan."""
    depth = _SubstitutionDepth(command_position=command_position)
    scanned: list[str] = []
    for token in normalize_shell_command(cmd)[skip:]:
        scanned.append(token)
        if depth.feed(token):
            break
    return scanned


class TestCasePatternDoesNotCloseTheWindow:
    """A ``kill $( ... )`` whose body ends in a lookup: the lookup stays in the window."""

    @pytest.mark.parametrize(
        "body",
        [
            # the issue's measured spelling
            "case x in x) :;; esac; pgrep -f {n}",
            # a leading ``(`` on the pattern, and the ``$( case`` split spelling
            "case x in (x) :;; esac; pgrep -f {n}",
            " case x in (x) :;; esac; pgrep -f {n}",
            # a pattern alternation: its ``|`` is not a pipe
            "case x in x|y) :;; esac; pgrep -f {n}",
            "case x in (x|y) :;; esac; pgrep -f {n}",
            # several clauses: ``;;`` re-arms the pattern each time
            "case x in x) :;; y) :;; esac; pgrep -f {n}",
            # the other clause terminators
            "case x in x) :;& y) :;;& esac; pgrep -f {n}",
            # an empty case, and a body glued to its pattern
            "case x in esac; pgrep -f {n}",
            "case x in x)pgrep -f {n};; esac",
            # a substitution and an extglob group INSIDE the pattern balance first
            "case x in $(echo x)) :;; esac; pgrep -f {n}",
            "case x in @(a|b)) :;; esac; pgrep -f {n}",
            # nested: a case inside a substitution inside a case body, and a
            # plain nested case whose ``esac;;`` closes the inner and re-arms the outer
            "case x in x) $(case y in y) :;; esac);; esac; pgrep -f {n}",
            "case x in x) case y in y) :;; esac;; esac; pgrep -f {n}",
            # case in every command position bash grants it
            "if true; then case x in x) :;; esac; fi; pgrep -f {n}",
            "while :; do case x in x) :;; esac; break; done; pgrep -f {n}",
            "function f case x in x) :;; esac; pgrep -f {n}",
            "time -p case x in x) :;; esac; pgrep -f {n}",
            "coproc case x in x) :;; esac; pgrep -f {n}",
            # ``esac`` as an ARGUMENT does not disarm; ``esac`` in command position does
            "case x in x) echo esac;; esac; pgrep -f {n}",
            # a clause body glued to its pattern is fed through the grammar too:
            # a nested ``case``, an ``esac``, a substitution opener
            "case x in x)case y in y) :;; esac;; esac; pgrep -f {n}",
            "case x in x)esac; pgrep -f {n}",
            "case x in x)$(case y in y) :;; esac);; esac; pgrep -f {n}",
            # a ``)`` inside a parameter expansion is expansion text, in the
            # pattern and in an ordinary word of the body alike
            "case x in ${{v:-x)}}) :;; esac; pgrep -f {n}",
            "case x in ${{v:-$(echo x)}}) :;; esac; pgrep -f {n}",
            "case x in ${{a:-${{b)}}}}) :;; esac; pgrep -f {n}",
            ": ${{v:-x)}}; pgrep -f {n}",
            # a control operator glued to ``case`` still opens it in command position
            ":;case x in x) :;; esac; pgrep -f {n}",
            "x&&case x in x) :;; esac; pgrep -f {n}",
            "x|case x in x) :;; esac; pgrep -f {n}",
            # the operator alone (``esac;case``): the re-fed ``;case`` re-arms the second case
            "case y in y) :;; esac;case x in x) :;; esac; pgrep -f {n}",
            "case y in y) :;; esac;case x in x) :;; esac;case z in z) :;; esac; pgrep -f {n}",
            "case y in y) :;; esac&&case x in x) :;; esac; pgrep -f {n}",
            # a clause terminator is an operator: the next pattern may be glued to it,
            # and the first pattern may be glued to ``in``
            "case x in x) :;;y) :;; esac; pgrep -f {n}",
            "case x in x) :;&y) :;; esac; pgrep -f {n}",
            "case x in x) :;;&y) :;; esac; pgrep -f {n}",
            "case x in(x) :;; esac; pgrep -f {n}",
            "case x in(x) :;;(y) :;; esac; pgrep -f {n}",
            # a quoted ``esac`` as the pattern: ``esac)`` cannot be followed by ``;;``
            # once the case is closed, so the compound reopens; ``esac|`` is a pattern
            "case x in 'esac') :;; esac; pgrep -f {n}",
            "case x in x) :;; 'esac') :;; esac; pgrep -f {n}",
            "case x in 'esac'|x) :;; esac; pgrep -f {n}",
            # ...and with an EMPTY body the terminator is glued: ``esac);;`` (R13 minted)
            "case x in 'esac');; esac; pgrep -f {n}",
            "case x in 'esac');& esac; pgrep -f {n}",
            "case esac in 'esac');; esac; pgrep -f {n}",
            # a quoted ``)`` in a position bash refuses unquoted is pattern text
            "case x in ')') :;; esac; pgrep -f {n}",
            "case x in 'x)') :;; esac; pgrep -f {n}",
            # the EMPTY pattern de-quotes to a bare ``)``: it terminates
            "case x in '') :;; esac; pgrep -f {n}",
            'case x in "") :;; esac; pgrep -f {n}',
            "case x in '))') :;; esac; pgrep -f {n}",
            "case x in (')') :;; esac; pgrep -f {n}",
            # the WORD spans tokens: ``in`` follows once its substitution closes
            "case $(echo x) in x) :;; esac; pgrep -f {n}",
            "case $(echo x y) in x) :;; esac; pgrep -f {n}",
            "case $(echo $(echo x)) in x) :;; esac; pgrep -f {n}",
        ],
    )
    def test_the_whole_body_is_one_argument(self, body: str) -> None:
        cmd = "kill $(" + body.format(n=_NAME) + ")"
        assert _window(cmd) == normalize_shell_command(cmd)[1:]

    def test_the_counter_alone_still_closes_early(self) -> None:
        # The regression this walker exists for, measured on the raw counter.
        depth = 0
        seen: list[str] = []
        for token in normalize_shell_command("kill $(case x in x) :;; esac; pgrep -f x)")[1:]:
            seen.append(token)
            depth += _substitution_depth_delta(token)
            if depth <= 0 and (";" in token or "|" in token):
                break
        assert seen == ["$(case", "x", "in", "x)", ":;;"]


class TestBoundariesOutsideACaseAreUnchanged:
    """Everything the counter got right, the walker gets right the same way."""

    @pytest.mark.parametrize(
        ("cmd", "expected"),
        [
            ("kill 123; echo $(cat /tmp/x)", ["123;"]),
            ("kill $(true; echo 1)", ["$(true;", "echo", "1)"]),
            ("kill $(pgrep -f other); case x in x) :;; esac; echo x", ["$(pgrep", "-f", "other);"]),
            # a case that STARTS after the window's own argv ended is never entered
            ("kill 123; case x in x) :;; esac; pgrep -f x", ["123;"]),
            # the window's first token is an ARGUMENT: ``case`` there is data
            ("kill case x in x) :;; esac; pgrep -f x", ["case", "x", "in", "x)", ":;;"]),
            # ``case`` after an ordinary verb, an assignment or a redirect is data
            ("kill $(echo case x in x); pgrep -f x)", ["$(echo", "case", "x", "in", "x);"]),
            ("kill $(v=1 case x in x); pgrep -f x)", ["$(v=1", "case", "x", "in", "x);"]),
            (
                "kill $(echo -n case x in x); pgrep -f x)",
                ["$(echo", "-n", "case", "x", "in", "x);"],
            ),
            # a redirect glued to ``esac`` still disarms, and the substitution's
            # own ``)`` behind it closes the window at the separator
            (
                "kill $(case x in a) :;; esac>/dev/null); echo x",
                ["$(case", "x", "in", "a)", ":;;", "esac>/dev/null);"],
            ),
            (
                "kill $(case x in a) :;; esac 2>&1); echo x",
                ["$(case", "x", "in", "a)", ":;;", "esac", "2>&1);"],
            ),
        ],
    )
    def test_window(self, cmd: str, expected: list[str]) -> None:
        assert _window(cmd) == expected

    def test_a_frame_walk_starts_in_command_position(self) -> None:
        # The rsync/ssh outer walk feeds a whole frame from its first token,
        # where ``case`` IS the reserved word: the pattern's ``|`` must not be
        # read as a pipe there either.
        cmd = "case $x in a|b) rsync -e ssh x y;; esac; echo z"
        depth = _SubstitutionDepth(command_position=True)
        boundaries = [tok for tok in normalize_shell_command(cmd) if depth.feed(tok)]
        assert boundaries == ["y;;", "esac;"]

    def test_depth_never_goes_negative(self) -> None:
        depth = _SubstitutionDepth()
        depth.feed(")")
        depth.feed(")")
        assert depth.top_level
        depth.feed("$(x")
        assert not depth.top_level
        depth.feed("y)")
        assert depth.top_level


class TestParensInsideAParameterExpansionAreText:
    """``_outside_expansions``: what the walker counts of a token with ``${ … }``."""

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("${v:-x)}", "v:-x"),
            ("${v:-x)})", "v:-x)"),
            # a ``$(`` opened inside the expansion is real, and so is its closer
            ("${v:-$(a)}b)", "v:-$(a)b)"),
            ("${v:-$(a;", "v:-$(a;"),
            ("b)}", "b)}"),
            ("${PIDS:-$(pgrep x)};", "PIDS:-$(pgrep x);"),
            ("${a:-${b)}}x)", "a:-bx)"),
            # a process substitution is the ``$( … )`` it opens
            ("<(printf", "$(printf"),
            ("2>(cat)", "2$(cat)"),
            ("<(x)", "$(x)"),
            # no expansion: the token is returned as is
            ("$(pgrep", "$(pgrep"),
            ("plain)", "plain)"),
        ],
    )
    def test_reduction(self, token: str, expected: str) -> None:
        assert _outside_expansions(token) == expected

    @pytest.mark.parametrize(
        "cmd",
        [
            # the case WORD is a process substitution: its ``)`` closes the ``<(``,
            # not the ``$(`` the window is inside of (measured: the counter let
            # ``kirocrew $(case <(printf x) in *) :;; esac) token`` mint)
            "kill $(case <(printf x) in *) :;; esac; pgrep -f x)",
            "kill $(case >(cat) in *) :;; esac; pgrep -f x)",
            # a process substitution in a clause body
            "kill $(case x in x) cat <(echo y);; esac; pgrep -f x)",
        ],
    )
    def test_a_process_substitution_does_not_end_the_window(self, cmd: str) -> None:
        assert _window(cmd) == normalize_shell_command(cmd)[1:]

    def test_a_balanced_process_substitution_still_ends_the_argv(self) -> None:
        cmd = "kill $(pgrep x) <(echo); echo y"
        assert _window(cmd) == ["$(pgrep", "x)", "<(echo);"]

    def test_an_expansion_closer_does_not_end_the_window(self) -> None:
        # ``kill $(: ${v:-x)}; pgrep -f <name>)``: the body is one argument.
        cmd = "kill $(: ${v:-x)}; pgrep -f x)"
        assert _window(cmd) == normalize_shell_command(cmd)[1:]

    def test_a_real_closer_after_an_expansion_still_ends_it(self) -> None:
        cmd = "kill ${PIDS:-$(pgrep x; echo 1)}; echo x"
        assert _window(cmd) == ["${PIDS:-$(pgrep", "x;", "echo", "1)};"]


class TestParensInsideABracketArePatternText:
    """A ``)`` between ``[`` and a later ``]`` was quoted: bash refuses ``[)]`` unquoted."""

    @pytest.mark.parametrize(
        "cmd",
        [
            # measured on R10: ``x[\\)]|x)`` de-quotes to ``x[)]|x)`` and the first
            # ``)`` closed the pattern, so the window ended at the ``)`` after ``]|x``
            "kill $(case x in x[\\)]|x) :;; esac; pgrep -f x)",
            "kill $(case x in x[\\)]) :;; esac; pgrep -f x)",
            "kill $(case x in [\\)]) :;; esac; pgrep -f x)",
            # ``!``/``^`` negation and a leading ``]`` are bracket text
            "kill $(case x in [!\\)]) :;; esac; pgrep -f x)",
            "kill $(case x in [^\\)]) :;; esac; pgrep -f x)",
            "kill $(case x in []\\)]) :;; esac; pgrep -f x)",
            # two quoted parens, a range, and a character class inside the bracket
            "kill $(case x in x[\\)y\\)]) :;; esac; pgrep -f x)",
            "kill $(case x in x[a-z\\)]|y) :;; esac; pgrep -f x)",
            "kill $(case x in [[:alpha:]\\)]) :;; esac; pgrep -f x)",
            # a quoted ``(`` inside the bracket opens no group (bash refuses ``[(]``)
            "kill $(case x in [\\(]) :;; esac; pgrep -f x)",
        ],
    )
    def test_the_bracket_does_not_end_the_window(self, cmd: str) -> None:
        assert _window(cmd) == normalize_shell_command(cmd)[1:]

    @pytest.mark.parametrize(
        ("cmd", "last"),
        [
            # a bracket without a ``)`` inside changes nothing
            ("kill $(case x in [a-z]) :;; esac); echo x", "esac);"),
            ("kill $(case x in [[:alpha:]]) :;; esac); echo x", "esac);"),
            ("kill $(case x in []a]) :;; esac); echo x", "esac);"),
            # an unclosed ``[`` is a literal, so the ``)`` after it still terminates
            ("kill $(case x in x[) :;; esac); echo x", "esac);"),
            # the bracket closes at its ``]`` and the next ``)`` terminates the pattern
            ("kill $(case x in x[\\)]) :;; esac); echo x", "esac);"),
            # a ``]`` inside a nested ``$( )`` does not close the bracket
            ("kill $(case x in [$(echo ])]) :;; esac); echo x", "esac);"),
        ],
    )
    def test_the_pattern_still_terminates(self, cmd: str, last: str) -> None:
        assert _window(cmd)[-1] == last


class TestAnExpansionSplitAcrossTokensIsOneWord:
    """A blank inside ``${ … }`` splits it over tokens; bash reads on to the ``}``.

    Token-local reduction forgot the open brace, so ``y)};;`` scored its ``)`` as
    a closer and the window ended at the ``;;`` (measured: ``<name> $(case x in x)
    : ${v:-x y)};; esac; :) <verb>`` minted on the R9 head).  The walker now
    carries the token until the brace closes.
    """

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("${v:-x", ("v:-x", 1)),
            ("${v:-${w", ("v:-w", 2)),
            ("y)};;", ("y)};;", 0)),
            ("${v:-x y)}", ("v:-x y", 0)),
            ("plain", ("plain", 0)),
        ],
    )
    def test_open_braces_are_reported(self, token: str, expected: tuple[str, int]) -> None:
        assert _reduce_expansions(token) == expected

    @pytest.mark.parametrize(
        "cmd",
        [
            # in a clause body
            "kill $(case x in x) : ${v:-x y)};; esac; pgrep -f x)",
            # as the case WORD
            "kill $(case ${v:-x y)} in x) :;; esac; pgrep -f x)",
            # outside any case
            "kill $(: ${v:-x y)}; pgrep -f x)",
            # over three tokens, and nested
            "kill $(: ${v:-x y z)}; pgrep -f x)",
            "kill $(: ${v:-${w:-x y)}}; pgrep -f x)",
        ],
    )
    def test_the_body_stays_one_argument(self, cmd: str) -> None:
        assert _window(cmd) == normalize_shell_command(cmd)[1:]

    def test_the_word_ends_where_the_brace_closes(self) -> None:
        cmd = "kill ${v:-x y)}; echo x"
        assert _window(cmd) == ["${v:-x", "y)};"]

    def test_a_carried_word_does_not_arm_a_pattern_early(self) -> None:
        # the ``;;`` inside the expansion is text, the one after it is the clause end
        cmd = "kill $(case x in x) : ${v:-x ;;} ;; esac; pgrep -f x)"
        assert _window(cmd) == normalize_shell_command(cmd)[1:]


class TestAnOpenExpansionIsBounded:
    """A ``${`` that never closes costs each token once and, at top level, nothing."""

    def test_each_token_of_a_split_word_is_reduced_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # R10 re-reduced the JOINED word on every feed: ~10,000 one-char tokens after an
        # open ``${`` inside a substitution were ~10^8 character steps on the event loop.
        seen: list[int] = []
        real = shell_normalizer._reduce_from

        def counting(token: str, braces: int, subs: list[int]) -> "tuple[str, int, list[int]]":
            seen.append(len(token))
            return real(token, braces, subs)

        monkeypatch.setattr(shell_normalizer, "_reduce_from", counting)
        tokens = ["$(:", "${"] + ["a"] * 2000
        depth = _SubstitutionDepth()
        for token in tokens:
            assert depth.feed(token) is False
        assert not depth.top_level
        assert sum(seen) <= 2 * sum(len(token) for token in tokens)

    @pytest.mark.parametrize(
        ("cmd", "expected"),
        [
            # a quoted ``'${'`` literal at top level: the argv still ends at the ``;``
            ("kill '${' 1; echo x", ["${", "1;"]),
            ("pkill -f '${a' ; echo x", ["-f", "${a", ";"]),
            # ``$case`` is a parameter, not the reserved word
            ("pkill -f $case ; echo x", ["-f", "$case", ";"]),
        ],
    )
    def test_top_level_shapes_end_the_argv(self, cmd: str, expected: list[str]) -> None:
        assert _window(cmd) == expected

    def test_a_split_expansion_inside_a_substitution_is_still_one_word(self) -> None:
        cmd = "kill $(: ${v:-x y)}; pgrep -f x)"
        assert _window(cmd) == normalize_shell_command(cmd)[1:]

    @pytest.mark.parametrize(
        ("cmd", "expected"),
        [
            # no ``}`` follows anywhere, so bash would refuse an open ``${`` (measured):
            # the quoted literal's ``)`` closes the substitution and the argv ends at ``;``
            ("kill $(pgrep -f 'sync-${ENV'); echo x", ["$(pgrep", "-f", "sync-${ENV);"]),
            ("kill $(pgrep -f 'a${b' 'c)d'); echo x", ["$(pgrep", "-f", "a${b", "c)d);"]),
            # a ``}`` still to come keeps the ``)`` as text: the expansion is real
            ("kill $(: ${v:-x) y}; pgrep -f x)", None),
            ("kill $(: ${v:-x); pgrep -f x}; echo y)", None),
            # a ``}`` with NO ``)`` after it cannot be that closer: the ``$(`` would stay
            # open too (bash refuses it, measured), so the ``${`` was quoted (R12 denied)
            ("kill $(grep -c '${' x); awk '{print}' y; echo x", ["$(grep", "-c", "${", "x);"]),
        ],
    )
    def test_an_unclosable_brace_is_quoted_text(
        self, cmd: str, expected: "list[str] | None"
    ) -> None:
        tokens = normalize_shell_command(cmd)
        depth = _SubstitutionDepth(rest=tokens[1:])
        scanned: list[str] = []
        for token in tokens[1:]:
            scanned.append(token)
            if depth.feed(token):
                break
        assert scanned == (expected if expected is not None else tokens[1:])

    def test_a_backtick_case_opens_the_substitution_it_is_in(self) -> None:
        # ``kirocrew `case x in x) :;; esac; :` token`` runs ``kirocrew token``; the
        # counter scores one backtick as 0, so the window closed at ``:;;`` (R10)
        cmd = "kill `case x in x) :;; esac; :` x"
        assert _window(cmd) == normalize_shell_command(cmd)[1:]


class TestGrammarTokensAreNotOperands:
    """``grammar_next`` is True exactly while the next token is the case WORD, ``in``
    or a PATTERN -- the tokens bash never hands to the command.  The ssh-family walk
    read a ``*)`` pattern as a host that matches every self name (scope rows)."""

    def test_the_word_in_and_patterns_are_grammar(self) -> None:
        depth = _SubstitutionDepth()
        assert depth.feed("$(case") is False and depth.grammar_next
        tokens = ["x", "in", "*)", "echo", "b;;", "y|z)", ":;;", "esac)", "host:/x"]
        grammar = [True, True, True, False, False, True, False, True, False]
        for token, expected in zip(tokens, grammar):
            assert depth.grammar_next is expected, token
            depth.feed(token)
        assert not depth.grammar_next

    def test_outside_a_case_nothing_is_grammar(self) -> None:
        depth = _SubstitutionDepth()
        for token in ["$(echo", "-v)", "localhost", "case", "x)"]:
            assert not depth.grammar_next, token
            depth.feed(token)


class TestGluedClausesAreFedInALoop:
    """A run of glued ``x);;`` clauses is ONE token; R12 recursed twice per clause and a
    2 KB command raised ``RecursionError`` out of the gate instead of a verdict."""

    def test_two_thousand_glued_clauses(self) -> None:
        depth = _SubstitutionDepth()
        for token in ["$(case", "x", "in"]:
            depth.feed(token)
        assert depth.feed("x);;" * 2000 + "esac)") is False
        assert depth.top_level
        assert depth.feed("x;") is True

    def test_a_long_glued_esac_chain(self) -> None:
        depth = _SubstitutionDepth()
        for token in ["$(case", "x", "in"] + ["x)case", "y", "in"] * 600:
            depth.feed(token)
        assert depth.feed("x)" + ":;;esac;" * 601) is False
        assert depth.depth == 1

    def test_an_empty_token_is_still_fed_once(self) -> None:
        """A frame keeps an empty word (``""``) as its own token, and ``_ends_argv("")``
        is a boundary: the R13 loop skipped the token, so the self-protection
        floor read ``<cli> "" restart`` as a restart (CI shard 7)."""
        assert _SubstitutionDepth().feed("") is True
        assert _window('prog "" restart') == [""]
        depth = _SubstitutionDepth()
        for token in ["$(case", "x", "in", '"")', ":;;", "esac"]:
            assert depth.feed(token) is False, token  # an empty pattern is still a pattern
        assert depth.depth == 1 and not depth.top_level


class TestABacktickInsideAPatternIsASubstitution:
    """Between a pattern's backticks a ``)`` is text: a backtick pattern with a quoted
    ``)`` de-quotes to a token whose first ``)`` sits inside the backticks (R12 minted)."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "kill $(case x in `printf ')'`) :;; esac; pgrep -f x)",
            "kill $(case x in `echo x`) :;; esac; pgrep -f x)",
            "kill $(case x in a|`printf ')'`) :;; esac; pgrep -f x)",
            "kill $(case x in `printf ')'`|`printf ')'`) :;; esac; pgrep -f x)",
        ],
    )
    def test_the_body_stays_one_argument(self, cmd: str) -> None:
        assert _window(cmd) == normalize_shell_command(cmd)[1:]

    def test_the_pattern_still_terminates(self) -> None:
        depth = _SubstitutionDepth()
        for token in ["$(case", "x", "in", "`echo", "x`)", ":;;", "esac)"]:
            depth.feed(token)
        assert depth.top_level


class TestAPrefixedOpenerArmsTheCase:
    """``$x$(case`` opens a compound the way ``$(case`` does -- bash refuses every other
    ``(case`` spelling (``$x(case``, measured), so the prefix never changes the reading."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "kill $x$(case y in y) :;; esac; pgrep -f x)",
            "kill x$(case y in y) :;; esac; pgrep -f x)",
            "kill $x`case y in y) :;; esac; pgrep -f x`",
            "kill $(:)$(case y in y) :;; esac; pgrep -f x)",
        ],
    )
    def test_the_body_stays_one_argument(self, cmd: str) -> None:
        assert _window(cmd) == normalize_shell_command(cmd)[1:]

    def test_a_word_ending_in_case_is_not_the_reserved_word(self) -> None:
        assert _window("pkill -f showcase ; echo x") == ["-f", "showcase", ";"]
        assert _window("kill $(echo showcase); echo x") == ["$(echo", "showcase);"]


class TestNewlinesInsideTheGrammarAreTransparent:
    """A frame renders a newline as a standalone ``;``; the case state holds across it."""

    @pytest.mark.parametrize(
        "frame",
        [
            # newline between the WORD and ``in``, one and two of them
            ["$(case", "x", ";", "in", "x)", ":;;", "esac;", "pgrep", "-f", "x)"],
            ["$(case", "x", ";", ";", "in", "x)", ":;;", "esac;", "pgrep", "-f", "x)"],
            # newline between ``in`` and the first pattern, and between ``;;`` and the next
            ["$(case", "x", "in", ";", "x)", ":;;", "esac;", "pgrep", "-f", "x)"],
            ["$(case", "x", "in", "x)", ":;;", ";", "y)", ":;;", "esac;", "pgrep", "-f", "x)"],
            # newline between ``;;`` and ``esac``: ``esac`` still reads as the word
            ["$(case", "x", "in", "x)", ":;;", ";", "esac;", "pgrep", "-f", "x)"],
        ],
    )
    def test_the_body_stays_one_argument(self, frame: list[str]) -> None:
        depth = _SubstitutionDepth()
        ended = [depth.feed(tok) for tok in frame]
        assert ended == [False] * len(frame)
        assert depth.top_level

    def test_esac_after_a_newline_closes_the_case_for_real(self) -> None:
        # ``kill $(case x in x) :;;<newline>esac); echo <name>``: the ``esac)``
        # must close the compound AND the substitution, so the window ends at ``;``.
        depth = _SubstitutionDepth()
        frame = ["$(case", "x", "in", "x)", ":;;", ";", "esac);", "echo", "x"]
        ended = [depth.feed(tok) for tok in frame]
        assert ended == [False, False, False, False, False, False, True, False, False]
        assert depth.top_level


class TestDataTokensStillAdvanceAPendingPattern:
    """``feed_data``: a caller's data token is inert unless bash reads it as the pattern."""

    def test_outside_a_case_nothing_changes(self) -> None:
        depth = _SubstitutionDepth()
        depth.feed("$(x")
        assert depth.feed_data("print(1); y)") is False
        assert depth.depth == 1  # the payload's parens and separator are data

    def test_an_option_shaped_pattern_is_still_the_pattern(self) -> None:
        # ``case $1 in -c) echo hi;; esac); grep x`` -- an operand scan reads
        # ``-c)`` as an option and would skip it; the pattern must close anyway.
        depth = _SubstitutionDepth()
        for tok in ["$(case", "$1", "in"]:
            depth.feed(tok)
        assert depth.feed_data("-c)") is False
        ended = [depth.feed(tok) for tok in ["echo", "hi;;", "esac);"]]
        assert ended == [False, False, True]
        assert depth.top_level


class TestEsacReArms:
    """After ``esac`` the walker is out of the case: a later ``)`` closes for real."""

    def test_a_closer_after_esac_counts(self) -> None:
        depth = _SubstitutionDepth()
        for token in normalize_shell_command("kill $(case x in x) :;; esac)")[1:]:
            ended = depth.feed(token)
        assert depth.top_level
        assert ended is False

    def test_a_pattern_after_esac_is_a_closer_again(self) -> None:
        # Once the compound is closed, ``x)`` is what it always was to the counter.
        depth = _SubstitutionDepth()
        for token in ["$(case", "x", "in", "x)", ":;;", "esac;", "$(echo", "x)"]:
            depth.feed(token)
        assert depth.depth == 1
        depth.feed("x)")
        assert depth.top_level
