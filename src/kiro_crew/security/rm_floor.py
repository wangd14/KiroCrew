"""Argv-structural floor for recursive-force ``rm`` deletion of root / home.

Split out of ``argv_floor.py`` as a cohesive sibling: this module owns the
recursive-force ``rm`` deny floor and nothing else. ``argv_floor.is_denied``'s
caller reaches it through :func:`_recursive_force_rm_targets` (and the
fail-closed fallback), which read only the ``rm`` command's OWN argv and return
the catastrophic target set ``{"root", "home"}`` a command deletes.

Enforcement lives here rather than in the regex tier: a whole-line text pattern
cannot tell an ``rm`` operand from the same text quoted in a ``git commit -m``
message or a ``grep`` pattern, so the two ``rm`` catalog patterns are stripped
from the regex tier in ``is_denied`` and this floor is their sole enforcement.
"""

from __future__ import annotations

import re

from . import shell_normalizer as _shell_normalizer
from .shell_normalizer import (
    _DATA_CONSUMER_PROGRAMS,
    _argv_programs,
    _data_consumer_exempt,
    _decode_shell_quoted_literals,
    _ends_argv,
    _program_basename,
    _shell_payload_walk,
    _split_shell_words,
    _substitution_depth_delta,
)

# ── Recursive-force ``rm`` deletion floor ──
# ``rm`` recursively force-deleting the filesystem ROOT or the user's HOME is
# catastrophic; a path UNDER either (``/tmp/scratch``, ``$HOME/.cache``) is an
# ordinary cleanup and must stay allowed. The catalog literals ``rm -rf /`` /
# ``rm -rf ~`` only matched one flag spelling, and every attempt to widen them
# as a REGEX went wrong two ways at once:
#   * a left-to-right pattern cannot see flags AFTER the operand, which GNU
#     ``getopt`` accepts (``rm / -rf --no-preserve-root``); and
#   * a text pattern matches a SUBSTRING of the whole command line, so it fired
#     on ``/tmp/x`` (a descendant of ``/``) and on the words ``rm -fr /`` sitting
#     inside a ``git commit -m`` message or a ``grep`` pattern.
# The only sound closure is argv-STRUCTURAL and EXACT, like the self-protection
# and git-publish floors: tokenize, look only at the ``rm`` command's OWN argv,
# collect the flags from every position, and deny only when a resolved operand
# IS the root or the home directory itself — never a descendant, never a text
# mention. This floor is therefore the SOLE enforcement (its catalog patterns are
# stripped from the regex tier in ``is_denied``, exactly as git-publish is), so
# there is no whole-line text match left to fire on a commit message.
#
# The tokens come from ``_split_shell_words`` — the RAW, quote-resolved but
# ENV-UNEXPANDED split — for two reasons the review named: (1) a home operand
# must be classified by its written spelling (``~`` / ``$HOME`` / ``${HOME}``),
# because the expanding tokenizer turns ``$HOME`` into ``/home/user`` which then
# reads as ROOT, inverting the home rule's opt-out (GPT + Opus finding); and
# (2) it keeps the classification on what the argv literally is.
#
# The floor fires only for a command whose PROGRAM is ``rm`` (``_argv_programs``
# tracks command boundaries), so ``confirm -rf /``, an ``rm`` mentioned as data
# (``echo rm -rf /``), and a sibling command's flags (``ls -rf; rm /tmp/x``) do
# not trigger it.


#: ``rm``'s long options, so an abbreviation can be tested for ambiguity. GNU
#: ``getopt_long`` accepts any UNAMBIGUOUS prefix of a long option, so ``rm
#: --rec …`` and ``rm --for …`` run the identical recursive/force delete while a
#: fixed ``--recursive``/``--force`` string comparison would miss them (GPT
#: security-class). A prefix is honoured only when it matches exactly ONE
#: of ``rm``'s long options — ``--r`` resolves to ``--recursive`` (nothing else
#: begins with ``r``), ``--f`` to ``--force`` — never a prefix shared by two.
_RM_LONG_OPTIONS: tuple[str, ...] = (
    "--recursive",
    "--force",
    "--dir",
    "--interactive",
    "--no-preserve-root",
    "--one-file-system",
    "--preserve-root",
    "--verbose",
    "--help",
    "--version",
)


def _rm_long_option_resolves_to(tok: str, target: str) -> bool:
    """Whether *tok* is an unambiguous long-option abbreviation of *target*.

    *tok* must be ``--`` followed by a NON-EMPTY prefix (``--`` alone is the
    end-of-options marker, handled elsewhere), and among ``rm``'s long options
    exactly one must start with that prefix, and it must be *target*. An exact
    spelling is trivially unambiguous. GNU stops at the first ``=`` (``--rec=…``),
    so the option name is taken up to it.
    """
    if not tok.startswith("--") or tok == "--":
        return False
    name = tok[: tok.index("=")] if "=" in tok else tok
    matches = [opt for opt in _RM_LONG_OPTIONS if opt.startswith(name)]
    return matches == [target] or (target in matches and name == target)


#: Whether an ``rm`` argument token carries the recursive flag: the long option
#: ``--recursive`` (or an unambiguous prefix of it), or a single-dash short
#: cluster containing ``r`` (``-r`` / ``-rf`` / ``-fr`` / ``-rfv`` …). A ``--``
#: long option is never read as a short cluster, so ``--force`` is not recursive.
def _rm_is_recursive_flag(tok: str) -> bool:
    if tok.startswith("--"):
        return _rm_long_option_resolves_to(tok, "--recursive")
    return bool(re.fullmatch(r"-[a-z]*r[a-z]*", tok))


#: Whether an ``rm`` argument token carries the force flag (``--force`` or an
#: unambiguous prefix of it, or a single-dash short cluster containing ``f``).
def _rm_is_force_flag(tok: str) -> bool:
    if tok.startswith("--"):
        return _rm_long_option_resolves_to(tok, "--force")
    return bool(re.fullmatch(r"-[a-z]*f[a-z]*", tok))


#: The filesystem ROOT or any absolute path UNDER it — ``/``, ``/tmp/x``,
#: ``/etc``, the ``/*`` glob — matching base ``main``'s ``rm -rf /.*`` contract.
#: Used ONLY for a DIRECT ``rm`` (at program position or a dispatcher applet):
#: base ``main`` denied a recursive-force delete of the root OR any path under it
#: (``test_builtin_denied_by_default`` pins ``rm -rf /tmp/foo`` as DENY), and
#: closing the flag-spelling gap must only ADD denies. A wrapper-reached ``rm`` uses the narrower
#: ``_RM_ROOT_ITSELF_RE`` instead (see the classifier).
_RM_ROOT_DESCENDANT_RE = re.compile(r"/.*", re.DOTALL)
#: The HOME directory or any path under it — ``~`` / ``$HOME`` / ``${HOME}`` and
#: anything after it (``~/.cache``, ``$HOME/x``, the ``~/*`` glob) — base's
#: ``rm -rf ~.*`` widened to the ``$HOME`` spellings, DIRECT ``rm`` only.
#: A variable-name boundary after bare ``$home`` keeps an unrelated variable such
#: as ``$HOME_BACKUP`` from matching; ``${home}`` is delimited by its own ``}``.
_RM_HOME_DESCENDANT_RE = re.compile(
    r"(?:~|\$\{home\}|\$home(?![a-z0-9_])).*", re.IGNORECASE | re.DOTALL
)
#: The filesystem ROOT ITSELF — ``/`` (a run of slashes) or the ``/*`` glob over
#: its children, with an optional trailing slash, and NOTHING under it. Used for
#: an ``rm`` reached through an EXEC WRAPPER (``setsid rm -rf /``, ``sudo …``):
#: base ``main`` caught a wrapper-reached descendant only incidentally, as a
#: substring of its ``rm -rf /`` literal, so denying wrapper-reached DESCENDANTS
#: newly refuses benign work the widening never intended (``docker exec kc-ci
#: rm -fr /tmp/build-cache`` — a container cache cleanup base ALLOWED, since its
#: literal is ``rm -rf`` not ``rm -fr``; Security Scope ruling). So a
#: wrapper-reached ``rm`` denies only the catastrophic root ITSELF.
_RM_ROOT_ITSELF_RE = re.compile(r"/+(?:\*/*)?")
#: The HOME dir ITSELF — ``~`` / ``$HOME`` / ``${HOME}``, bare or with a RUN of
#: trailing slashes (``~//`` / ``~///``) or the ``~/*`` glob, nothing under it.
#: A path-collapsing shell treats ``~//`` and ``~///`` as home, so any run of
#: trailing slashes is accepted; the glob ``*`` is
#: admitted only as the whole remainder after the slashes. Wrapper-reached only.
_RM_HOME_ITSELF_RE = re.compile(
    r"(?:~|\$\{home\}|\$home(?![a-z0-9_]))(?:/+(?:\*/*)?)?", re.IGNORECASE
)
#: Escape / quote / substitution characters that can reconstruct the ``rm``
#: program name from text that does not contain the literal ``rm`` (a folded
#: ``"r\<nl>m"``, an octal ``$'r\555'``). The cheap pre-filter admits a command
#: carrying any of these so the walk gets a chance to decode it.
_RM_OBFUSCATION_MACHINERY_RE = re.compile(r"[\\$`'\"]")

#: Ceiling on how many nested-frame descents ONE top-level classification may
#: make. Each ``find -exec`` / ``sh -c`` / interpreter-code span classified as
#: its own argv recurses back into :func:`_rm_targets_in_argv`, so a crafted
#: nest (``sh -c 'sh -c 'sh -c … rm -rf /'''``) would fan out and hang the
#: SYNCHRONOUS PreToolUse gate — measured seconds on a ~120-byte command (Opus
#: security-class). The budget is a single mutable cell threaded through the
#: recursion and decremented on every descent; once it reaches zero no further
#: nested span is opened, so the total work is linear in the cap regardless of
#: nesting depth. It fails SAFE: a real ``rm`` at any reachable depth is already
#: classified by the frames the walk visits BEFORE the cap bites (a genuine
#: nested wipe denies at the shallow frame that carries it), and the raw tier
#: still sees the whole command text — so the cap drops only pathological
#: deep-nest coverage, never a shallow real target. 64 is far past any real
#: command's nesting yet bounds a hostile one to a few milliseconds.
_RM_DESCENT_BUDGET = 64

#: Shell control operators that END a command's argv when they appear UNQUOTED
#: — a glued one (``/;reboot``, ``/&&id``) leaves the real operand before it, so
#: an operand token is classified only up to the first of these and the argv
#: ends there. ``&`` covers ``&`` and ``&&``; ``|`` covers ``|`` and ``||``.
_RM_OPERAND_BOUNDARY_RE = re.compile(r"[;&|\n]")


def _rm_operand_before_boundary(operand: str) -> "tuple[str, bool]":
    """The operand text up to its first unquoted control-operator boundary.

    Returns ``(head, ended)``: *head* is the operand with everything from the
    first ``;`` / ``&`` / ``|`` / newline onward removed, and *ended* is True
    when such a boundary was present. The tokens reaching here have already had
    their quotes resolved (raw split) or normalized away (decoded view), so a
    remaining operator character is unquoted and genuinely separates commands —
    ``rm -rf /;reboot`` tokenizes to the single operand ``/;reboot`` whose real
    target is ``/``. Splitting here classifies that
    ``/`` and stops the argv, so a command glued after the boundary is neither
    read as another rm operand nor able to hide the target before it.
    """
    match = _RM_OPERAND_BOUNDARY_RE.search(operand)
    if match is None:
        return operand, False
    return operand[: match.start()], True


def _rm_strip_surrounding_quotes(token: str) -> str:
    """Peel balanced surrounding quote pairs from a raw operand token.

    ``_split_shell_words`` leaves a quoted operand quoted (``"$home"``), so an
    exact operand match needs the wrapper removed. Only a matching leading and
    trailing quote of the same kind is peeled, to a fixed point, so an operand
    that merely CONTAINS a quote is left alone.
    """
    previous = None
    while token != previous:
        previous = token
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            token = token[1:-1]
    return token


def _rm_walk_frames(text_lower: str, raw_text: "str | None") -> "list[tuple[str, list[str], bool]]":
    """``(source, norm_tokens, repaired)`` frames for the rm floor to classify.

    The first block is ``_shell_payload_walk(text_lower)`` — the ordinary
    lowercased walk. When *raw_text* is supplied AND carries an ANSI-C span, a
    SECOND block is the walk of that text with its ``$'…'`` spans decoded
    (case-preserved) then lowercased, so the width-sensitive ``\\U`` unicode
    escape resolves (``is_denied`` lowercases first, which would turn ``\\U`` into
    ``\\u`` and truncate the read at 4 digits).

    ``repaired`` marks the second block, and the caller uses it to classify a
    repaired frame ONLY through its decoded ``norm_tokens`` — never through a
    quote-stripping raw split. ``_decode_shell_quoted_literals`` re-quotes an
    ANSI-C value with ``shlex.quote`` (``$'\\"/\\"'`` -> ``'"/"'``), so a raw
    split of the repaired source would strip BOTH the added shell quotes and the
    LITERAL quotes the decode produced, reading the filename ``"/"`` as the root.
    The original-text walk (always included) is where a raw ``$HOME`` / ``~`` home
    operand is classified, so the repaired block loses no coverage by skipping it.
    Deduplicated: the ordinary command (no ``$'…'``, or the decode changes
    nothing) yields only the first block.
    """
    frames: "list[tuple[str, list[str], bool]]" = [
        (source, toks, False) for source, toks in _shell_payload_walk(text_lower)
    ]
    if raw_text is not None and "$'" in raw_text:
        repaired = _decode_shell_quoted_literals(raw_text).lower()
        if repaired != text_lower:
            frames.extend((source, toks, True) for source, toks in _shell_payload_walk(repaired))
    return frames


def _recursive_force_rm_targets(
    text_lower: str, *, raw_text: "str | None" = None
) -> "frozenset[str]":
    """Which catastrophic target(s) a top-level ``rm`` recursively force-deletes.

    Returns a subset of ``{"root", "home"}`` — ``root`` when a resolved operand
    IS the filesystem root, ``home`` when one IS the home directory (by ``~`` or
    the ``$HOME`` variable). Empty when the command is not a recursive-force
    ``rm`` against such an EXACT target; a descendant (``/tmp/x``,
    ``$HOME/.cache``) and a mere text mention both return empty.

    ``--no-preserve-root`` is a trigger on its own (meaningless without ``-rf``,
    and its whole purpose is to defeat the ``/`` guard); otherwise BOTH a
    recursive and a force flag must be present, in any position. A ``--``
    end-of-options marker stops flag parsing, so a token after it is an operand
    even if it is dash-shaped — matching GNU ``rm``.

    Every command FRAME is inspected — the top-level argv and the argv of every
    nested shell payload (``bash -c '…'``, ``sh -c``, ``$(…)``, a here-string, a
    chained segment). Each frame is re-split from its RAW source with
    ``_split_shell_words`` (quote-resolved but ENV-UNEXPANDED), so ``$HOME`` is
    classified by its written form rather than the home path a shlex expansion
    would produce (which would read as root). This is the same payload descent
    the self-protection floor uses, so a wrapper (``sudo rm -rf /``), a nested
    script (``bash -c 'rm -rf /'``) and a chain (``… && rm -rf /``) are all
    reached, while the frame's own ``_argv_programs`` scoping keeps a string that
    is merely an argument to another program (a ``git commit -m`` message, a
    ``grep`` pattern) from ever being read as an ``rm`` command.

    *raw_text* is the ORIGINAL-case command, when the caller has it. Bash's
    ANSI-C unicode escapes are CASE-SENSITIVE in width (``\\u`` is 4 hex digits,
    ``\\U`` is 8), so a ``$'\\U0000002d…'`` spelling decodes correctly only from
    case-preserved text -- the lowercased ``\\u`` truncates at 4 digits and reads
    the wrong character. When *raw_text* is supplied its ANSI-C spans are decoded
    (case-preserved) then lowercased and walked as an ADDITIONAL frame source, so
    the ``\\U`` spelling is caught the same as its ``\\u`` twin.
    """
    # Cheap necessary condition. A plain ``rm`` invocation contains the literal
    # ``rm``; an OBFUSCATED one (``"r\<nl>m"``, ``$'r\555'``) does not — its ``rm``
    # is built by escape/quote/substitution machinery whose decoded output can be
    # any character, so the only sound cheap gate is "contains ``rm`` OR contains
    # such machinery". When neither is present the walk cannot yield an ``rm``.
    if "rm" not in text_lower and not _RM_OBFUSCATION_MACHINERY_RE.search(text_lower):
        return frozenset()
    found: set[str] = set()
    for source, norm_tokens, repaired in _rm_walk_frames(text_lower, raw_text):
        # The DECODED view (payload walk's own tokens) is always classified: it
        # resolves ANSI-C / unicode escapes and env expansion, so ``rm -rf $'/'``
        # / ``$'\u002f'`` is caught as the exact root, and a ``$'"/"'`` filename's
        # LITERAL quotes stay in the token so it is NOT misread as root.
        found |= _rm_targets_in_argv(norm_tokens, strip_quotes=False)
        # A REPAIRED frame (from the ANSI-C-decoded copy) is classified ONLY via
        # its decoded tokens above. Its raw source has been through
        # ``_decode_shell_quoted_literals`` + ``shlex.quote``, so a quote-stripping
        # raw split would peel the shell quotes shlex added AND the LITERAL quotes
        # the decode produced (``$'"/"'`` -> ``'"/"'`` -> ``/``), reading a
        # filename as the root. The raw-spelling ``$HOME`` / ``~`` classification
        # it would otherwise add is already covered by the ORIGINAL-text frame.
        if repaired:
            if {"root", "home"} <= found:
                break
            continue
        # Non-repaired frame: also classify the RAW split, which keeps ``$HOME`` /
        # ``~`` unexpanded so home is classified by its written spelling. Surrounding
        # SHELL quotes are stripped only here (``"$HOME"`` -> ``$HOME``).
        found |= _rm_targets_in_argv(_split_shell_words(source), strip_quotes=True)
        # Execution-substitution bodies the shared walk does not surface as their
        # own frames: a ``$(…)`` / backtick command substitution nested INSIDE a
        # double-quoted argument (the enclosing quote makes the closing ``)``
        # quote-inactive, so ``_substitution_bodies`` over-reads it), a bash 5.3
        # ``${ …;}`` funsub, and a bare ``(…)`` subshell. Each EXECUTES the command
        # it carries, so an ``rm`` inside one is a real wipe even when the
        # substitution's OUTPUT is then consumed as data (``grep -rn "$(rm -rf /)"
        # test/`` runs the wipe before grep starts). Each extracted body is
        # classified as its own argv — flag order, the exact-operand test and the
        # glob shape all apply inside it.
        for body in _rm_exec_substitution_bodies(source):
            found |= _rm_targets_in_argv(_split_shell_words(body), strip_quotes=True)
        if {"root", "home"} <= found:
            break
    return frozenset(found)


#: The BASE-LITERAL spellings of the two ``rm`` deny rules, as whole-line
#: patterns (``rm -rf /`` / ``rm -rf ~`` followed by anything). They are the
#: FAIL-CLOSED fallback: the structural floor above needs a tokenizer, and if
#: that tokenizer RAISES the floor must still deny the one spelling the
#: bare-literal rule denies WITHOUT any tokenizer — a text ``re.search`` over the
#: lowercased command. base ``main`` denies ``rm -rf /`` even with no tokenizer,
#: so a tokenizer hiccup must not turn that into an ALLOW (First Principles
#: items 5+6). This recovers ONLY the exact base spelling, not the
#: widened flag/glob/obfuscation coverage — those depend on the tokenizer and
#: are simply unavailable when it breaks; the point is that the floor never fails
#: OPEN on the catastrophic literal.
_RM_ROOT_LITERAL_RE = re.compile(r"rm -rf /")
_RM_HOME_LITERAL_RE = re.compile(r"rm -rf ~")


def _recursive_force_rm_targets_fail_closed(text_lower: str) -> "frozenset[str]":
    """Base-literal ``rm`` targets, for when the structural tokenizer RAISED.

    Applies the pre-widening bare-literal check (``rm -rf /`` / ``rm -rf ~`` as a
    substring of the lowercased command) with NO tokenization, so the floor
    denies the catastrophic literal even when :func:`_recursive_force_rm_targets`
    could not run. This is deliberately the SAME shape ``main``'s deny rule had
    before this change, so the fail path is no weaker than base was.
    """
    found: set[str] = set()
    if _RM_ROOT_LITERAL_RE.search(text_lower):
        found.add("root")
    if _RM_HOME_LITERAL_RE.search(text_lower):
        found.add("home")
    return frozenset(found)


#: A bash 5.3 command funsub: ``${ COMMANDS; }`` / ``${|COMMANDS; }`` — runs the
#: commands in the current shell (unlike ``$(…)``, no subshell). The body runs to
#: the matching ``}``; the leading ``|`` (value-returning form) and a trailing
#: ``;`` are stripped when the body is classified.
_FUNSUB_OPEN_RE = re.compile(r"\$\{[ \t\n|]")


def _index_in_single_quote(source: str, index: int) -> bool:
    """True if *index* falls inside a single-quoted span of *source*.

    A single quote in bash suppresses every expansion, so a ``${`` (or ``$(``,
    backtick, ``(``) inside one is literal text, not a construct. But a single
    quote INSIDE a double-quoted span is itself a literal apostrophe — it opens
    no span — so a naive count of single quotes flips state on an apostrophe in
    ``"it's $(rm -rf /)"`` and wrongly reads the executing ``$(…)`` after it as
    single-quoted. So BOTH quote contexts are
    tracked: a ``'`` toggles single-quote state only when NOT already inside
    double quotes, and a ``"`` toggles double-quote state only when NOT inside
    single quotes. A backslash escape outside single quotes skips the next
    character (in bash a ``\\'`` outside single quotes is a literal apostrophe,
    not a span opener). The result is single-quote state at *index*.
    """
    in_single = False
    in_double = False
    i = 0
    while i < index and i < len(source):
        ch = source[i]
        if ch == "\\" and not in_single:
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        i += 1
    return in_single


def _rm_matching_close(source: str, start: int, opener: str, closer: str) -> int:
    """Index of the *closer* that balances the *opener* already consumed, QUOTE-AWARE.

     Scans from *start* tracking nesting of ``opener``/``closer`` and both quote
     contexts, so an ``opener``/``closer`` INSIDE a single- or double-quoted span
     (or backslash-escaped) does not change the depth — ``$(echo "a)b"; rm -rf /)``
     keeps its real close, where a quote-blind paren count would stop at the ``)``
     inside ``"a)b"`` and truncate the body before the wipe (GPT security-class,
    ). Returns the index of the balancing ``closer``, or ``len(source)``
     when the construct is unterminated (the caller then takes the remainder,
     which only ever feeds the classifier MORE text — the fail-closed direction).
    """
    depth = 1
    j = start
    n = len(source)
    in_single = in_double = False
    while j < n:
        ch = source[j]
        if ch == "\\" and not in_single:
            j += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return j
        j += 1
    return n


def _rm_matching_backtick(source: str, start: int) -> int:
    """Index of the backtick closing the one already consumed, QUOTE-AWARE.

    A backtick inside a SINGLE-quoted span is literal and does not close the
    substitution; inside double quotes a backtick DOES still delimit a command
    substitution, so only single-quote state suppresses it. A backslash escapes
    the next character outside single quotes. Returns the closing backtick's
    index, or ``len(source)`` when unterminated (caller takes the remainder).
    """
    j = start
    n = len(source)
    in_single = in_double = False
    while j < n:
        ch = source[j]
        if ch == "\\" and not in_single:
            j += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "`" and not in_single:
            return j
        j += 1
    return n


def _rm_exec_substitution_bodies(source: str) -> "list[str]":
    """Inner command lines of the execution-substitutions the shared walk misses.

    Three shapes, all of which RUN the command they carry (so an ``rm`` inside is
    executed, not data), and none of which the payload walk surfaces as a frame
    of its own:

    * a ``$(…)`` command substitution or a backtick one nested INSIDE a
      double-quoted word — ``grep -rn "$(rm -rf /)" test/``. The enclosing
      double quote makes the closing ``)`` quote-INACTIVE to the quote-aware
      body scan, so ``_substitution_bodies`` reads past it; but ``$(…)`` executes
      inside double quotes and unquoted, so the body is extracted here.
    * a bash 5.3 ``${ …;}`` funsub — ``grep x ${ rm -rf /;}`` — which the walk
      does not recognise as a substitution at all.
    * a bare ``(…)`` SUBSHELL — ``(rm -rf /)`` — which runs its body in a child
      shell. When it is glued (``(rm``) the tokenizer keeps the ``(`` on the
      program word, so ``rm`` never reaches program position; extracting the
      parenthesised body and classifying it as its own argv recovers it. (The
      spaced form ``( rm -rf / )`` already tokenizes cleanly, so this only ADDS
      the glued spelling.)

    SINGLE-QUOTE AWARE, and that is load-bearing: inside single quotes ``$(``,
    a backtick, ``(`` and ``${`` are all LITERAL — bash executes none of them —
    so ``git commit -m 'see `rm -rf /` warning'`` and ``grep '`rm -rf /`' src/``
    run no ``rm`` and must NOT be extracted (that is exactly the text false
    positive the Security Scope lane rejects). An opener inside DOUBLE quotes, or
    unquoted, does execute and is extracted. Double-quote state is not tracked
    because it does not suppress these forms.

    Returned bodies are command lines; the caller classifies each as its own
    argv. Over-extraction (a body that is not really an ``rm``) yields nothing,
    and an unbalanced/unterminated construct yields the remainder, which only ever
    feeds the classifier MORE text — the fail-closed direction.
    """
    bodies: list[str] = []
    n = len(source)
    # ``$(…)`` command substitutions, bare ``(…)`` subshells, and backtick
    # substitutions — skipped when inside a SINGLE-quoted span, where they are
    # literal. A ``(`` preceded by ``$`` is the command-sub opener; any other
    # ``(`` opens a subshell (both matched by the same paren walk).
    #
    # Both quote contexts are tracked: a ``'`` toggles single-quote state only
    # when NOT inside double quotes (an apostrophe in ``"it's $(rm -rf /)"`` is a
    # literal, and must not suppress the executing ``$(…)`` that follows — GPT
    # security-class), and a ``"`` toggles double-quote state only when
    # NOT inside single quotes. A backslash outside single quotes escapes the
    # next character.
    i = 0
    in_single = False
    in_double = False
    while i < n:
        ch = source[i]
        if ch == "\\" and not in_single:
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if in_single:
            i += 1
            continue
        if ch == "(":
            open_at = i + 1  # body starts just past the '('
            j = _rm_matching_close(source, open_at, "(", ")")
            bodies.append(source[open_at:j] if j < n else source[open_at:])
            i = j + 1
            continue
        if ch == "`":
            j = _rm_matching_backtick(source, i + 1)
            bodies.append(source[i + 1 : j] if j < n else source[i + 1 :])
            i = j + 1 if j < n else n
            continue
        i += 1
    # ``${ …;}`` / ``${|…;}`` funsubs, matched to their closing brace — also only
    # OUTSIDE a single-quoted span (a ``${`` in single quotes is literal).
    for match in _FUNSUB_OPEN_RE.finditer(source):
        if _index_in_single_quote(source, match.start()):
            continue
        depth = 1
        j = match.end()
        while j < n and depth:
            if source[j] == "{":
                depth += 1
            elif source[j] == "}":
                depth -= 1
            j += 1
        body = source[match.end() : j - 1] if depth == 0 else source[match.end() :]
        # Strip the value-returning ``|`` lead and a trailing statement ``;``.
        bodies.append(body.lstrip("|").rstrip().rstrip(";"))
    return bodies


#: Multi-call binaries that DISPATCH to the applet named by their first
#: (non-flag) argument: ``busybox rm -rf /`` runs the ``rm`` applet, and
#: ``toybox``/``busybox.exe`` do the same. Here ``rm`` is the dispatcher's first
#: ARGUMENT, not the line's program word and not behind an exec wrapper, so the
#: plain program-position scan and the wrapper set both miss it (GPT
#: security-class). Unlike an exec wrapper, ONLY the first argument is
#: the applet — ``busybox echo rm -rf /`` runs ``echo``, not ``rm`` — so the
#: dispatch is matched positionally, not by wrapper membership.
_RM_APPLET_DISPATCHERS: frozenset[str] = frozenset({"busybox", "toybox"})


def _rm_deescape_unquoted_backslashes(text: str) -> str:
    """Remove backslash escapes as an UNQUOTED inner shell would, so an escaped
    program name reforms.

    A ``bash -c $"\\r\\m -rf /"`` payload reaches the inner shell as the script
    ``\\r\\m -rf /``; unquoted, bash drops each backslash before an ordinary
    character, so ``\\r\\m`` becomes the word ``rm``. The outer walk's
    ``_decode_printf_escapes`` instead maps ``\\r`` to whitespace and drops the
    ``r``, so the ``rm`` never reforms and the wipe was missed (Item 4).

    Backslashes INSIDE single quotes are literal and are left untouched; a
    backslash outside single quotes removes itself and keeps the next character
    (``\\n`` -> ``n``, matching the inner shell's own unquoted lexing rather than
    the C-escape meaning — the shell does not turn an unquoted ``\\n`` into a
    newline). A trailing backslash is dropped.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_single = False
    while i < n:
        ch = text[i]
        if ch == "'":
            in_single = not in_single
            out.append(ch)
            i += 1
            continue
        if ch == "\\" and not in_single and i + 1 < n:
            out.append(text[i + 1])
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


#: ``find``'s two flags that RUN their trailing command span as a real argv
#: (``-execdir`` differs from ``-exec`` only in the working directory), so an
#: ``rm`` inside that span is executed, not data. ``-ok``/``-okdir`` prompt first
#: but still execute, so they are included — the prompt is not a control an agent
#: session can rely on.
_FIND_EXEC_FLAGS: frozenset[str] = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
#: The two tokens that TERMINATE a ``find -exec`` command span: ``;`` (run once
#: per match) and ``+`` (batch).
_FIND_EXEC_TERMINATORS: frozenset[str] = frozenset({";", "+"})
#: ``find`` operators that START the expression proper, ending the path-root
#: list — a grouping/negation token or the ``,`` separator. A leading OPTION
#: (``-L``/``-maxdepth``/``-type``…) does NOT end it: GNU find is lenient about
#: options appearing before the paths, so those are consumed first (see
#: ``_find_search_roots``) rather than hiding the real search root behind them.
_FIND_EXPRESSION_START = frozenset({"(", ")", "!", ","})
#: ``find`` options that take a following OPERAND, so the operand is skipped with
#: the option when consuming leading options — otherwise ``find -maxdepth 3 $HOME``
#: would read ``3`` as a search root and miss ``$HOME``. Covers the global
#: options that take a value (``-D``/``-O``) and the
#: common leading positional-option predicates (``-maxdepth``/``-mindepth``/
#: ``-type``/``-name``/…). An option NOT listed here is assumed flag-only.
_FIND_OPTIONS_WITH_OPERAND = frozenset(
    {
        "-d",
        "-o",
        "-maxdepth",
        "-mindepth",
        "-type",
        "-xtype",
        "-name",
        "-iname",
        "-path",
        "-ipath",
        "-regex",
        "-iregex",
        "-perm",
        "-user",
        "-group",
        "-uid",
        "-gid",
        "-size",
        "-newer",
        "-mtime",
        "-atime",
        "-ctime",
        "-mmin",
        "-amin",
        "-cmin",
    }
)


def _find_search_roots(tokens: "list[str]", find_at: int) -> "list[str]":
    """The leading path ROOTS of the ``find`` command whose program word is at
    *find_at* — the operands ``{}`` expands to.

    ``find [option …] [root …] [expression]``: GNU find accepts options before
    the paths (``find -maxdepth 3 $HOME …``), so leading OPTIONS are consumed
    first — each ``-flag`` and, when it takes a value (``_FIND_OPTIONS_WITH_OPERAND``),
    its operand — before the path roots are collected. Without that, a leading
    ``-maxdepth 3`` hid ``$HOME`` behind it and ``{}`` resolved to nothing (GPT
    security-class). Collection then runs from the first non-option token
    to the next option / grouping / ``,`` separator. ``find $HOME -exec …`` has
    root ``$HOME``; ``find -maxdepth 3 / -exec …`` has root ``/``; ``find . -exec
    …`` has root ``.`` (relative, not catastrophic).
    """
    n = len(tokens)
    j = find_at + 1
    # Consume leading options (and operand-taking option values).
    while j < n:
        tok = tokens[j]
        if tok.startswith("-") and tok not in _FIND_EXEC_FLAGS:
            if tok in _FIND_OPTIONS_WITH_OPERAND and j + 1 < n:
                j += 2  # skip the option AND its operand
            else:
                j += 1  # flag-only option
            continue
        break
    # Collect the path roots up to the first expression / option token.
    roots: list[str] = []
    while j < n:
        tok = tokens[j]
        if not tok or tok.startswith("-") or tok in _FIND_EXPRESSION_START:
            break
        roots.append(tok)
        j += 1
    return roots


def _rm_targets_in_find_exec(
    tokens: "list[str]", programs: "list[str]", *, strip_quotes: bool, _budget: "list[int]"
) -> "frozenset[str]":
    """Catastrophic ``rm`` targets inside a ``find … -exec <cmd> … ;/+`` span.

    ``find`` does not read its ``-exec`` argument as data — it runs the span as
    its own argv once per match (``-exec rm -rf {} ;``), so a rooted delete there
    is a real wipe the plain program-position scan cannot see (``rm`` is not the
    line's program, ``find`` is, and ``find`` is not an exec wrapper). GPT
    security-class finding: ``find . -exec rm -rf --no-preserve-root /``
    passed the floor because the exec span was never parsed.

    The ``{}`` placeholder is the CRUX of the home-deletion guard: ``find $HOME
    -exec rm -rf {} ;`` names no rooted operand in the span, yet ``{}`` expands to
    every match under ``$HOME`` and the run wipes the home tree (GPT
    security-class). So ``{}`` is not discarded — it is classified against
    the find command's SEARCH ROOTS (``$HOME`` here), the operands it expands to.
    A ``{}`` under a root/home root therefore reads as that catastrophic target,
    while ``{}`` under a relative or ordinary root (``find . -exec …``, ``find
    /tmp/x …`` through a wrapper) does not.

    Only a command whose PROGRAM is ``find`` is inspected (``_argv_programs``
    scopes it). Each ``-exec``/``-execdir``/``-ok``/``-okdir`` span runs from just
    after the flag to its ``;``/``+`` terminator (or the argv end), with each
    ``{}`` replaced by the search roots, and is classified as its own argv by the
    same rule — so flag order/position and the operand tests all apply inside it.
    """
    found: set[str] = set()
    i = 0
    n = len(tokens)
    find_at = -1
    while i < n:
        token = tokens[i]
        # Track the program word of the find command this token belongs to, so a
        # ``{}`` in its exec span can be resolved to that command's search roots.
        if _program_basename(programs[i]) == "find" and (i == 0 or programs[i] != programs[i - 1]):
            find_at = i
        # The flag is find's only when find is the command this token belongs to.
        if token in _FIND_EXEC_FLAGS and _program_basename(programs[i]) == "find":
            roots = _find_search_roots(tokens, find_at) if find_at >= 0 else []
            literal_span: list[str] = []
            resolved_span: list[str] = []
            has_placeholder = False
            j = i + 1
            while j < n and tokens[j] not in _FIND_EXEC_TERMINATORS:
                if tokens[j] == "{}":
                    has_placeholder = True
                    resolved_span.extend(roots)  # {} -> the search roots it expands to
                else:
                    literal_span.append(tokens[j])
                    resolved_span.append(tokens[j])
                j += 1
            if literal_span:
                # The LITERAL span (``{}`` removed) is a command argv classified
                # DIRECT: a literal rooted operand (``find . -exec rm -rf /tmp/x``)
                # is a wipe base matched by substring, so it keeps the descendant
                # contract.
                if _budget[0] > 0:
                    _budget[0] -= 1
                    found |= _rm_targets_in_argv(
                        literal_span, strip_quotes=strip_quotes, _budget=_budget
                    )
            if has_placeholder and roots:
                # ``{}`` expands to each match under the search roots, so an
                # ``rm -rf {}`` there wipes the root tree — the home-deletion guard
                # base could not see (no literal path to substring-match; GPT
                # security-class). Classify the FULL span with ``{}``
                # replaced by the roots, against the root/home-ITSELF contract:
                # ``find $HOME -exec rm -rf {}`` denies (home itself), while
                # ``find /tmp/x -exec rm -rf {}`` allows (a descendant root,
                # benign, base never denied it).
                if _budget[0] > 0:
                    _budget[0] -= 1
                    found |= _rm_targets_in_argv(
                        resolved_span,
                        strip_quotes=strip_quotes,
                        force_wrapper=True,
                        _budget=_budget,
                    )
            i = j + 1  # step past the terminator
            continue
        i += 1
    return frozenset(found)


#: Shell programs whose ``-c`` argument is a command STRING they execute. When a
#: nested payload's escaped quoting defeats the walk's own descent, the walk can
#: still hand this frame a FLATTENED argv (``['sh', '-c', 'rm', '-rf', '/']``);
#: the tokens after ``-c`` are then the executed command, read here as their own
#: argv so the ``rm`` leads its own command instead of sitting behind ``sh``.
_RM_SHELL_C_PROGRAMS: frozenset[str] = frozenset(
    {"sh", "bash", "zsh", "dash", "ksh", "ash", "busybox"}
)

#: Language interpreters whose ``-c`` / ``-e`` / ``-E`` argument is a PROGRAM
#: string they execute. A shell command the program runs (``os.system("rm -rf
#: /")``, Perl/Ruby ``system("rm -rf /")``, Node ``execSync("rm -rf /")``) sits
#: in a quoted string literal inside that code, so it is invisible to a shell
#: tokenizer — the interpreter, not ``rm``, is the argv program (GPT
#: security-class). The floor extracts each quoted-string literal from
#: the code and classifies it as an ``rm`` argv.
_RM_INTERPRETER_PROGRAMS: frozenset[str] = frozenset(
    {"python", "python2", "python3", "perl", "ruby", "node", "nodejs", "php"}
)
#: The code-string flags those interpreters accept: ``-c`` (python/php),
#: ``-e``/``-E`` (perl/ruby/node). The token after one is the program string.
_RM_INTERPRETER_CODE_FLAGS: frozenset[str] = frozenset({"-c", "-e", "-E"})
#: A quoted string literal inside interpreter code: a single- or double-quoted
#: run with no embedded quote of the same kind. The shell command an
#: ``os.system`` / ``system`` / ``exec`` call runs is always such a literal.
_RM_CODE_STRING_LITERAL_RE = re.compile(r"""(?:"([^"]*)"|'([^']*)')""")
#: A herestring redirection (``python3 <<<'code'``): the code is fed to the
#: interpreter's stdin, glued to the ``<<<`` token or the token after it.
_RM_HERESTRING_RE = re.compile(r"^<<<(.*)$", re.DOTALL)
#: A heredoc redirection MARKER (``python3 <<'EOF'`` / ``<<-EOF``): the body
#: runs from the token after the marker to the closing DELIMITER word (here
#: ``EOF``), which is what the interpreter reads on stdin.
_RM_HEREDOC_RE = re.compile(r"^<<-?\s*(.*)$", re.DOTALL)


def _rm_targets_in_interpreter_code_payload(
    code: str, *, strip_quotes: bool, _budget: "list[int]"
) -> "frozenset[str]":
    """Catastrophic ``rm`` targets inside one interpreter code string.

    Shared by the ``-c``/``-e`` flag path and the stdin/heredoc/herestring path:
    every quoted string LITERAL in the code is a candidate shell command an
    ``os.system`` / ``system`` / ``exec`` call runs, so each is classified as its
    own ``rm`` argv; the space-joined sequence of all literals covers an argv
    SPLIT across list elements (``subprocess.run(['rm','-rf','/'])``); and the
    whole decoded code covers an unquoted ``rm`` reached without a literal.
    """
    found: set[str] = set()
    literals = [
        (m.group(1) if m.group(1) is not None else m.group(2))
        for m in _RM_CODE_STRING_LITERAL_RE.finditer(code)
    ]
    for literal in literals:
        if literal and "rm" in literal.lower():
            if _budget[0] <= 0:
                break
            _budget[0] -= 1
            found |= _rm_targets_in_argv(
                _split_shell_words(literal), strip_quotes=strip_quotes, _budget=_budget
            )
    if any("rm" in lit.lower() for lit in literals) and _budget[0] > 0:
        _budget[0] -= 1
        found |= _rm_targets_in_argv(
            _split_shell_words(" ".join(literals)), strip_quotes=strip_quotes, _budget=_budget
        )
    if _budget[0] > 0:
        _budget[0] -= 1
        found |= _rm_targets_in_argv(
            _split_shell_words(code), strip_quotes=strip_quotes, _budget=_budget
        )
    return frozenset(found)


def _rm_targets_in_interpreter_code(
    tokens: "list[str]", programs: "list[str]", *, strip_quotes: bool, _budget: "list[int]"
) -> "frozenset[str]":
    """Catastrophic ``rm`` targets a language-interpreter code payload runs.

    ``python -c 'import os; os.system("rm -rf /")'`` executes ``rm`` through the
    interpreter: ``python`` is the argv program, and the ``rm -rf /`` lives in a
    quoted string INSIDE the code, so the plain scan and the shell-payload walk
    never see it. When a frame's program is one of ``_RM_INTERPRETER_PROGRAMS``
    the code reaches it three ways, all classified via
    :func:`_rm_targets_in_interpreter_code_payload`:

    * a ``-c``/``-e``/``-E`` code FLAG — the token after it is the program string;
    * a HERESTRING (``python3 <<<'code'``) — stdin fed inline, the code glued to
      the ``<<<`` token or the token after it;
    * a HEREDOC (``python3 - <<'EOF'`` … ``EOF`` / ``python3 <<'EOF'`` …) — the
      body from the token after the ``<<`` marker to the closing delimiter word,
      which the interpreter reads on stdin exactly as ``-c`` code.

    Scoped to a frame whose PROGRAM is the interpreter, so a code flag or a
    heredoc feeding data to another command is untouched.

    Residual: code piped in as another command's OUTPUT
    (``echo 'os.system("rm -rf /")' | python3``) is NOT read here — the code is
    the UPSTREAM command's stdout, not a token of the interpreter's own frame, so
    the per-frame model cannot see it. That is the same class the pipe-into-shell
    case leaves to the raw tier, and the regex second net never covered it either.
    """
    found: set[str] = set()
    i = 0
    n = len(tokens)
    while i < n:
        if not (
            _program_basename(tokens[i]) in _RM_INTERPRETER_PROGRAMS
            and _program_basename(programs[i]) in _RM_INTERPRETER_PROGRAMS
        ):
            i += 1
            continue
        j = i + 1
        while j < n:
            token = tokens[j]
            if token in _RM_INTERPRETER_CODE_FLAGS and j + 1 < n:
                code = _rm_strip_surrounding_quotes(tokens[j + 1])
                found |= _rm_targets_in_interpreter_code_payload(
                    code, strip_quotes=strip_quotes, _budget=_budget
                )
                break
            herestring = _RM_HERESTRING_RE.match(token)
            if herestring is not None:
                # ``<<<'code'`` — the code may be glued to the ``<<<`` token, or
                # (when the shell split there) the next token is the code word.
                inline = _rm_strip_surrounding_quotes(herestring.group(1))
                if not inline and j + 1 < n:
                    inline = _rm_strip_surrounding_quotes(tokens[j + 1])
                found |= _rm_targets_in_interpreter_code_payload(
                    inline, strip_quotes=strip_quotes, _budget=_budget
                )
                break
            if _RM_HEREDOC_RE.match(token) is not None:
                # ``<<'EOF'`` / ``<<-EOF`` — the delimiter is what follows ``<<``
                # in the marker (``EOF``, quotes stripped); the heredoc BODY is the
                # run of tokens from here to that delimiter word, and that body is
                # the interpreter's stdin code.
                delim = _rm_strip_surrounding_quotes(
                    _RM_HEREDOC_RE.match(token).group(1)  # type: ignore[union-attr]
                ).strip()
                body: list[str] = []
                k = j + 1
                while k < n and _rm_strip_surrounding_quotes(tokens[k]).strip() != delim:
                    body.append(tokens[k])
                    k += 1
                found |= _rm_targets_in_interpreter_code_payload(
                    " ".join(body), strip_quotes=strip_quotes, _budget=_budget
                )
                break
            # A plain command boundary ends this interpreter's argv. Checked AFTER
            # the redirection markers above, because a herestring/heredoc token can
            # itself carry a shell separator inside its code (``<<<'a; b'``) and
            # would otherwise be read as a boundary before the code is inspected.
            if _ends_argv(token):
                break
            j += 1
        i += 1
    return frozenset(found)


def _rm_targets_in_shell_c(
    tokens: "list[str]", programs: "list[str]", *, strip_quotes: bool, _budget: "list[int]"
) -> "frozenset[str]":
    """Catastrophic ``rm`` targets in a ``sh -c <cmd>`` argv flattened into a frame.

    The payload walk normally descends ``bash -c '<script>'`` into a frame of its
    own, but a two-level nest with ESCAPED inner quotes
    (``bash -c 'sh -c "rm -rf \\"/\\""'``) can defeat the inner extraction and
    leave the ``sh -c`` frame's argv flattened to ``['sh', '-c', 'rm', '-rf',
    '/']``. There ``rm`` is not at program position (``sh`` is) and ``sh`` is not
    an exec wrapper, so the plain scan misses it. When a nested-shell program is
    followed by a ``-c`` flag, the tokens after ``-c`` are the command string it
    runs, so they are classified as their own argv — the same treatment
    ``find -exec`` gets. Scoped to a frame whose PROGRAM is the shell
    (``_argv_programs``), so a ``-c`` that is data to another command is untouched.
    """
    found: set[str] = set()
    i = 0
    n = len(tokens)
    while i < n:
        if (
            _program_basename(tokens[i]) in _RM_SHELL_C_PROGRAMS
            and _program_basename(programs[i]) in _RM_SHELL_C_PROGRAMS
        ):
            # Find this shell command's own ``-c`` (before its argv ends), then
            # read the rest of the argv as the command string it executes.
            j = i + 1
            while j < n and not _ends_argv(tokens[j]):
                if tokens[j] == "-c" and j + 1 < n:
                    span = []
                    k = j + 1
                    while k < n and not _ends_argv(tokens[k]):
                        span.append(tokens[k])
                        k += 1
                    if span:
                        # The ``-c`` argument is a command STRING the inner shell
                        # re-parses, so re-split it — its OWN backslash de-escaping
                        # runs there. ``bash -c $"\r\m -rf /"`` reaches the inner
                        # shell as the script ``\r\m -rf /``, whose ``\r\m`` the
                        # inner bash de-escapes to ``rm``; the outer walk's
                        # printf-escape pass had mangled ``\r`` to whitespace and
                        # dropped the ``r``. De-escaping the joined payload the way
                        # the unquoted inner shell does, then splitting, recovers
                        # the ``rm`` program word (Item 4); classify that ONE view.
                        #
                        # Only ONE descent per ``-c`` span: de-escaping a payload
                        # with no backslash escapes is the identity, so the
                        # de-escaped split already covers the plain span — a second
                        # ``_rm_targets_in_argv(span)`` classified the same tokens
                        # again and let one nesting level fan out TWICE, which is
                        # what made a chain of ``sh -c`` spans exponential and hung
                        # the synchronous gate (Opus security-class).
                        if _budget[0] > 0:
                            _budget[0] -= 1
                            payload = _rm_deescape_unquoted_backslashes(" ".join(span))
                            found |= _rm_targets_in_argv(
                                _split_shell_words(payload),
                                strip_quotes=strip_quotes,
                                _budget=_budget,
                            )
                    break
                j += 1
        i += 1
    return frozenset(found)


def _rm_targets_in_argv(
    tokens: "list[str]",
    *,
    strip_quotes: bool,
    force_wrapper: bool = False,
    _budget: "list[int] | None" = None,
) -> "frozenset[str]":
    """The catastrophic ``rm`` targets deleted within ONE frame's raw argv.

    Fires for each token whose basename is ``rm`` and that is EXECUTED — ``rm``
    at program position (its command's leading word), the first argument of a
    multi-call dispatcher (``busybox rm``), or ``rm`` whose parent command does
    NOT treat its arguments as data. That last test is a DENYLIST: an ``rm``
    behind ANY parent is executed UNLESS the parent is in
    ``_DATA_CONSUMER_PROGRAMS`` (``echo`` prints, ``cat`` reads, ``cp``/``mv``
    move paths), so an unknown exec wrapper — ``setsid``/``nohup``/``chrt``/… —
    is treated as executable rather than slipping a fixed allowlist. ``rm`` is
    itself a data-consumer program, but that never mis-fires here because a
    program-position ``rm`` is caught by the leading-word test first and never
    reaches the parent check. An ``rm`` that is an argument of a data consumer
    (``echo rm -rf /`` prints, it does not run ``rm``) is skipped.

    From each executed ``rm`` its OWN argv is read forward until the command
    ends, so a sibling command's flags never leak in. A resolved operand is
    classified as base ``main``'s ``rm -rf /.*`` / ``rm -rf ~.*``: deny when it
    is root (``/``) or home (``~``/``$HOME``) OR any path under either.

    ``strip_quotes`` peels surrounding SHELL quotes from each operand — True for
    the raw split (``"$HOME"`` -> ``$HOME``), False for the decoded view where a
    surrounding quote is a literal character the decode produced (``$'"/"'`` ->
    ``"/"``, a filename, not the root).
    """
    if not tokens:
        return frozenset()
    # One shared descent budget per top-level classification. The public entry
    # (and every non-recursive caller) passes None, so a fresh cell is created
    # here; the sub-helpers thread the SAME cell into their recursive
    # ``_rm_targets_in_argv`` calls, so nested spans draw down one common budget.
    if _budget is None:
        _budget = [_RM_DESCENT_BUDGET]
    programs = _argv_programs(tokens)
    found: set[str] = set()
    found |= _rm_targets_in_find_exec(tokens, programs, strip_quotes=strip_quotes, _budget=_budget)
    found |= _rm_targets_in_shell_c(tokens, programs, strip_quotes=strip_quotes, _budget=_budget)
    found |= _rm_targets_in_interpreter_code(
        tokens, programs, strip_quotes=strip_quotes, _budget=_budget
    )
    # Computed once per argv (not per ``rm`` token): the pipe-into-shell / trailing
    # operator guards ``_data_consumer_exempt`` consults, whose sweep is quadratic
    # per token. ``None`` until the first ``rm`` needs it.
    disqualified: "bool | None" = None
    expect_program = True
    #: Index of the most recent command's program word, so a dispatcher's FIRST
    #: argument (its applet) can be recognised: ``busybox rm -rf /`` runs ``rm``.
    program_word_at = -1
    for i, token in enumerate(tokens):
        is_program_word = (
            expect_program and bool(token) and not _shell_normalizer.ENV_ASSIGNMENT_RE.match(token)
        )
        starts_command = is_program_word
        if is_program_word:
            expect_program = False
            program_word_at = i
        # A glued ``&`` / ``&&`` ENDS the command (backgrounds or chains it), so
        # the next token starts a NEW command that really runs — ``echo hi& rm
        # -rf /`` is two commands, and the ``rm`` is executed, not echo's data.
        # ``_ends_argv`` catches ``|``/``;`` glued to a token but not ``&``, and a
        # standalone ``&`` token is already covered; this adds the glued-tail case
        # . A ``2>&1`` redirection ends in ``1``, not
        # ``&``, so it is not mistaken for a boundary.
        if _ends_argv(token) or token.endswith("&"):
            expect_program = True
        if _program_basename(token) != "rm":
            continue
        # Executed iff ``rm`` leads its own command, or ``rm`` is the FIRST
        # argument of a multi-call dispatcher (``busybox rm`` runs the rm applet),
        # or its parent command does NOT treat its arguments as data. The last
        # test is a DENYLIST, not an allowlist: an exec wrapper set could only
        # ever name the wrappers someone thought of, and ``setsid``/``nohup``/
        # ``chrt``/``ionice``/… or any future one
        # would slip through. So the default for an UNKNOWN parent is EXECUTABLE,
        # and only a parent in ``_DATA_CONSUMER_PROGRAMS`` (``echo`` prints, ``cat``
        # reads, ``cp``/``mv`` move paths) makes the ``rm`` a data mention.
        # ``_data_consumer_exempt`` also refuses the exemption when the argument
        # pipes into a shell or carries a glued new-program operator, so
        # ``echo rm -rf / | sh`` is still executed.
        dispatched_applet = (
            i == program_word_at + 1
            and program_word_at >= 0
            and _program_basename(tokens[program_word_at]) in _RM_APPLET_DISPATCHERS
        )
        # When the command's program is a multi-call dispatcher, its FIRST
        # argument is the applet that actually runs, so THAT — not ``busybox`` —
        # is the effective parent of a later ``rm``. ``busybox echo rm -rf /``
        # runs ``echo``, which prints ``rm -rf /``: a mention, not a wipe. Resolve
        # the effective parent to the applet before the data-consumer test so the
        # dispatcher itself (never a data consumer) does not make its applet's
        # arguments look executed.
        dispatcher_applet_is_consumer = (
            not dispatched_applet
            and program_word_at >= 0
            and program_word_at + 1 < len(tokens)
            and _program_basename(tokens[program_word_at]) in _RM_APPLET_DISPATCHERS
            and _program_basename(tokens[program_word_at + 1]) in _DATA_CONSUMER_PROGRAMS
        )
        if not (starts_command or dispatched_applet):
            if dispatcher_applet_is_consumer:
                continue
            if disqualified is None:
                disqualified = _shell_normalizer._data_consumer_command_disqualified(tokens)
            if _data_consumer_exempt(i, token, programs, tokens, command_disqualified=disqualified):
                continue
        # A DIRECT ``rm`` (program position or a dispatcher applet) denies the
        # root/home dir AND any path under it (base's ``rm -rf /.*`` / ``rm -rf
        # ~.*``). An ``rm`` reached through an EXEC WRAPPER (``setsid``/``sudo``/
        # ``docker exec`` …) denies only the root/home ITSELF: base caught a
        # wrapper-reached descendant only as a substring of its ``rm -rf /``
        # literal, so denying wrapper-reached descendants newly refuses benign
        # work (``docker exec kc-ci rm -fr /tmp/build-cache``, which base allowed
        # because its literal is ``rm -rf`` not ``rm -fr``). Match base there
        # (Security Scope ruling, pinned by
        # ``test_rm_rf_behind_an_unknown_exec_wrapper_denies_root_home_not_descendants``).
        via_wrapper = force_wrapper or not (starts_command or dispatched_applet)
        root_re = _RM_ROOT_ITSELF_RE if via_wrapper else _RM_ROOT_DESCENDANT_RE
        home_re = _RM_HOME_ITSELF_RE if via_wrapper else _RM_HOME_DESCENDANT_RE
        has_rec = has_force = has_npr = False
        end_of_options = False
        depth = 0
        operands: list[str] = []
        for arg in tokens[i + 1 :]:
            operand = _rm_strip_surrounding_quotes(arg) if strip_quotes else arg
            # A glued control operator (``/;reboot``, ``/&&id``) leaves the real
            # operand before it; classify only that head and, outside a
            # substitution, end this rm's argv at the boundary so a command glued
            # after it is not read as another operand.
            glued_boundary = False
            if depth + _substitution_depth_delta(arg) <= 0:
                operand, glued_boundary = _rm_operand_before_boundary(operand)
            if arg == "--" and not end_of_options:
                end_of_options = True
            elif not end_of_options and arg == "--no-preserve-root":
                has_npr = True
            elif not end_of_options and _rm_is_recursive_flag(arg):
                has_rec = True
                if _rm_is_force_flag(arg):
                    has_force = True
            elif not end_of_options and _rm_is_force_flag(arg):
                has_force = True
            elif operand:
                operands.append(operand)
            depth += _substitution_depth_delta(arg)
            if depth <= 0 and (glued_boundary or _ends_argv(arg)):
                break
            depth = max(depth, 0)
        root_target = any(root_re.fullmatch(op) for op in operands)
        home_target = any(home_re.fullmatch(op) for op in operands)
        if has_npr or (has_rec and has_force):
            if root_target:
                found.add("root")
            if home_target:
                found.add("home")
    return frozenset(found)
