"""RE2 O(n) Thompson-NFA engine, trigram query planner, and Zoekt SPI/DTOs."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, runtime_checkable
from uuid import UUID

from context_scout.db.schema import OverlayFileDeltaRecord
from context_scout.schemas.contracts import PrincipalContext

logger = logging.getLogger(__name__)

_RE_PARSER: Any = getattr(re, "_parser", None)
_RE_CONSTANTS: Any = getattr(re, "_constants", None)

MAX_RE2_PATTERN_LENGTH: Final[int] = 2048
MAX_RE2_REPEAT_COUNT: Final[int] = 1000
MAX_LAZY_DFA_CACHE_ENTRIES: Final[int] = 4096
UNCAPPED_ZOEK_DISPLAY_COUNT: Final[int] = 2_147_483_647
_CONNECT_RPC_SERVICE: Final[str] = "zoekt.webserver.v1.WebserverService"


class RE2SyntaxError(ValueError):
    """Raised when a regex violates RE2 syntax or linear-time safety bounds."""


class ZoektUnavailableError(RuntimeError):
    """Raised when a remote Zoekt service cannot satisfy a search request."""


@dataclass(frozen=True)
class TrigramQueryPlan:
    """Conjunctive normal form of required lowercase trigrams extracted from a query."""

    required_groups: tuple[tuple[str, ...], ...]

    @property
    def is_unindexed(self) -> bool:
        return len(self.required_groups) == 0


def _scan_unsupported_re2_tokens(pattern: str) -> None:
    """Rejects backreferences, lookarounds, conditionals, and excessive bounded repeats."""
    if not isinstance(pattern, str) or not pattern:
        raise RE2SyntaxError("RE2 pattern must be a non-empty string.")
    if "\x00" in pattern:
        raise RE2SyntaxError("RE2 pattern cannot contain NUL bytes.")
    if len(pattern) > MAX_RE2_PATTERN_LENGTH:
        raise RE2SyntaxError(
            f"RE2 pattern length {len(pattern)} exceeds maximum {MAX_RE2_PATTERN_LENGTH}."
        )

    i = 0
    n = len(pattern)
    in_char_class = False
    while i < n:
        ch = pattern[i]
        if ch == "\\":
            if i + 1 >= n:
                raise RE2SyntaxError("Trailing backslash in regex pattern.")
            nxt = pattern[i + 1]
            if not in_char_class:
                if nxt in "123456789":
                    raise RE2SyntaxError(
                        f"Backreference '\\{nxt}' is unsupported in O(n) RE2 syntax."
                    )
                if nxt == "k" and i + 2 < n and pattern[i + 2] == "<":
                    raise RE2SyntaxError(
                        "Named backreference '\\k<...>' is unsupported in O(n) RE2 syntax."
                    )
            i += 2
            continue

        if in_char_class:
            if ch == "]":
                in_char_class = False
            i += 1
            continue

        if ch == "[":
            in_char_class = True
            if i + 1 < n and pattern[i + 1] == "^":
                i += 2
            else:
                i += 1
            if i < n and pattern[i] == "]":
                i += 1
            continue

        if ch == "(" and i + 1 < n and pattern[i + 1] == "?" and i + 2 < n:
            third = pattern[i + 2]
            if third in {"=", "!"}:
                raise RE2SyntaxError(
                    "Lookahead assertions '(?=...)' / '(?!...)' are unsupported in RE2."
                )
            if third == "<" and i + 3 < n and pattern[i + 3] in {"=", "!"}:
                raise RE2SyntaxError(
                    "Lookbehind assertions '(?<=...)' / '(?<!...)' are unsupported in RE2."
                )
            if third == "(":
                raise RE2SyntaxError("Conditional subpatterns '(?(...)' are unsupported in RE2.")
            if third == "#":
                raise RE2SyntaxError("Inline regex comments '(?#...)' are unsupported in RE2.")
            if third == ">":
                raise RE2SyntaxError("Atomic groups '(?>...)' are unsupported in RE2.")
            if third == "P" and i + 3 < n and pattern[i + 3] == "=":
                raise RE2SyntaxError(
                    "Named backreference '(?P=...)' is unsupported in O(n) RE2 syntax."
                )

        if ch == "{":
            close_idx = pattern.find("}", i + 1)
            if close_idx != -1:
                body = pattern[i + 1 : close_idx]
                if re.fullmatch(r"\d+(?:,\d*)?", body):
                    for num_str in body.split(","):
                        if num_str and int(num_str) > MAX_RE2_REPEAT_COUNT:
                            raise RE2SyntaxError(
                                f"Repetition bound {num_str} exceeds RE2 limit "
                                f"{MAX_RE2_REPEAT_COUNT}."
                            )
                    if close_idx + 1 < n and pattern[close_idx + 1] == "+":
                        raise RE2SyntaxError(
                            "Possessive quantifier '{...}+' is unsupported in RE2."
                        )

        if ch in {"*", "+", "?"} and i + 1 < n and pattern[i + 1] == "+":
            # RE2 does not support possessive quantifiers (*+, ++, ?+).
            raise RE2SyntaxError(f"Possessive quantifier '{ch}+' is unsupported in RE2.")

        i += 1


def _ast_has_unsupported_re2_nodes(subpattern: Sequence[tuple[Any, Any]]) -> None:
    """Validates parsed regex AST nodes against RE2 feature constraints."""
    for op, av in subpattern:
        op_name = str(op)
        if op_name in {
            "GROUPREF",
            "GROUPREF_IGNORE",
            "GROUPREF_EXISTS",
            "ASSERT",
            "ASSERT_NOT",
            "ATOMIC_GROUP",
            "POSSESSIVE_REPEAT",
            "POSSESSIVE_REPEAT_ONE",
        }:
            raise RE2SyntaxError(f"Unsupported RE2 regex construct: {op_name}")
        if op_name == "SUBPATTERN":
            _ast_has_unsupported_re2_nodes(av[-1])
        elif op_name == "BRANCH":
            for branch in av[1]:
                _ast_has_unsupported_re2_nodes(branch)
        elif op_name in {"MAX_REPEAT", "MIN_REPEAT"}:
            min_rep, max_rep, inner = av
            if min_rep > MAX_RE2_REPEAT_COUNT or (
                max_rep != _RE_CONSTANTS.MAXREPEAT and max_rep > MAX_RE2_REPEAT_COUNT
            ):
                raise RE2SyntaxError(f"Repetition bound exceeds RE2 limit {MAX_RE2_REPEAT_COUNT}.")
            _ast_has_unsupported_re2_nodes(inner)


def validate_re2_pattern(pattern: str, *, max_pattern_length: int = MAX_RE2_PATTERN_LENGTH) -> str:
    """Validates that a regex conforms to linear-time RE2 rules without backtracking traps."""
    if len(pattern) > max_pattern_length:
        raise RE2SyntaxError(
            f"RE2 pattern length {len(pattern)} exceeds maximum {max_pattern_length}."
        )
    _scan_unsupported_re2_tokens(pattern)
    if _RE_PARSER is not None and _RE_CONSTANTS is not None:
        try:
            parsed = _RE_PARSER.parse(pattern)
        except re.error as exc:
            raise RE2SyntaxError(f"Invalid regex syntax: {exc}") from exc
        _ast_has_unsupported_re2_nodes(parsed)
    return pattern


def _analyze_ast_risk(subpattern: Sequence[tuple[Any, Any]]) -> tuple[int, bool, int]:
    """Returns (star_height, has_quantified_alternation, wildcard_repeat_count)."""
    max_sh = 0
    quant_alt = False
    wildcard_repeats = 0

    for op, av in subpattern:
        op_name = str(op)
        if op_name == "SUBPATTERN":
            sh, qa, wr = _ast_analyze_inner(av[-1])
            max_sh = max(max_sh, sh)
            quant_alt = quant_alt or qa
            wildcard_repeats += wr
        elif op_name == "BRANCH":
            for branch in av[1]:
                sh, qa, wr = _ast_analyze_inner(branch)
                max_sh = max(max_sh, sh)
                quant_alt = quant_alt or qa
                wildcard_repeats += wr
        elif op_name in {"MAX_REPEAT", "MIN_REPEAT"}:
            _, max_rep, inner = av
            inner_sh, inner_qa, inner_wr = _ast_analyze_inner(inner)
            is_unbounded = max_rep == _RE_CONSTANTS.MAXREPEAT or max_rep > 1
            if is_unbounded:
                max_sh = max(max_sh, 1 + inner_sh)
                if _contains_branch(inner):
                    quant_alt = True
                if _contains_wide_matcher(inner):
                    wildcard_repeats += 1
            else:
                max_sh = max(max_sh, inner_sh)
            quant_alt = quant_alt or inner_qa
            wildcard_repeats += inner_wr

    return max_sh, quant_alt, wildcard_repeats


def _ast_analyze_inner(subpattern: Sequence[tuple[Any, Any]]) -> tuple[int, bool, int]:
    return _analyze_ast_risk(subpattern)


def _contains_branch(subpattern: Sequence[tuple[Any, Any]]) -> bool:
    for op, av in subpattern:
        op_name = str(op)
        if op_name == "BRANCH":
            return True
        if op_name == "SUBPATTERN" and _contains_branch(av[-1]):
            return True
        if op_name in {"MAX_REPEAT", "MIN_REPEAT"} and _contains_branch(av[2]):
            return True
    return False


def _contains_wide_matcher(subpattern: Sequence[tuple[Any, Any]]) -> bool:
    for op, av in subpattern:
        op_name = str(op)
        if op_name in {"ANY", "IN", "NOT_LITERAL", "CATEGORY"}:
            return True
        if op_name == "SUBPATTERN" and _contains_wide_matcher(av[-1]):
            return True
        if op_name == "BRANCH" and any(_contains_wide_matcher(b) for b in av[1]):
            return True
        if op_name in {"MAX_REPEAT", "MIN_REPEAT"} and _contains_wide_matcher(av[2]):
            return True
    return False


@dataclass(frozen=True)
class RE2Match:
    """Lightweight match object compatible with standard regex match usage."""

    start_pos: int
    end_pos: int
    text: str

    def start(self) -> int:
        return self.start_pos

    def end(self) -> int:
        return self.end_pos

    def span(self) -> tuple[int, int]:
        return (self.start_pos, self.end_pos)

    def group(self, idx: int = 0) -> str:
        if idx != 0:
            raise IndexError(f"RE2Match only captures group(0), got {idx}")
        return self.text[self.start_pos : self.end_pos]


def _build_char_predicate(
    op_name: str,
    av: Any,
    *,
    case_insensitive: bool,
    dotall: bool,
) -> Callable[[int], bool]:
    """Builds a fast single-codepoint predicate from a parsed re._parser AST leaf."""
    if op_name == "ANY":
        if dotall:
            return lambda _c: True
        return lambda c: c != 10

    if op_name == "LITERAL":
        lit = int(av)
        if case_insensitive:
            folded = {ord(chr(lit).lower()), ord(chr(lit).upper()), lit}
            return lambda c: c in folded
        return lambda c: c == lit

    if op_name == "NOT_LITERAL":
        lit = int(av)
        if case_insensitive:
            folded = {ord(chr(lit).lower()), ord(chr(lit).upper()), lit}
            return lambda c: c not in folded
        return lambda c: c != lit

    if op_name == "CATEGORY":
        return _build_category_predicate(str(av))

    if op_name == "IN":
        items = list(av)
        negate = bool(items and str(items[0][0]) == "NEGATE")
        if negate:
            items = items[1:]
        single_chars: set[int] = set()
        ranges: list[tuple[int, int]] = []
        cat_preds: list[Callable[[int], bool]] = []
        for item_op, item_av in items:
            item_name = str(item_op)
            if item_name == "LITERAL":
                code = int(item_av)
                single_chars.add(code)
                if case_insensitive:
                    single_chars.add(ord(chr(code).lower()))
                    single_chars.add(ord(chr(code).upper()))
            elif item_name == "RANGE":
                lo, hi = int(item_av[0]), int(item_av[1])
                ranges.append((lo, hi))
                if case_insensitive:
                    lo_ch = chr(lo)
                    hi_ch = chr(hi)
                    if lo_ch.isalpha() and hi_ch.isalpha():
                        ranges.append((ord(lo_ch.lower()), ord(hi_ch.lower())))
                        ranges.append((ord(lo_ch.upper()), ord(hi_ch.upper())))
            elif item_name == "CATEGORY":
                cat_preds.append(_build_category_predicate(str(item_av)))

        def _in_pred(c: int) -> bool:
            matched = c in single_chars
            if not matched and ranges:
                matched = any(lo <= c <= hi for lo, hi in ranges)
            if not matched and cat_preds:
                matched = any(fn(c) for fn in cat_preds)
            return (not matched) if negate else matched

        return _in_pred

    return lambda _c: False


def _build_category_predicate(cat_name: str) -> Callable[[int], bool]:
    if cat_name.endswith("_NOT_DIGIT"):
        return lambda c: not (48 <= c <= 57 or chr(c).isdigit())
    if cat_name.endswith("_DIGIT"):
        return lambda c: 48 <= c <= 57 or chr(c).isdigit()
    if cat_name.endswith("_NOT_SPACE"):
        return lambda c: not chr(c).isspace()
    if cat_name.endswith("_SPACE"):
        return lambda c: chr(c).isspace()
    if cat_name.endswith("_NOT_WORD"):
        return lambda c: not (c == 95 or chr(c).isalnum())
    if cat_name.endswith("_WORD"):
        return lambda c: c == 95 or chr(c).isalnum()
    if cat_name.endswith("LINEBREAK"):
        return lambda c: c in {10, 11, 12, 13}
    return lambda _c: False


def _is_word_char(text: str, idx: int) -> bool:
    if idx < 0 or idx >= len(text):
        return False
    ch = text[idx]
    return ch == "_" or ch.isalnum()


def _check_anchor(kind: str, text: str, pos: int, *, multiline: bool) -> bool:
    n = len(text)
    if kind in {"AT_BEGINNING", "AT_BEGINNING_LINE"}:
        if pos == 0:
            return True
        return multiline and pos > 0 and text[pos - 1] == "\n"
    if kind == "AT_BEGINNING_STRING":
        return pos == 0
    if kind in {"AT_END", "AT_END_LINE"}:
        if pos == n:
            return True
        return multiline and pos < n and text[pos] == "\n"
    if kind == "AT_END_STRING":
        return pos == n
    if kind in {"AT_BOUNDARY", "AT_UNI_BOUNDARY", "AT_LOC_BOUNDARY"}:
        left_w = _is_word_char(text, pos - 1)
        right_w = _is_word_char(text, pos)
        return left_w != right_w
    if kind in {"AT_NON_BOUNDARY", "AT_UNI_NON_BOUNDARY", "AT_LOC_NON_BOUNDARY"}:
        left_w = _is_word_char(text, pos - 1)
        right_w = _is_word_char(text, pos)
        return left_w == right_w
    return False


class _ThompsonNFA:
    """Bit-parallel Thompson NFA with bounded Lazy-DFA transition cache for O(n) execution."""

    def __init__(
        self,
        parsed: Sequence[tuple[Any, Any]],
        *,
        case_insensitive: bool,
        multiline: bool,
        dotall: bool,
    ) -> None:
        self.case_insensitive = case_insensitive
        self.multiline = multiline
        self.dotall = dotall

        self._eps: list[int] = []
        self._char_pred: dict[int, Callable[[int], bool]] = {}
        self._char_next: dict[int, int] = {}
        self._anchors: dict[int, tuple[str, int]] = {}
        self._lock = threading.Lock()
        self._dfa_cache: dict[tuple[int, int], int] = {}

        start_state, accept_state = self._compile_seq(parsed)
        self.start_state = start_state
        self.accept_state = accept_state
        self.accept_bit = 1 << accept_state
        self.num_states = len(self._eps)
        self.has_anchors = bool(self._anchors)

        # Precompute transitive unconditional epsilon-closure bitmask per state.
        self._static_closure: list[int] = [0] * self.num_states
        for s in range(self.num_states):
            visited = 1 << s
            stack = [s]
            while stack:
                curr = stack.pop()
                nxt_mask = self._eps[curr] & ~visited
                while nxt_mask:
                    lsb = nxt_mask & -nxt_mask
                    nxt = lsb.bit_length() - 1
                    visited |= lsb
                    stack.append(nxt)
                    nxt_mask &= nxt_mask - 1
            self._static_closure[s] = visited

        self.start_mask = self._expand_static(1 << self.start_state)
        self._anchor_bits = 0
        for s in self._anchors:
            self._anchor_bits |= 1 << s

    def _new_state(self) -> int:
        sid = len(self._eps)
        self._eps.append(0)
        return sid

    def _add_eps(self, src: int, dst: int) -> None:
        self._eps[src] |= 1 << dst

    def _compile_seq(self, items: Sequence[tuple[Any, Any]]) -> tuple[int, int]:
        if not items:
            s = self._new_state()
            return s, s
        head_start, prev_end = self._compile_node(items[0])
        for op, av in items[1:]:
            node_start, node_end = self._compile_node((op, av))
            self._add_eps(prev_end, node_start)
            prev_end = node_end
        return head_start, prev_end

    def _compile_node(self, node: tuple[Any, Any]) -> tuple[int, int]:
        op, av = node
        op_name = str(op)

        if op_name == "SUBPATTERN":
            return self._compile_seq(av[-1])

        if op_name == "BRANCH":
            s = self._new_state()
            e = self._new_state()
            for branch in av[1]:
                bs, be = self._compile_seq(branch)
                self._add_eps(s, bs)
                self._add_eps(be, e)
            return s, e

        if op_name == "AT":
            s = self._new_state()
            e = self._new_state()
            self._anchors[s] = (str(av), e)
            return s, e

        if op_name in {"LITERAL", "NOT_LITERAL", "ANY", "IN", "CATEGORY"}:
            s = self._new_state()
            e = self._new_state()
            self._char_pred[s] = _build_char_predicate(
                op_name,
                av,
                case_insensitive=self.case_insensitive,
                dotall=self.dotall,
            )
            self._char_next[s] = e
            return s, e

        if op_name in {"MAX_REPEAT", "MIN_REPEAT"}:
            min_rep, max_rep, inner = av
            start = self._new_state()
            curr = start
            for _ in range(min_rep):
                is_s, is_e = self._compile_seq(inner)
                self._add_eps(curr, is_s)
                curr = is_e

            if max_rep == _RE_CONSTANTS.MAXREPEAT:
                # Unbounded tail: 0 or more repetitions of inner.
                loop_in = self._new_state()
                loop_out = self._new_state()
                is_s, is_e = self._compile_seq(inner)
                self._add_eps(curr, loop_in)
                self._add_eps(loop_in, is_s)
                self._add_eps(loop_in, loop_out)
                self._add_eps(is_e, loop_in)
                return start, loop_out

            rem = max_rep - min_rep
            if rem > 0:
                end = self._new_state()
                opt_curr = curr
                for _ in range(rem):
                    is_s, is_e = self._compile_seq(inner)
                    self._add_eps(opt_curr, is_s)
                    self._add_eps(opt_curr, end)
                    opt_curr = is_e
                self._add_eps(opt_curr, end)
                return start, end
            return start, curr

        raise RE2SyntaxError(f"Unsupported regex AST node '{op_name}' in Thompson NFA.")

    def _expand_static(self, mask: int) -> int:
        res = 0
        m = mask
        while m:
            lsb = m & -m
            sid = lsb.bit_length() - 1
            res |= self._static_closure[sid]
            m &= m - 1
        return res

    def _resolve_anchors(self, mask: int, text: str, pos: int) -> int:
        if not self.has_anchors or not (mask & self._anchor_bits):
            return mask
        curr = mask
        while True:
            pending = curr & self._anchor_bits
            added = 0
            while pending:
                lsb = pending & -pending
                sid = lsb.bit_length() - 1
                pending &= pending - 1
                kind, target = self._anchors[sid]
                if _check_anchor(kind, text, pos, multiline=self.multiline):
                    added |= self._static_closure[target]
            new_mask = curr | added
            if new_mask == curr:
                return curr
            curr = new_mask

    def _step_char(self, state_mask: int, char_code: int) -> int:
        if state_mask == 0:
            return 0
        if not self.has_anchors:
            cache_key = (state_mask, char_code)
            with self._lock:
                cached = self._dfa_cache.get(cache_key)
            if cached is not None:
                return cached

        nxt_raw = 0
        for sid, pred in self._char_pred.items():
            if (state_mask & (1 << sid)) and pred(char_code):
                nxt_raw |= 1 << self._char_next[sid]
        nxt_mask = self._expand_static(nxt_raw)

        if not self.has_anchors:
            with self._lock:
                if len(self._dfa_cache) >= MAX_LAZY_DFA_CACHE_ENTRIES:
                    self._dfa_cache.clear()
                self._dfa_cache[cache_key] = nxt_mask
        return nxt_mask

    def contains(self, text: str) -> bool:
        active = self._resolve_anchors(self.start_mask, text, 0)
        if active & self.accept_bit:
            return True
        n = len(text)
        for pos in range(n):
            code = ord(text[pos])
            active = self._step_char(active, code)
            active |= self.start_mask
            active = self._resolve_anchors(active, text, pos + 1)
            if active & self.accept_bit:
                return True
        return False

    def match_at(self, text: str, start_pos: int, *, fullmatch: bool = False) -> RE2Match | None:
        active = self._resolve_anchors(self.start_mask, text, start_pos)
        best_end: int | None = None
        if (active & self.accept_bit) and (not fullmatch or start_pos == len(text)):
            best_end = start_pos
        n = len(text)
        for pos in range(start_pos, n):
            active = self._step_char(active, ord(text[pos]))
            if active == 0:
                break
            active = self._resolve_anchors(active, text, pos + 1)
            if (active & self.accept_bit) and (not fullmatch or (pos + 1) == n):
                best_end = pos + 1
        if best_end is None:
            return None
        return RE2Match(start_pos=start_pos, end_pos=best_end, text=text)

    def _expand_births(
        self,
        births: dict[int, int],
        text: str,
        pos: int,
    ) -> dict[int, int]:
        """Propagates earliest birth offset through static epsilon and anchor transitions."""
        out: dict[int, int] = {}
        queue: list[tuple[int, int]] = []
        for sid, b_pos in births.items():
            expanded = self._static_closure[sid]
            while expanded:
                lsb = expanded & -expanded
                tgt = lsb.bit_length() - 1
                expanded &= expanded - 1
                prev = out.get(tgt)
                if prev is None or b_pos < prev:
                    out[tgt] = b_pos
                    if tgt in self._anchors:
                        queue.append((tgt, b_pos))

        while queue:
            a_sid, b_pos = queue.pop()
            kind, a_tgt = self._anchors[a_sid]
            if not _check_anchor(kind, text, pos, multiline=self.multiline):
                continue
            expanded = self._static_closure[a_tgt]
            while expanded:
                lsb = expanded & -expanded
                tgt = lsb.bit_length() - 1
                expanded &= expanded - 1
                prev = out.get(tgt)
                if prev is None or b_pos < prev:
                    out[tgt] = b_pos
                    if tgt in self._anchors:
                        queue.append((tgt, b_pos))
        return out

    def search_from(self, text: str, offset: int = 0) -> RE2Match | None:
        if offset == 0 and not self.contains(text):
            return None
        n = len(text)
        best_start: int | None = None
        best_end: int | None = None
        active: dict[int, int] = {self.start_state: offset}
        active = self._expand_births(active, text, offset)
        if self.accept_state in active:
            best_start = active[self.accept_state]
            best_end = offset

        for pos in range(offset, n):
            code = ord(text[pos])
            next_raw: dict[int, int] = {}
            for sid, b_pos in active.items():
                if best_start is not None and b_pos > best_start:
                    continue
                pred = self._char_pred.get(sid)
                if pred is not None and pred(code):
                    tgt = self._char_next[sid]
                    prev = next_raw.get(tgt)
                    if prev is None or b_pos < prev:
                        next_raw[tgt] = b_pos

            if best_start is None:
                prev_start = next_raw.get(self.start_state)
                if prev_start is None or (pos + 1) < prev_start:
                    next_raw[self.start_state] = pos + 1

            active = self._expand_births(next_raw, text, pos + 1)
            if self.accept_state in active:
                acc_birth = active[self.accept_state]
                if best_start is None or acc_birth < best_start:
                    best_start = acc_birth
                    best_end = pos + 1
                elif acc_birth == best_start:
                    best_end = pos + 1

            if best_start is not None and not any(b <= best_start for b in active.values()):
                break

        if best_start is None or best_end is None:
            return None
        return RE2Match(start_pos=best_start, end_pos=best_end, text=text)


class RE2Pattern:
    """Linear-time O(n) RE2 regex pattern wrapper."""

    def __init__(
        self,
        pattern: str,
        *,
        case_insensitive: bool = False,
        multiline: bool = True,
        dotall: bool = False,
        force_thompson: bool = False,
    ) -> None:
        validate_re2_pattern(pattern)
        self.pattern = pattern
        self.case_insensitive = case_insensitive
        self.multiline = multiline
        self.dotall = dotall

        flags = 0
        if case_insensitive:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL

        self._fast_re: re.Pattern[str] | None = None
        self._nfa: _ThompsonNFA | None = None

        parsed = _RE_PARSER.parse(pattern, flags) if _RE_PARSER is not None else None
        if not force_thompson and parsed is not None:
            star_height, quant_alt, wildcard_repeats = _analyze_ast_risk(parsed)
            if star_height <= 1 and not quant_alt and wildcard_repeats <= 1:
                self._fast_re = re.compile(pattern, flags)

        if self._fast_re is None:
            if parsed is None:
                raise RE2SyntaxError("Python regex parser unavailable for Thompson NFA fallback.")
            self._nfa = _ThompsonNFA(
                parsed,
                case_insensitive=case_insensitive,
                multiline=multiline,
                dotall=dotall,
            )

    @property
    def uses_thompson_nfa(self) -> bool:
        return self._nfa is not None

    def contains(self, text: str) -> bool:
        if self._fast_re is not None:
            return self._fast_re.search(text) is not None
        assert self._nfa is not None
        return self._nfa.contains(text)

    def search(self, text: str, pos: int = 0) -> RE2Match | None:
        if self._fast_re is not None:
            m = self._fast_re.search(text, pos)
            if m is None:
                return None
            return RE2Match(start_pos=m.start(), end_pos=m.end(), text=text)
        assert self._nfa is not None
        return self._nfa.search_from(text, offset=pos)

    def match(self, text: str) -> RE2Match | None:
        if self._fast_re is not None:
            m = self._fast_re.match(text)
            if m is None:
                return None
            return RE2Match(start_pos=m.start(), end_pos=m.end(), text=text)
        assert self._nfa is not None
        return self._nfa.match_at(text, 0, fullmatch=False)

    def fullmatch(self, text: str) -> RE2Match | None:
        if self._fast_re is not None:
            m = self._fast_re.fullmatch(text)
            if m is None:
                return None
            return RE2Match(start_pos=m.start(), end_pos=m.end(), text=text)
        assert self._nfa is not None
        return self._nfa.match_at(text, 0, fullmatch=True)

    def finditer(self, text: str) -> Iterator[RE2Match]:
        pos = 0
        n = len(text)
        while pos <= n:
            m = self.search(text, pos)
            if m is None:
                break
            yield m
            pos = m.end() + 1 if m.end() == m.start() else m.end()

    def findall(self, text: str) -> list[str]:
        return [m.group(0) for m in self.finditer(text)]


def compile_re2(
    pattern: str,
    *,
    case_insensitive: bool = False,
    multiline: bool = True,
    dotall: bool = False,
    force_thompson: bool = False,
) -> RE2Pattern:
    """Validates and compiles an O(n) RE2 regex pattern."""
    return RE2Pattern(
        pattern,
        case_insensitive=case_insensitive,
        multiline=multiline,
        dotall=dotall,
        force_thompson=force_thompson,
    )


def _extract_trigrams_from_literal(text: str) -> list[str]:
    lowered = text.lower()
    if len(lowered) < 3:
        return []
    return list(dict.fromkeys(lowered[i : i + 3] for i in range(len(lowered) - 2)))


def _ast_trigram_groups(subpattern: Sequence[tuple[Any, Any]]) -> list[tuple[str, ...]]:
    """Extracts conjunctive/disjunctive trigram groups from a parsed regex AST."""
    groups: list[tuple[str, ...]] = []
    literal_run: list[str] = []

    def _flush_literal() -> None:
        if len(literal_run) >= 3:
            for trig in _extract_trigrams_from_literal("".join(literal_run)):
                groups.append((trig,))
        literal_run.clear()

    for op, av in subpattern:
        op_name = str(op)
        if op_name == "LITERAL":
            literal_run.append(chr(int(av)))
            continue

        _flush_literal()
        if op_name == "SUBPATTERN":
            groups.extend(_ast_trigram_groups(av[-1]))
        elif op_name in {"MAX_REPEAT", "MIN_REPEAT"}:
            min_rep, _, inner = av
            if min_rep >= 1:
                groups.extend(_ast_trigram_groups(inner))
        elif op_name == "BRANCH":
            branch_sets: list[list[tuple[str, ...]]] = [
                _ast_trigram_groups(branch) for branch in av[1]
            ]
            if branch_sets and all(len(bs) > 0 for bs in branch_sets):
                first_choices: list[str] = []
                for bs in branch_sets:
                    first_choices.extend(bs[0])
                deduped = tuple(dict.fromkeys(first_choices))
                if deduped:
                    groups.append(deduped)

    _flush_literal()
    return groups


def extract_required_trigrams(
    pattern: str,
    *,
    is_regex: bool = True,
) -> TrigramQueryPlan:
    """Extracts required lowercase 3-char trigrams from a literal or RE2 regex query."""
    if not pattern:
        return TrigramQueryPlan(required_groups=())
    if not is_regex:
        return TrigramQueryPlan(
            required_groups=tuple((trig,) for trig in _extract_trigrams_from_literal(pattern))
        )
    validate_re2_pattern(pattern)
    if _RE_PARSER is None:
        return TrigramQueryPlan(required_groups=())
    parsed = _RE_PARSER.parse(pattern)
    groups = _ast_trigram_groups(parsed)
    unique_groups: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for g in groups:
        if g not in seen:
            seen.add(g)
            unique_groups.append(g)
    return TrigramQueryPlan(required_groups=tuple(unique_groups))


@dataclass(frozen=True)
class ZoektLineMatch:
    """Single matched line inside a Zoekt file hit."""

    line_number: int
    line_text: str
    before_lines: tuple[str, ...] = ()
    after_lines: tuple[str, ...] = ()
    fragments: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ZoektFileMatch:
    """Matched repository file returned by Zoekt or LocalTrigramZoektIndex."""

    repo_name: str
    repo_id: int | None
    file_path: str
    branch: str
    language: str
    score: float
    ranking_weight: float = 1.0
    is_generated: bool = False
    is_vendored: bool = False
    source_layer: Literal["base_snapshot", "overlay"] = "base_snapshot"
    line_matches: tuple[ZoektLineMatch, ...] = ()
    checksum: str | None = None


@dataclass(frozen=True)
class ZoektSearchStats:
    """Telemetry and ACL summary emitted by a Zoekt search call."""

    duration_ms: float
    files_considered: int
    files_matched: int
    matches_returned: int
    truncated: bool
    short_circuited_acl: bool
    authorized_repo_count: int
    authorized_repo_bitmap_popcount: int
    trigram_candidates: int = 0


@dataclass(frozen=True)
class ZoektSearchResponse:
    """Structured result envelope returned by ZoektClientSPI implementations."""

    query: str
    file_matches: tuple[ZoektFileMatch, ...]
    stats: ZoektSearchStats


@runtime_checkable
class ZoektClientSPI(Protocol):
    """Pluggable Zoekt search interface supporting capped ranking and uncapped enumeration."""

    def search(
        self,
        query: str,
        principal: PrincipalContext,
        *,
        domain_id: str | None = None,
        file_pattern: str | None = None,
        repo_names: Sequence[str] | None = None,
        branch: str | None = None,
        limit: int = 50,
        uncapped: bool = False,
        case_sensitive: bool = False,
        context_lines: int = 1,
        overlay_deltas: Sequence[OverlayFileDeltaRecord] | None = None,
    ) -> ZoektSearchResponse: ...

    def enumerate_file_matches(
        self,
        query: str,
        principal: PrincipalContext,
        *,
        domain_id: str | None = None,
        file_pattern: str | None = None,
        repo_names: Sequence[str] | None = None,
        branch: str | None = None,
        case_sensitive: bool = False,
        context_lines: int = 1,
        overlay_deltas: Sequence[OverlayFileDeltaRecord] | None = None,
    ) -> Iterator[ZoektFileMatch]: ...


def format_zoekt_shard_repo_name(tenant_id: UUID | str, repo_name: str) -> str:
    """Namespaces remote Zoekt repository shards as ``{tenant_id}::{repo_name}``."""
    tid_str = str(tenant_id).strip()
    cleaned = repo_name.strip()
    prefix = f"{tid_str}::"
    if cleaned.startswith(prefix):
        return cleaned
    return f"{prefix}{cleaned}"


def strip_zoekt_shard_repo_name(tenant_id: UUID | str, raw_repo_name: str) -> tuple[str, bool]:
    """Strips ``{tenant_id}::`` when ``raw_repo_name`` belongs to ``tenant_id``.

    Returns ``(bare_repo_name, matched_tenant)`` and fails closed on foreign-tenant or
    unprefixed shared-storage shards.
    """
    tid_str = str(tenant_id).strip()
    cleaned = raw_repo_name.strip()
    prefix = f"{tid_str}::"
    if cleaned.startswith(prefix):
        return cleaned[len(prefix) :], True
    return "", False
