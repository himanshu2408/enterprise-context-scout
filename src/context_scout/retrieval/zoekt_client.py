"""Zoekt gRPC/JSON trigram search client with ACL bitmaps and O(n) RE2 regex.

Implements Pillar 1 (Zoekt Trigram Shards) for M2-01:
1. O(n) RE2 pattern validation and linear-time execution (bit-parallel Thompson NFA
   with bounded Lazy-DFA transition cache for nested quantifiers / alternations).
2. Multi-tenant Repo-ACL bitmap pre-filtering (`RepoACLBitmapRegistry`) that short-circuits
   unauthorized principals before touching shard posting lists or network sockets.
3. `ZoektGrpcJsonClient` supporting Connect gRPC-JSON unary (`/Search`), streaming
   (`/StreamSearch` Connect binary envelopes & NDJSON), and `/api/search` JSON RPC.
4. `LocalTrigramZoektIndex` in-process inverted trigram shard engine with `.gitattributes`
   down-weighting, Copy-on-Write MR overlay merging, and uncapped file-match enumeration.
"""

from __future__ import annotations

import base64
import heapq
import json
import logging
import re
import struct
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, runtime_checkable
from uuid import UUID

import httpx

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
class RepoACLResult:
    """Resolved repository authorization bitmap and IDs for a PrincipalContext."""

    tenant_id: UUID
    bitmap: int
    repo_ids: tuple[int, ...]
    repo_names: frozenset[str]
    popcount: int


@dataclass
class _RegisteredRepoEntry:
    repo_id: int
    tenant_id: UUID
    domain_id: str
    repo_name: str
    allowed_scopes: frozenset[str]


class RepoACLBitmapRegistry:
    """Thread-safe repository ACL bitmap registry for fail-closed Zoekt pre-filtering."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._next_repo_id = 1
        self._entries_by_key: dict[tuple[UUID, str, str], _RegisteredRepoEntry] = {}
        self._entries_by_id: dict[int, _RegisteredRepoEntry] = {}
        self._tenant_masks: dict[UUID, int] = {}
        self._domain_masks: dict[tuple[UUID, str], int] = {}
        self._scope_masks: dict[tuple[UUID, str], int] = {}

    def _rebuild_masks(self) -> None:
        self._tenant_masks.clear()
        self._domain_masks.clear()
        self._scope_masks.clear()
        for entry in self._entries_by_id.values():
            bit = 1 << entry.repo_id
            self._tenant_masks[entry.tenant_id] = self._tenant_masks.get(entry.tenant_id, 0) | bit
            dom_key = (entry.tenant_id, entry.domain_id)
            self._domain_masks[dom_key] = self._domain_masks.get(dom_key, 0) | bit
            for scope in entry.allowed_scopes:
                s_key = (entry.tenant_id, scope)
                self._scope_masks[s_key] = self._scope_masks.get(s_key, 0) | bit

    def register_repo(
        self,
        tenant_id: UUID,
        domain_id: str,
        repo_name: str,
        allowed_scopes: Sequence[str],
        *,
        repo_id: int | None = None,
    ) -> int:
        clean_scopes = frozenset(s.strip() for s in allowed_scopes if s and s.strip())
        key = (tenant_id, domain_id, repo_name)
        with self._lock:
            existing = self._entries_by_key.get(key)
            if existing is not None:
                if repo_id is not None and repo_id != existing.repo_id:
                    raise ValueError(
                        f"Repository {key!r} already registered with repo_id={existing.repo_id}"
                    )
                existing.allowed_scopes = clean_scopes
                self._rebuild_masks()
                return existing.repo_id

            if repo_id is not None:
                if repo_id <= 0:
                    raise ValueError(f"Zoekt repo_id must be positive, got {repo_id}")
                if repo_id in self._entries_by_id:
                    raise ValueError(f"Duplicate repo_id={repo_id} in RepoACLBitmapRegistry")
                assigned_id = repo_id
                if assigned_id >= self._next_repo_id:
                    self._next_repo_id = assigned_id + 1
            else:
                assigned_id = self._next_repo_id
                self._next_repo_id += 1

            entry = _RegisteredRepoEntry(
                repo_id=assigned_id,
                tenant_id=tenant_id,
                domain_id=domain_id,
                repo_name=repo_name,
                allowed_scopes=clean_scopes,
            )
            self._entries_by_key[key] = entry
            self._entries_by_id[assigned_id] = entry
            self._rebuild_masks()
            return assigned_id

    def update_repo_scopes(
        self,
        tenant_id: UUID,
        domain_id: str,
        repo_name: str,
        allowed_scopes: Sequence[str],
    ) -> int:
        return self.register_repo(tenant_id, domain_id, repo_name, allowed_scopes)

    def remove_repo(self, tenant_id: UUID, domain_id: str, repo_name: str) -> None:
        key = (tenant_id, domain_id, repo_name)
        with self._lock:
            entry = self._entries_by_key.pop(key, None)
            if entry is not None:
                self._entries_by_id.pop(entry.repo_id, None)
                self._rebuild_masks()

    def unregister_repo(self, tenant_id: UUID, domain_id: str, repo_name: str) -> bool:
        """Removes repository from bitmap ACL index and returns True if previously present."""
        key = (tenant_id, domain_id, repo_name)
        with self._lock:
            entry = self._entries_by_key.pop(key, None)
            if entry is not None:
                self._entries_by_id.pop(entry.repo_id, None)
                self._rebuild_masks()
                return True
            return False

    def get_repo_id(self, tenant_id: UUID, domain_id: str, repo_name: str) -> int | None:
        with self._lock:
            entry = self._entries_by_key.get((tenant_id, domain_id, repo_name))
            return entry.repo_id if entry is not None else None

    def compute_authorized_bitmap(
        self,
        principal: PrincipalContext,
        *,
        domain_id: str | None = None,
        repo_names: Sequence[str] | None = None,
    ) -> RepoACLResult:
        clean_principal_scopes = [s.strip() for s in principal.allowed_scopes if s and s.strip()]
        if not clean_principal_scopes:
            return RepoACLResult(
                tenant_id=principal.tenant_id,
                bitmap=0,
                repo_ids=(),
                repo_names=frozenset(),
                popcount=0,
            )

        with self._lock:
            tenant_mask = self._tenant_masks.get(principal.tenant_id, 0)
            if tenant_mask == 0:
                return RepoACLResult(
                    tenant_id=principal.tenant_id,
                    bitmap=0,
                    repo_ids=(),
                    repo_names=frozenset(),
                    popcount=0,
                )

            scope_mask = 0
            for scope in clean_principal_scopes:
                scope_mask |= self._scope_masks.get((principal.tenant_id, scope), 0)

            bitmap = tenant_mask & scope_mask
            if domain_id is not None:
                bitmap &= self._domain_masks.get((principal.tenant_id, domain_id), 0)

            if repo_names is not None:
                requested = {r.strip() for r in repo_names if r and r.strip()}
                filter_mask = 0
                for entry in self._entries_by_id.values():
                    if entry.tenant_id == principal.tenant_id and entry.repo_name in requested:
                        filter_mask |= 1 << entry.repo_id
                bitmap &= filter_mask

            if bitmap == 0:
                return RepoACLResult(
                    tenant_id=principal.tenant_id,
                    bitmap=0,
                    repo_ids=(),
                    repo_names=frozenset(),
                    popcount=0,
                )

            repo_ids: list[int] = []
            names: set[str] = set()
            m = bitmap
            while m:
                lsb = m & -m
                rid = lsb.bit_length() - 1
                m &= m - 1
                entry_opt = self._entries_by_id.get(rid)
                if entry_opt is not None:
                    repo_ids.append(rid)
                    names.add(entry_opt.repo_name)

            return RepoACLResult(
                tenant_id=principal.tenant_id,
                bitmap=bitmap,
                repo_ids=tuple(sorted(repo_ids)),
                repo_names=frozenset(names),
                popcount=len(repo_ids),
            )


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


def _decode_zoekt_text(raw: Any) -> str:
    """Decodes Zoekt base64-encoded bytes fields or UTF-8 strings."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        return raw
    return str(raw)


def _decode_zoekt_b64_or_text(raw: Any) -> str:
    """Decodes base64-encoded Zoekt payload fields (`FileName`, `Line`, `Before`, `After`)."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    text = str(raw)
    if not text:
        return ""
    try:
        decoded = base64.b64decode(text, validate=True)
        return decoded.decode("utf-8", errors="strict")
    except (ValueError, UnicodeDecodeError):
        return text


def _scan_content_lines(
    content: str,
    compiled_re: RE2Pattern,
    *,
    context_lines: int,
) -> list[ZoektLineMatch]:
    """Scans file content line-by-line with O(n) RE2 pattern and collects line matches."""
    if not compiled_re.contains(content):
        return []
    lines = content.splitlines()
    matches: list[ZoektLineMatch] = []
    ctx = max(0, min(context_lines, 5))
    for idx, line_text in enumerate(lines):
        frags: list[tuple[int, int]] = []
        for m in compiled_re.finditer(line_text):
            frags.append((m.start(), m.end()))
        if not frags:
            continue
        before = tuple(lines[max(0, idx - ctx) : idx]) if ctx > 0 else ()
        after = tuple(lines[idx + 1 : min(len(lines), idx + 1 + ctx)]) if ctx > 0 else ()
        matches.append(
            ZoektLineMatch(
                line_number=idx + 1,
                line_text=line_text,
                before_lines=before,
                after_lines=after,
                fragments=tuple(frags),
            )
        )
    return matches


def _build_overlay_matches(
    *,
    overlay_deltas: Sequence[OverlayFileDeltaRecord],
    compiled_query: RE2Pattern,
    compiled_file_re: RE2Pattern | None,
    authorized_repos: frozenset[str],
    acl_registry: RepoACLBitmapRegistry,
    principal: PrincipalContext,
    branch_label: str,
    context_lines: int,
) -> tuple[list[ZoektFileMatch], set[tuple[str, str]]]:
    """Scans CoW overlay deltas and returns (overlay_matches, suppressed_base_keys)."""
    suppressed: set[tuple[str, str]] = set()
    matches: list[ZoektFileMatch] = []
    principal_scopes = set(principal.allowed_scopes)

    for delta in overlay_deltas:
        if delta.tenant_id != principal.tenant_id:
            continue
        if delta.repo_name not in authorized_repos:
            continue
        if not set(delta.allowed_scopes).intersection(principal_scopes):
            continue

        suppressed.add((delta.repo_name, delta.file_path))
        if delta.previous_file_path:
            suppressed.add((delta.repo_name, delta.previous_file_path))

        if delta.change_type == "deleted" or delta.raw_content is None:
            continue
        if compiled_file_re is not None and not compiled_file_re.contains(delta.file_path):
            continue

        line_matches = _scan_content_lines(
            delta.raw_content, compiled_query, context_lines=context_lines
        )
        if not line_matches:
            continue

        base_score = 1.0 + 0.1 * min(len(line_matches), 20)
        weighted_score = round(base_score * delta.ranking_weight, 6)
        rid = acl_registry.get_repo_id(delta.tenant_id, delta.domain_id, delta.repo_name)
        matches.append(
            ZoektFileMatch(
                repo_name=delta.repo_name,
                repo_id=rid,
                file_path=delta.file_path,
                branch=branch_label,
                language=delta.language,
                score=weighted_score,
                ranking_weight=delta.ranking_weight,
                is_generated=delta.is_generated,
                is_vendored=delta.is_vendored,
                source_layer="overlay",
                line_matches=tuple(line_matches),
                checksum=delta.overlay_blob_sha,
            )
        )
    return matches, suppressed


@dataclass
class _IndexedShardDoc:
    doc_id: int
    tenant_id: UUID
    domain_id: str
    repo_id: int
    repo_name: str
    file_path: str
    branch: str
    language: str
    content: str
    allowed_scopes: frozenset[str]
    ranking_weight: float
    is_generated: bool
    is_vendored: bool
    checksum: str | None
    trigrams: frozenset[str]


class LocalTrigramZoektIndex:
    """In-process Zoekt-compatible inverted trigram shard engine with CoW overlay support."""

    def __init__(self, acl_registry: RepoACLBitmapRegistry | None = None) -> None:
        self.acl_registry = acl_registry or RepoACLBitmapRegistry()
        self._lock = threading.RLock()
        self._next_doc_id = 0
        self._docs: dict[int, _IndexedShardDoc] = {}
        self._doc_key_to_id: dict[tuple[UUID, str, str, str, str], int] = {}
        self._trigram_postings: dict[str, int] = {}
        self._repo_doc_masks: dict[int, int] = {}

    def register_repository(
        self,
        *,
        tenant_id: UUID,
        domain_id: str,
        repo_name: str,
        allowed_scopes: Sequence[str],
        repo_id: int | None = None,
    ) -> int:
        return self.acl_registry.register_repo(
            tenant_id=tenant_id,
            domain_id=domain_id,
            repo_name=repo_name,
            allowed_scopes=allowed_scopes,
            repo_id=repo_id,
        )

    def _remove_doc_locked(self, doc_id: int) -> None:
        doc = self._docs.pop(doc_id, None)
        if doc is None:
            return
        self._doc_key_to_id.pop(
            (doc.tenant_id, doc.domain_id, doc.repo_name, doc.branch, doc.file_path), None
        )
        clear_mask = ~(1 << doc_id)
        if doc.repo_id in self._repo_doc_masks:
            updated_repo_mask = self._repo_doc_masks[doc.repo_id] & clear_mask
            if updated_repo_mask:
                self._repo_doc_masks[doc.repo_id] = updated_repo_mask
            else:
                self._repo_doc_masks.pop(doc.repo_id, None)
        for trig in doc.trigrams:
            posting = self._trigram_postings.get(trig, 0) & clear_mask
            if posting:
                self._trigram_postings[trig] = posting
            else:
                self._trigram_postings.pop(trig, None)

    def index_file(
        self,
        *,
        tenant_id: UUID,
        domain_id: str,
        repo_name: str,
        file_path: str,
        content: str,
        allowed_scopes: Sequence[str],
        language: str = "text",
        branch: str = "HEAD",
        ranking_weight: float = 1.0,
        is_generated: bool = False,
        is_vendored: bool = False,
        checksum: str | None = None,
    ) -> int:
        clean_scopes = frozenset(s.strip() for s in allowed_scopes if s and s.strip())
        doc_key = (tenant_id, domain_id, repo_name, branch, file_path)
        lowered = content.lower()
        trigrams = frozenset(lowered[i : i + 3] for i in range(max(0, len(lowered) - 2)))

        with self._lock:
            existing_rid = self.acl_registry.get_repo_id(tenant_id, domain_id, repo_name)
            if existing_rid is None:
                repo_id = self.acl_registry.register_repo(
                    tenant_id, domain_id, repo_name, list(clean_scopes)
                )
            else:
                repo_id = existing_rid
                if clean_scopes:
                    existing_entry = self.acl_registry._entries_by_id[repo_id]
                    merged_scopes = existing_entry.allowed_scopes | clean_scopes
                    if merged_scopes != existing_entry.allowed_scopes:
                        self.acl_registry.update_repo_scopes(
                            tenant_id, domain_id, repo_name, sorted(merged_scopes)
                        )

            old_doc_id = self._doc_key_to_id.get(doc_key)
            if old_doc_id is not None:
                self._remove_doc_locked(old_doc_id)

            doc_id = self._next_doc_id
            self._next_doc_id += 1
            bit = 1 << doc_id
            doc = _IndexedShardDoc(
                doc_id=doc_id,
                tenant_id=tenant_id,
                domain_id=domain_id,
                repo_id=repo_id,
                repo_name=repo_name,
                file_path=file_path,
                branch=branch,
                language=language,
                content=content,
                allowed_scopes=clean_scopes,
                ranking_weight=ranking_weight,
                is_generated=is_generated,
                is_vendored=is_vendored,
                checksum=checksum,
                trigrams=trigrams,
            )
            self._docs[doc_id] = doc
            self._doc_key_to_id[doc_key] = doc_id
            self._repo_doc_masks[repo_id] = self._repo_doc_masks.get(repo_id, 0) | bit
            for trig in trigrams:
                self._trigram_postings[trig] = self._trigram_postings.get(trig, 0) | bit
            return doc_id

    def delete_file(
        self,
        *,
        tenant_id: UUID,
        domain_id: str,
        repo_name: str,
        file_path: str,
        branch: str = "HEAD",
    ) -> bool:
        doc_key = (tenant_id, domain_id, repo_name, branch, file_path)
        with self._lock:
            doc_id = self._doc_key_to_id.get(doc_key)
            if doc_id is None:
                return False
            self._remove_doc_locked(doc_id)
            return True

    def delete_repo_files(self, *, tenant_id: UUID, domain_id: str, repo_name: str) -> int:
        with self._lock:
            to_delete = [
                doc_id
                for doc_id, doc in self._docs.items()
                if doc.tenant_id == tenant_id
                and doc.domain_id == domain_id
                and doc.repo_name == repo_name
            ]
            for doc_id in to_delete:
                self._remove_doc_locked(doc_id)
            return len(to_delete)

    def delete_file_all_branches(
        self,
        *,
        tenant_id: UUID,
        domain_id: str,
        repo_name: str,
        file_path: str,
    ) -> int:
        """Deletes all branch shards matching (tenant_id, domain_id, repo_name, file_path)."""
        with self._lock:
            to_delete = [
                doc_id
                for doc_id, doc in self._docs.items()
                if doc.tenant_id == tenant_id
                and doc.domain_id == domain_id
                and doc.repo_name == repo_name
                and doc.file_path == file_path
            ]
            for doc_id in to_delete:
                self._remove_doc_locked(doc_id)
            return len(to_delete)

    def purge_repo(self, *, tenant_id: UUID, domain_id: str, repo_name: str) -> int:
        """Hard-deletes all repository shards and revokes bitmap ACL registration."""
        with self._lock:
            deleted = self.delete_repo_files(
                tenant_id=tenant_id,
                domain_id=domain_id,
                repo_name=repo_name,
            )
            self.acl_registry.unregister_repo(tenant_id, domain_id, repo_name)
            return deleted

    def purge_by_checksum(self, *, tenant_id: UUID, domain_id: str, checksum: str) -> int:
        """Hard-deletes all indexed shards matching a specific blob SHA/checksum."""
        clean_sum = checksum.strip()
        if not clean_sum:
            return 0
        with self._lock:
            to_delete = [
                doc_id
                for doc_id, doc in self._docs.items()
                if doc.tenant_id == tenant_id
                and doc.domain_id == domain_id
                and doc.checksum == clean_sum
            ]
            for doc_id in to_delete:
                self._remove_doc_locked(doc_id)
            return len(to_delete)

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
    ) -> ZoektSearchResponse:
        t0 = time.perf_counter()
        compiled_query = compile_re2(query, case_insensitive=not case_sensitive)
        compiled_file_re = (
            compile_re2(file_pattern, case_insensitive=False)
            if file_pattern and file_pattern.strip()
            else None
        )
        plan = extract_required_trigrams(query, is_regex=True)

        acl = self.acl_registry.compute_authorized_bitmap(
            principal,
            domain_id=domain_id,
            repo_names=repo_names,
        )
        if acl.popcount == 0:
            duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return ZoektSearchResponse(
                query=query,
                file_matches=(),
                stats=ZoektSearchStats(
                    duration_ms=duration_ms,
                    files_considered=0,
                    files_matched=0,
                    matches_returned=0,
                    truncated=False,
                    short_circuited_acl=True,
                    authorized_repo_count=0,
                    authorized_repo_bitmap_popcount=0,
                    trigram_candidates=0,
                ),
            )

        overlay_matches: list[ZoektFileMatch] = []
        suppressed_base: set[tuple[str, str]] = set()
        if overlay_deltas:
            overlay_matches, suppressed_base = _build_overlay_matches(
                overlay_deltas=overlay_deltas,
                compiled_query=compiled_query,
                compiled_file_re=compiled_file_re,
                authorized_repos=acl.repo_names,
                acl_registry=self.acl_registry,
                principal=principal,
                branch_label=branch or "overlay",
                context_lines=context_lines,
            )

        principal_scopes = frozenset(principal.allowed_scopes)
        with self._lock:
            auth_doc_mask = 0
            for rid in acl.repo_ids:
                auth_doc_mask |= self._repo_doc_masks.get(rid, 0)

            candidate_mask = auth_doc_mask
            for group in plan.required_groups:
                group_mask = 0
                for trig in group:
                    group_mask |= self._trigram_postings.get(trig, 0)
                candidate_mask &= group_mask
                if candidate_mask == 0:
                    break

            candidates: list[_IndexedShardDoc] = []
            m = candidate_mask
            while m:
                lsb = m & -m
                doc_id = lsb.bit_length() - 1
                m &= m - 1
                doc = self._docs.get(doc_id)
                if doc is not None:
                    candidates.append(doc)

        trigram_candidates = len(candidates)
        all_matches: list[ZoektFileMatch] = list(overlay_matches)
        considered = len(overlay_deltas) if overlay_deltas else 0

        for doc in candidates:
            if doc.tenant_id != principal.tenant_id:
                continue
            if not doc.allowed_scopes.intersection(principal_scopes):
                continue
            if branch is not None and doc.branch != branch:
                continue
            if (doc.repo_name, doc.file_path) in suppressed_base:
                continue
            if compiled_file_re is not None and not compiled_file_re.contains(doc.file_path):
                continue
            considered += 1
            line_matches = _scan_content_lines(
                doc.content, compiled_query, context_lines=context_lines
            )
            if not line_matches:
                continue

            base_score = 1.0 + 0.1 * min(len(line_matches), 20)
            weighted_score = round(base_score * doc.ranking_weight, 6)
            all_matches.append(
                ZoektFileMatch(
                    repo_name=doc.repo_name,
                    repo_id=doc.repo_id,
                    file_path=doc.file_path,
                    branch=doc.branch,
                    language=doc.language,
                    score=weighted_score,
                    ranking_weight=doc.ranking_weight,
                    is_generated=doc.is_generated,
                    is_vendored=doc.is_vendored,
                    source_layer="base_snapshot",
                    line_matches=tuple(line_matches),
                    checksum=doc.checksum,
                )
            )

        total_matched = len(all_matches)
        if uncapped or limit <= 0:
            selected = sorted(
                all_matches,
                key=lambda item: (-item.score, item.repo_name, item.file_path),
            )
            truncated = False
        else:
            selected = heapq.nsmallest(
                limit,
                all_matches,
                key=lambda item: (-item.score, item.repo_name, item.file_path),
            )
            truncated = total_matched > len(selected)

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return ZoektSearchResponse(
            query=query,
            file_matches=tuple(selected),
            stats=ZoektSearchStats(
                duration_ms=duration_ms,
                files_considered=considered,
                files_matched=total_matched,
                matches_returned=len(selected),
                truncated=truncated,
                short_circuited_acl=False,
                authorized_repo_count=len(acl.repo_ids),
                authorized_repo_bitmap_popcount=acl.popcount,
                trigram_candidates=trigram_candidates,
            ),
        )

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
    ) -> Iterator[ZoektFileMatch]:
        response = self.search(
            query,
            principal,
            domain_id=domain_id,
            file_pattern=file_pattern,
            repo_names=repo_names,
            branch=branch,
            limit=UNCAPPED_ZOEK_DISPLAY_COUNT,
            uncapped=True,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            overlay_deltas=overlay_deltas,
        )
        yield from response.file_matches


def _parse_connect_stream_envelopes(raw: bytes) -> list[dict[str, Any]]:
    """Parses Connect streaming binary frames, NDJSON lines, or single JSON payloads."""
    if not raw:
        return []

    # Connect streaming binary frames begin with flags byte 0x00 or 0x02 + 4-byte uint32 BE len.
    if len(raw) >= 5 and raw[0] in {0, 2}:
        frames: list[dict[str, Any]] = []
        offset = 0
        parsed_valid_frames = False
        while offset + 5 <= len(raw):
            flags = raw[offset]
            frame_len = struct.unpack(">I", raw[offset + 1 : offset + 5])[0]
            offset += 5
            if flags not in {0, 1, 2, 3} or offset + frame_len > len(raw):
                parsed_valid_frames = False
                break
            chunk = raw[offset : offset + frame_len]
            offset += frame_len
            parsed_valid_frames = True
            obj = json.loads(chunk.decode("utf-8"))
            if isinstance(obj, dict):
                if flags & 0x02:
                    if obj.get("error"):
                        raise ZoektUnavailableError(
                            f"Zoekt StreamSearch error frame: {obj['error']}"
                        )
                    continue
                frames.append(obj)
        if parsed_valid_frames and offset == len(raw):
            return frames

    text = raw.decode("utf-8").strip()
    if not text:
        return []
    try:
        single = json.loads(text)
        if isinstance(single, dict):
            return [single]
        if isinstance(single, list):
            return [item for item in single if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    ndjson_frames: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            ndjson_frames.append(obj)
    return ndjson_frames


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


class ZoektGrpcJsonClient:
    """Zoekt client speaking Connect gRPC-JSON (`/Search`, `/StreamSearch`) and `/api/search`."""

    def __init__(
        self,
        endpoint_url: str,
        *,
        acl_registry: RepoACLBitmapRegistry | None = None,
        protocol: Literal["grpc_json", "json_api"] = "grpc_json",
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
        http_client: httpx.Client | None = None,
        weight_resolver: Callable[[str, str, str], tuple[float, bool, bool]] | None = None,
        repository: Any | None = None,
        connection_manager: Any | None = None,
    ) -> None:
        cleaned = endpoint_url.strip().rstrip("/")
        if not cleaned:
            raise ValueError("Zoekt endpoint_url cannot be empty.")
        self.endpoint_url = cleaned
        self.acl_registry = acl_registry or RepoACLBitmapRegistry()
        self.protocol = protocol
        self.timeout_seconds = timeout_seconds
        self._weight_resolver = weight_resolver
        self._repository = repository
        self._connection_manager = connection_manager
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=timeout_seconds,
            transport=transport,
        )

    def _hydrate_acl_registry_from_db(
        self,
        principal: PrincipalContext,
        *,
        domain_id: str | None = None,
    ) -> int:
        """Hydrates RepoACLBitmapRegistry from PG17 `repositories` (or InMemory) under RLS."""
        hydrated = 0
        if self._repository is not None and hasattr(self._repository, "_repos"):
            repos_map = getattr(self._repository, "_repos", {})
            for rec in repos_map.values():
                if rec.tenant_id != principal.tenant_id:
                    continue
                if domain_id is not None and rec.domain_id != domain_id:
                    continue
                if not getattr(rec, "is_active", True):
                    continue
                if principal.allowed_scopes and not (
                    set(rec.allowed_scopes) & set(principal.allowed_scopes)
                ):
                    continue
                self.acl_registry.register_repo(
                    rec.tenant_id,
                    rec.domain_id,
                    rec.repo_name,
                    rec.allowed_scopes,
                )
                hydrated += 1
            return hydrated

        if self._connection_manager is None:
            return 0

        from context_scout.security.rbac import scoped_db_session

        with scoped_db_session(self._connection_manager, principal) as conn:
            if domain_id is not None:
                rows = conn.execute(
                    """
                    SELECT tenant_id, domain_id, repo_name, allowed_scopes
                    FROM repositories
                    WHERE is_active = TRUE
                      AND domain_id = %s;
                    """,
                    (domain_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT tenant_id, domain_id, repo_name, allowed_scopes
                    FROM repositories
                    WHERE is_active = TRUE;
                    """
                ).fetchall()
            for r in rows:
                t_id = r[0] if isinstance(r[0], UUID) else UUID(str(r[0]))
                d_id = str(r[1])
                r_name = str(r[2])
                scopes_raw = r[3]
                scopes = (
                    [str(s) for s in scopes_raw] if isinstance(scopes_raw, (list, tuple)) else []
                )
                self.acl_registry.register_repo(t_id, d_id, r_name, scopes)
                hydrated += 1
        return hydrated

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def purge_repo(self, *, tenant_id: UUID, domain_id: str, repo_name: str) -> int:
        """Revoke ACL bitmap entry immediately and request remote Zoekt shard purge."""
        bare_repo, matched = strip_zoekt_shard_repo_name(tenant_id, repo_name)
        logical_repo = bare_repo if matched else repo_name
        shard_repo = format_zoekt_shard_repo_name(tenant_id, logical_repo)
        removed = 1 if self.acl_registry.unregister_repo(tenant_id, domain_id, logical_repo) else 0
        try:
            resp = self._client.post(
                f"{self.endpoint_url}/api/admin/purge",
                json={
                    "tenant_id": str(tenant_id),
                    "domain_id": domain_id,
                    "repo_name": shard_repo,
                    "logical_repo_name": logical_repo,
                },
            )
            if 200 <= resp.status_code < 300:
                removed = max(removed, 1)
        except httpx.HTTPError as exc:
            logger.warning(
                "Remote Zoekt purge endpoint failed for repo '%s' (%s); "
                "ACL bitmap entry was revoked fail-closed.",
                shard_repo,
                type(exc).__name__,
            )
        return removed

    def _build_zoekt_query(
        self,
        *,
        tenant_id: UUID,
        query: str,
        authorized_repos: frozenset[str],
        file_pattern: str | None,
        branch: str | None,
        case_sensitive: bool,
    ) -> str:
        repo_alt = "|".join(
            re.escape(format_zoekt_shard_repo_name(tenant_id, r)).replace(r"\-", "-")
            for r in sorted(authorized_repos)
        )
        clauses = [f"r:^({repo_alt})$"]
        if branch and branch.strip():
            clauses.append(f"b:^{re.escape(branch.strip())}$")
        if file_pattern and file_pattern.strip():
            clauses.append(f"f:{file_pattern.strip()}")
        clauses.append("case:yes" if case_sensitive else "case:no")
        clauses.append(f"({query})")
        return " ".join(clauses)

    def _parse_zoekt_payloads(
        self,
        payloads: Sequence[dict[str, Any]],
        *,
        acl: RepoACLResult,
        domain_id: str,
        suppressed_base: set[tuple[str, str]],
        branch_fallback: str,
    ) -> tuple[list[ZoektFileMatch], int]:
        matches: list[ZoektFileMatch] = []
        considered = 0
        authorized_ids = set(acl.repo_ids)

        for payload in payloads:
            raw_res = payload.get("Result")
            result_obj = raw_res if isinstance(raw_res, dict) else payload
            if not isinstance(result_obj, dict):
                continue
            stats_obj = result_obj.get("Stats")
            if isinstance(stats_obj, dict):
                considered += int(stats_obj.get("FilesConsidered", 0) or 0)
            raw_files = result_obj.get("Files") or result_obj.get("FileMatches") or []
            if not isinstance(raw_files, list):
                continue

            for raw_file in raw_files:
                if not isinstance(raw_file, dict):
                    continue
                raw_repo = _decode_zoekt_text(
                    raw_file.get("Repository") or raw_file.get("repo_name")
                )
                repo_name, matched_tenant = strip_zoekt_shard_repo_name(acl.tenant_id, raw_repo)
                raw_rid = raw_file.get("RepositoryID") or raw_file.get("RepoID")
                repo_id = int(raw_rid) if raw_rid is not None else None
                # Fail-closed defense-in-depth post-filter against stale/spoofed server shards.
                if not matched_tenant or repo_name not in acl.repo_names:
                    logger.warning(
                        "Zoekt post-filter dropped unauthorized repo '%s' for tenant %s",
                        raw_repo,
                        acl.tenant_id,
                    )
                    continue
                if repo_id is not None and authorized_ids and repo_id not in authorized_ids:
                    logger.warning(
                        "Zoekt post-filter dropped unauthorized repo_id=%s for tenant %s",
                        repo_id,
                        acl.tenant_id,
                    )
                    continue

                file_path = _decode_zoekt_b64_or_text(
                    raw_file.get("FileName") or raw_file.get("file_path")
                )
                if (repo_name, file_path) in suppressed_base:
                    continue

                branches = raw_file.get("Branches") or []
                file_branch = (
                    str(branches[0]) if isinstance(branches, list) and branches else branch_fallback
                )
                language = str(raw_file.get("Language") or "text")
                checksum = (
                    str(raw_file["Checksum"]) if raw_file.get("Checksum") is not None else None
                )
                raw_score = float(raw_file.get("Score") or 1.0)

                weight = 1.0
                is_gen = False
                is_vend = False
                if self._weight_resolver is not None:
                    weight, is_gen, is_vend = self._weight_resolver(domain_id, repo_name, file_path)

                line_matches: list[ZoektLineMatch] = []
                raw_lm_list = raw_file.get("LineMatches") or raw_file.get("ChunkMatches") or []
                if isinstance(raw_lm_list, list):
                    for lm in raw_lm_list:
                        if not isinstance(lm, dict):
                            continue
                        content_start = lm.get("ContentStart")
                        fallback_ln = (
                            content_start.get("LineNumber")
                            if isinstance(content_start, dict)
                            else None
                        )
                        line_num = int(lm.get("LineNumber") or fallback_ln or 1)
                        line_txt = _decode_zoekt_b64_or_text(
                            lm.get("Line") or lm.get("Content")
                        ).rstrip("\r\n")
                        before_raw = lm.get("Before")
                        after_raw = lm.get("After")
                        before_lines = (
                            tuple(_decode_zoekt_b64_or_text(before_raw).splitlines())
                            if before_raw
                            else ()
                        )
                        after_lines = (
                            tuple(_decode_zoekt_b64_or_text(after_raw).splitlines())
                            if after_raw
                            else ()
                        )
                        frags: list[tuple[int, int]] = []
                        for frag in lm.get("LineFragments") or []:
                            if isinstance(frag, dict):
                                off = int(frag.get("LineOffset") or 0)
                                length = int(frag.get("MatchLength") or 0)
                                frags.append((off, off + length))
                        line_matches.append(
                            ZoektLineMatch(
                                line_number=line_num,
                                line_text=line_txt,
                                before_lines=before_lines,
                                after_lines=after_lines,
                                fragments=tuple(frags),
                            )
                        )

                matches.append(
                    ZoektFileMatch(
                        repo_name=repo_name,
                        repo_id=repo_id,
                        file_path=file_path,
                        branch=file_branch,
                        language=language,
                        score=round(raw_score * weight, 6),
                        ranking_weight=weight,
                        is_generated=is_gen,
                        is_vendored=is_vend,
                        source_layer="base_snapshot",
                        line_matches=tuple(line_matches),
                        checksum=checksum,
                    )
                )
        return matches, considered

    def _post_rpc(
        self,
        *,
        effective_q: str,
        repo_ids: tuple[int, ...],
        max_docs: int,
        max_matches: int,
        context_lines: int,
        stream: bool,
    ) -> list[dict[str, Any]]:
        opts = {
            "MaxDocDisplayCount": max_docs,
            "TotalMaxMatchCount": max_matches,
            "NumContextLines": max(0, min(context_lines, 5)),
        }
        body = {
            "Q": effective_q,
            "RepoIds": list(repo_ids),
            "Opts": opts,
        }
        try:
            if self.protocol == "grpc_json":
                method = "StreamSearch" if stream else "Search"
                url = f"{self.endpoint_url}/{_CONNECT_RPC_SERVICE}/{method}"
                headers = {
                    "Content-Type": "application/json",
                    "Connect-Protocol-Version": "1",
                    "Accept": (
                        "application/connect+json, application/x-ndjson, application/json"
                        if stream
                        else "application/json"
                    ),
                }
                resp = self._client.post(url, json=body, headers=headers)
                resp.raise_for_status()
                if stream:
                    return _parse_connect_stream_envelopes(resp.content)
                data = resp.json()
                return [data] if isinstance(data, dict) else []

            url = f"{self.endpoint_url}/api/search"
            resp = self._client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
            return [data] if isinstance(data, dict) else []
        except (httpx.HTTPError, ValueError, struct.error) as exc:
            raise ZoektUnavailableError(
                f"Zoekt RPC request failed against {self.endpoint_url}: {exc}"
            ) from exc

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
    ) -> ZoektSearchResponse:
        t0 = time.perf_counter()
        compiled_query = compile_re2(query, case_insensitive=not case_sensitive)
        compiled_file_re = (
            compile_re2(file_pattern, case_insensitive=False)
            if file_pattern and file_pattern.strip()
            else None
        )
        acl = self.acl_registry.compute_authorized_bitmap(
            principal, domain_id=domain_id, repo_names=repo_names
        )
        missing_requested = bool(
            repo_names and any(r not in acl.repo_names for r in repo_names if r)
        )
        if (acl.popcount == 0 or missing_requested) and self._hydrate_acl_registry_from_db(
            principal, domain_id=domain_id
        ) > 0:
            acl = self.acl_registry.compute_authorized_bitmap(
                principal, domain_id=domain_id, repo_names=repo_names
            )
        if acl.popcount == 0:
            duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return ZoektSearchResponse(
                query=query,
                file_matches=(),
                stats=ZoektSearchStats(
                    duration_ms=duration_ms,
                    files_considered=0,
                    files_matched=0,
                    matches_returned=0,
                    truncated=False,
                    short_circuited_acl=True,
                    authorized_repo_count=0,
                    authorized_repo_bitmap_popcount=0,
                ),
            )

        overlay_matches: list[ZoektFileMatch] = []
        suppressed_base: set[tuple[str, str]] = set()
        if overlay_deltas:
            overlay_matches, suppressed_base = _build_overlay_matches(
                overlay_deltas=overlay_deltas,
                compiled_query=compiled_query,
                compiled_file_re=compiled_file_re,
                authorized_repos=acl.repo_names,
                acl_registry=self.acl_registry,
                principal=principal,
                branch_label=branch or "overlay",
                context_lines=context_lines,
            )

        effective_q = self._build_zoekt_query(
            tenant_id=principal.tenant_id,
            query=query,
            authorized_repos=acl.repo_names,
            file_pattern=file_pattern,
            branch=branch,
            case_sensitive=case_sensitive,
        )
        max_docs = UNCAPPED_ZOEK_DISPLAY_COUNT if uncapped or limit <= 0 else limit
        max_matches = (
            UNCAPPED_ZOEK_DISPLAY_COUNT if uncapped or limit <= 0 else max(limit * 20, 200)
        )
        payloads = self._post_rpc(
            effective_q=effective_q,
            repo_ids=acl.repo_ids,
            max_docs=max_docs,
            max_matches=max_matches,
            context_lines=context_lines,
            stream=uncapped,
        )
        base_matches, considered = self._parse_zoekt_payloads(
            payloads,
            acl=acl,
            domain_id=domain_id or "",
            suppressed_base=suppressed_base,
            branch_fallback=branch or "HEAD",
        )
        all_matches = overlay_matches + base_matches
        total_matched = len(all_matches)
        if uncapped or limit <= 0:
            selected = sorted(
                all_matches,
                key=lambda item: (-item.score, item.repo_name, item.file_path),
            )
            truncated = False
        else:
            selected = sorted(
                all_matches,
                key=lambda item: (-item.score, item.repo_name, item.file_path),
            )[:limit]
            truncated = total_matched > len(selected)

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return ZoektSearchResponse(
            query=query,
            file_matches=tuple(selected),
            stats=ZoektSearchStats(
                duration_ms=duration_ms,
                files_considered=considered + (len(overlay_deltas) if overlay_deltas else 0),
                files_matched=total_matched,
                matches_returned=len(selected),
                truncated=truncated,
                short_circuited_acl=False,
                authorized_repo_count=len(acl.repo_ids),
                authorized_repo_bitmap_popcount=acl.popcount,
            ),
        )

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
    ) -> Iterator[ZoektFileMatch]:
        response = self.search(
            query,
            principal,
            domain_id=domain_id,
            file_pattern=file_pattern,
            repo_names=repo_names,
            branch=branch,
            limit=UNCAPPED_ZOEK_DISPLAY_COUNT,
            uncapped=True,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            overlay_deltas=overlay_deltas,
        )
        yield from response.file_matches


__all__ = [
    "UNCAPPED_ZOEK_DISPLAY_COUNT",
    "LocalTrigramZoektIndex",
    "RE2Match",
    "RE2Pattern",
    "RE2SyntaxError",
    "RepoACLBitmapRegistry",
    "RepoACLResult",
    "TrigramQueryPlan",
    "ZoektClientSPI",
    "ZoektFileMatch",
    "ZoektGrpcJsonClient",
    "ZoektLineMatch",
    "ZoektSearchResponse",
    "ZoektSearchStats",
    "ZoektUnavailableError",
    "compile_re2",
    "extract_required_trigrams",
    "format_zoekt_shard_repo_name",
    "strip_zoekt_shard_repo_name",
    "validate_re2_pattern",
]
