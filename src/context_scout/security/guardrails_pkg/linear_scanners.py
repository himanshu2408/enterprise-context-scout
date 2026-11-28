"""O(n) linear-time HTML/chat-delimiter/script/event-handler state-machine scanners."""

from __future__ import annotations

import re
from typing import Final

from context_scout.security.guardrails_pkg.models_and_types import _fold_homoglyphs

_CHAT_TEMPLATE_DELIMITER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:<\s*(?:/\s*)?(?:start_of_turn|end_of_turn)\s*>|<turn\|\s*>|"
    r"<\|(?:im_start|im_end|system|user|assistant|developer|start_of_role|end_of_role|end_of_text)\|\s*>|"
    r"<<\s*(?:/\s*)?SYS\s*>>|\[\s*(?:/\s*)?INST\s*\])",
    re.IGNORECASE,
)

_SCRIPT_OPEN_RE: Final[re.Pattern[str]] = re.compile(r"<script\b", re.IGNORECASE)
_SCRIPT_CLOSE_RE: Final[re.Pattern[str]] = re.compile(
    r"</script(?:\s[^<>]*|/[^<>]*)?>",
    re.IGNORECASE,
)
_COMMENT_CLOSER_RE: Final[re.Pattern[str]] = re.compile(r"--!?>")
_SPECIAL_TAG_CHAR_RE: Final[re.Pattern[str]] = re.compile(r"[<>]")
_SPECIAL_ALL_CHAR_RE: Final[re.Pattern[str]] = re.compile(r"[<>\[\]]")
_ADVERSARIAL_COMMENT_KEYWORD_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:override|ignore|disregard|system|dump|exfiltrate|leak|"
    r"private\s+keys?|secret|password|database_url|instruction)\b",
    re.IGNORECASE,
)
_CHAT_GT_SKELETONS: Final[frozenset[str]] = frozenset(
    {
        "<start_of_turn",
        "</start_of_turn",
        "<end_of_turn",
        "</end_of_turn",
        "<turn|",
        "<|im_start|",
        "<|im_end|",
        "<|system|",
        "<|user|",
        "<|assistant|",
        "<|developer|",
        "<|start_of_role|",
        "<|end_of_role|",
        "<|end_of_text|",
        "<<sys>",
        "<</sys>",
    }
)


def _match_pending_chat_gt(
    out: list[str],
    non_ws_idx: list[int],
    lt_stack: list[tuple[int, int]],
    openers: list[tuple[str, int]],
) -> int | None:
    """Checks in O(1) whether out[-...] + '>' forms a split chat delimiter."""
    if not lt_stack:
        return None
    min_keep = (openers[-1][1] + 1) if openers else 0
    cand_idx, cand_nws = lt_stack[-1]
    if len(lt_stack) >= 2 and lt_stack[-2][0] == cand_idx - 1 and lt_stack[-2][0] >= min_keep:
        cand_idx, cand_nws = lt_stack[-2]
    if cand_idx < min_keep:
        return None
    nws_len = len(non_ws_idx)
    if nws_len - cand_nws > 24:
        return None
    skeleton = "".join(out[i] for i in non_ws_idx[cand_nws:]).lower()
    if skeleton not in _CHAT_GT_SKELETONS:
        return None
    if skeleton.startswith("<|") or skeleton == "<turn|":
        # No internal whitespace allowed inside <|...| or <turn|
        if non_ws_idx[-1] - non_ws_idx[cand_nws] != len(skeleton) - 1:
            return None
        return cand_idx
    if skeleton in {"<<sys>", "<</sys>"}:
        # Must end with '>' immediately before current '>' and '<<' at start
        if not out or out[-1] != ">":
            return None
        return cand_idx
    # < /? start_of_turn | end_of_turn >: keyword itself must be contiguous
    kw_len = 13 if "start_of_turn" in skeleton else 11
    if non_ws_idx[-1] - non_ws_idx[-kw_len] != kw_len - 1:
        return None
    return cand_idx


def _match_pending_chat_inst(
    out: list[str],
    non_ws_idx: list[int],
    lb_stack: list[tuple[int, int]],
    openers: list[tuple[str, int]],
) -> int | None:
    """Checks in O(1) whether out[-...] + ']' forms a split [/INST] delimiter."""
    if not lb_stack:
        return None
    min_keep = (openers[-1][1] + 1) if openers else 0
    cand_idx, cand_nws = lb_stack[-1]
    if cand_idx < min_keep or len(non_ws_idx) - cand_nws > 8:
        return None
    skeleton = "".join(out[i] for i in non_ws_idx[cand_nws:]).lower()
    if skeleton not in {"[inst", "[/inst"}:
        return None
    if non_ws_idx[-1] - non_ws_idx[-4] != 3:
        return None
    return cand_idx


def _strip_tokens_linear(
    text: str,
    *,
    strip_comments: bool = True,
    strip_scripts: bool = True,
    strip_chat_delimiters: bool = False,
    untrusted_tag_prefix_re: re.Pattern[str] | None = None,
    check_adversarial_comments: bool = False,
) -> tuple[str, bool]:
    """Single-pass O(n) opener-stack + bounded seam rollback token stripper."""
    if strip_scripts and ">" not in text:
        strip_scripts = False
    if strip_comments and not strip_scripts and "-->" not in text and "--!>" not in text:
        strip_comments = False
    if strip_chat_delimiters and ">" not in text and "]" not in text:
        strip_chat_delimiters = False
    if (
        not strip_comments
        and not strip_scripts
        and not strip_chat_delimiters
        and untrusted_tag_prefix_re is None
    ):
        return text, False
    if "<" not in text and (not strip_chat_delimiters or "[" not in text):
        return text, False
    if (
        strip_comments
        and not strip_scripts
        and not strip_chat_delimiters
        and untrusted_tag_prefix_re is None
        and ("<!--" not in text or ("-->" not in text and "--!>" not in text))
    ):
        return text, False

    out: list[str] = []
    lt_stack: list[tuple[int, int]] = []
    lb_stack: list[tuple[int, int]] = []
    non_ws_idx: list[int] = []
    openers: list[tuple[str, int]] = []
    counts: dict[str, int] = {"comment": 0, "script": 0, "untrusted_tag": 0}
    has_adversarial_comment = False
    no_closer_in_text = False
    unclosed_utag_script_gts: list[tuple[int, int]] = []

    head_str = ""
    head_pos = 0
    pos = 0
    n = len(text)

    def push_char(ch: str) -> None:
        idx = len(out)
        out.append(ch)
        if not ch.isspace():
            if ch == "<":
                lt_stack.append((idx, len(non_ws_idx)))
            elif ch == "[":
                lb_stack.append((idx, len(non_ws_idx)))
            non_ws_idx.append(idx)

    def push_str(segment: str) -> None:
        for ch in segment:
            push_char(ch)

    def truncate_out(k: int) -> None:
        while unclosed_utag_script_gts and k <= unclosed_utag_script_gts[-1][1]:
            unclosed_utag_script_gts.pop()
        del out[k:]
        while lt_stack and lt_stack[-1][0] >= k:
            lt_stack.pop()
        while lb_stack and lb_stack[-1][0] >= k:
            lb_stack.pop()
        while non_ws_idx and non_ws_idx[-1] >= k:
            non_ws_idx.pop()

    def pop_to_opener(target_kind: str) -> None:
        while openers:
            kind, start_idx = openers.pop()
            counts[kind] -= 1
            if kind == target_kind:
                truncate_out(start_idx)
                return

    def peek_str(length: int) -> str:
        rem = len(head_str) - head_pos
        if rem <= 0:
            return text[pos : pos + length]
        if rem >= length:
            return head_str[head_pos : head_pos + length]
        return head_str[head_pos:] + text[pos : pos + (length - rem)]

    def match_regex_at_cur(pattern: re.Pattern[str], window_len: int = 256) -> re.Match[str] | None:
        if head_pos >= len(head_str):
            return pattern.match(text, pos)
        window = head_str[head_pos:] + text[pos : pos + window_len]
        return pattern.match(window)

    def match_end_len(m: re.Match[str]) -> int:
        if head_pos >= len(head_str):
            return m.end() - pos
        return m.end()

    def advance_cur(count: int) -> None:
        nonlocal head_pos, head_str, pos
        rem = len(head_str) - head_pos
        if rem > 0:
            if count < rem:
                head_pos += count
                return
            count -= rem
            head_str = ""
            head_pos = 0
        pos += count

    def rollback_seam() -> None:
        nonlocal head_str, head_pos
        if not out or out[-1].isspace():
            return
        min_keep = (openers[-1][1] + 1) if openers else 0
        cur_len = len(out)
        cur_non_ws = len(non_ws_idx)
        rb_idx: int | None = None

        if lt_stack:
            cand_idx, cand_nws = lt_stack[-1]
            if (
                len(lt_stack) >= 2
                and lt_stack[-2][0] == cand_idx - 1
                and lt_stack[-2][0] >= min_keep
            ):
                cand_idx, cand_nws = lt_stack[-2]
            if cand_idx >= min_keep and cur_non_ws - cand_nws <= 32:
                prefix_len = (
                    2
                    if (
                        cand_nws + 1 < cur_non_ws
                        and non_ws_idx[cand_nws + 1] == cand_idx + 1
                        and out[cand_idx + 1] == "/"
                    )
                    else 1
                )
                has_gap_after_prefix = (
                    cand_nws + prefix_len < cur_non_ws
                    and non_ws_idx[cand_nws + prefix_len] > cand_idx + prefix_len
                )
                if has_gap_after_prefix:
                    word_start = non_ws_idx[cand_nws + prefix_len]
                    if cur_len - word_start <= 128:
                        rb_idx = word_start
                elif cur_len - cand_idx <= 128:
                    rb_idx = cand_idx

        if strip_chat_delimiters and lb_stack:
            b_idx, b_nws = lb_stack[-1]
            if (
                b_idx >= min_keep
                and cur_non_ws - b_nws <= 32
                and cur_len - b_idx <= 128
                and (b_nws + 1 >= cur_non_ws or non_ws_idx[b_nws + 1] == b_idx + 1)
                and (rb_idx is None or b_idx > rb_idx)
            ):
                rb_idx = b_idx

        if rb_idx is not None:
            rolled = "".join(out[rb_idx:])
            truncate_out(rb_idx)
            head_str = rolled + head_str[head_pos:] if head_pos < len(head_str) else rolled
            head_pos = 0

    def maybe_open_pending_untrusted_tag(cur_ch: str) -> bool:
        if (
            untrusted_tag_prefix_re is None
            or cur_ch.isspace()
            or cur_ch in "<>"
            or not lt_stack
            or not out
            or not out[-1].isspace()
        ):
            return False
        min_keep = (openers[-1][1] + 1) if openers else 0
        cand_idx, cand_nws = lt_stack[-1]
        if cand_idx < min_keep:
            return False
        nws_diff = len(non_ws_idx) - cand_nws
        if nws_diff == 1 and cur_ch != "/":
            prefix = "<"
        elif (
            nws_diff == 2 and non_ws_idx[cand_nws + 1] == cand_idx + 1 and out[cand_idx + 1] == "/"
        ):
            prefix = "</"
        else:
            return False
        window = prefix + peek_str(256)
        m_tag = untrusted_tag_prefix_re.match(window)
        if m_tag is None:
            return False
        m_len = m_tag.end() - len(prefix)
        openers.append(("untrusted_tag", cand_idx))
        counts["untrusted_tag"] += 1
        push_str(peek_str(m_len))
        advance_cur(m_len)
        return True

    while head_pos < len(head_str) or pos < n:
        if counts["comment"] > 0:
            if head_pos >= len(head_str):
                m_close = None if no_closer_in_text else _COMMENT_CLOSER_RE.search(text, pos)
                if m_close is None:
                    no_closer_in_text = True
                    openers = [(k, idx) for k, idx in openers if k != "comment"]
                    counts["comment"] = 0
                    continue
                c_start = openers[-1][1]
                if check_adversarial_comments and not has_adversarial_comment:
                    body = "".join(out[c_start + 4 :]) + text[pos : m_close.start()]
                    if _ADVERSARIAL_COMMENT_KEYWORD_RE.search(_fold_homoglyphs(body)):
                        has_adversarial_comment = True
                truncate_out(c_start)
                openers.pop()
                counts["comment"] -= 1
                pos = m_close.end()
                rollback_seam()
                continue

            # Drain rolled-back head_str while inside a comment
            ch = head_str[head_pos]
            if ch == "-":
                p3 = peek_str(3)
                if p3 == "-->":
                    c_start = openers[-1][1]
                    if check_adversarial_comments and not has_adversarial_comment:
                        body = "".join(out[c_start + 4 :])
                        if _ADVERSARIAL_COMMENT_KEYWORD_RE.search(_fold_homoglyphs(body)):
                            has_adversarial_comment = True
                    truncate_out(c_start)
                    openers.pop()
                    counts["comment"] -= 1
                    advance_cur(3)
                    rollback_seam()
                    continue
                p4 = peek_str(4)
                if p4 == "--!>":
                    c_start = openers[-1][1]
                    if check_adversarial_comments and not has_adversarial_comment:
                        body = "".join(out[c_start + 4 :])
                        if _ADVERSARIAL_COMMENT_KEYWORD_RE.search(_fold_homoglyphs(body)):
                            has_adversarial_comment = True
                    truncate_out(c_start)
                    openers.pop()
                    counts["comment"] -= 1
                    advance_cur(4)
                    rollback_seam()
                    continue
            push_char(ch)
            advance_cur(1)
            continue

        # Fast bulk jump when head_str is exhausted
        if head_pos >= len(head_str):
            if counts["script"] == 0 and counts["untrusted_tag"] == 0 and not strip_chat_delimiters:
                next_idx = text.find("<", pos)
            else:
                search_re = _SPECIAL_ALL_CHAR_RE if strip_chat_delimiters else _SPECIAL_TAG_CHAR_RE
                m_spec = search_re.search(text, pos)
                next_idx = m_spec.start() if m_spec is not None else -1
            scan_end = next_idx if next_idx != -1 else n
            if scan_end > pos and untrusted_tag_prefix_re is not None and lt_stack:
                ws_end = pos
                while ws_end < scan_end and text[ws_end].isspace():
                    ws_end += 1
                if ws_end > pos:
                    push_str(text[pos:ws_end])
                    pos = ws_end
                if pos < scan_end and maybe_open_pending_untrusted_tag(text[pos]):
                    continue
            if next_idx == -1:
                out.append(text[pos:])
                pos = n
                break
            if next_idx > pos:
                if not strip_chat_delimiters and untrusted_tag_prefix_re is None:
                    out.extend(text[pos:next_idx])
                else:
                    push_str(text[pos:next_idx])
                pos = next_idx

        ch = head_str[head_pos] if head_pos < len(head_str) else text[pos]

        if head_pos < len(head_str) and maybe_open_pending_untrusted_tag(ch):
            continue

        if ch == "<":
            in_text = head_pos >= len(head_str)
            c2 = (text[pos + 1] if pos + 1 < n else "") if in_text else peek_str(2)[1:2]
            if strip_comments and c2 == "!" and peek_str(4) == "<!--":
                if no_closer_in_text and head_pos >= len(head_str):
                    push_str("<!--")
                    advance_cur(4)
                    continue
                if head_pos >= len(head_str):
                    m_close = _COMMENT_CLOSER_RE.search(text, pos + 4)
                    if m_close is None:
                        no_closer_in_text = True
                        push_str("<!--")
                        advance_cur(4)
                        continue
                    if (
                        check_adversarial_comments
                        and not has_adversarial_comment
                        and (m_close.start() - (pos + 4)) >= 4
                        and _ADVERSARIAL_COMMENT_KEYWORD_RE.search(
                            _fold_homoglyphs(text[pos + 4 : m_close.start()])
                        )
                    ):
                        has_adversarial_comment = True
                    pos = m_close.end()
                    rollback_seam()
                    continue
                start_idx = len(out)
                push_str("<!--")
                openers.append(("comment", start_idx))
                counts["comment"] += 1
                advance_cur(4)
                continue

            if strip_scripts and counts["script"] > 0 and c2 == "/":
                m_sc_close = match_regex_at_cur(_SCRIPT_CLOSE_RE)
                if m_sc_close is not None:
                    pop_to_opener("script")
                    advance_cur(match_end_len(m_sc_close))
                    rollback_seam()
                    continue

            if strip_scripts and c2 in ("s", "S"):
                m_sc_open = match_regex_at_cur(_SCRIPT_OPEN_RE, window_len=16)
                if m_sc_open is not None:
                    start_idx = len(out)
                    push_str(peek_str(7))
                    openers.append(("script", start_idx))
                    counts["script"] += 1
                    advance_cur(7)
                    continue

            if untrusted_tag_prefix_re is not None and (c2 in ("/", "u", "U") or c2.isspace()):
                m_tag = match_regex_at_cur(untrusted_tag_prefix_re, window_len=256)
                if m_tag is not None:
                    m_len = match_end_len(m_tag)
                    start_idx = len(out)
                    push_str(peek_str(m_len))
                    openers.append(("untrusted_tag", start_idx))
                    counts["untrusted_tag"] += 1
                    advance_cur(m_len)
                    continue

            if strip_chat_delimiters and (
                c2 in ("|", "<", "/", "s", "S", "e", "E", "t", "T") or c2.isspace()
            ):
                m_chat = match_regex_at_cur(_CHAT_TEMPLATE_DELIMITER_RE, window_len=128)
                if m_chat is not None:
                    advance_cur(match_end_len(m_chat))
                    rollback_seam()
                    continue

            push_char("<")
            if in_text:
                pos += 1
            else:
                advance_cur(1)
            continue

        if ch == ">":
            if counts["untrusted_tag"] > 0 and openers and openers[-1][0] == "untrusted_tag":
                _, t_start = openers.pop()
                counts["untrusted_tag"] -= 1
                truncate_out(t_start)
                advance_cur(1)
                rollback_seam()
                continue

            if counts["script"] > 0:
                top_script_start = next(
                    (st for k, st in reversed(openers) if k == "script"),
                    -1,
                )
                if top_script_start != -1:
                    if len(non_ws_idx) >= 8:
                        s_idx = non_ws_idx[-8]
                        if (
                            s_idx > top_script_start
                            and non_ws_idx[-1] - s_idx == 7
                            and "".join(out[s_idx : s_idx + 8]).lower() == "</script"
                        ):
                            pop_to_opener("script")
                            advance_cur(1)
                            rollback_seam()
                            continue
                    if lt_stack:
                        cand_lt = lt_stack[-1][0]
                        if cand_lt > top_script_start:
                            tail = "".join(out[cand_lt : cand_lt + 9]).lower()
                            if tail == "</script" or (
                                len(tail) == 9
                                and tail.startswith("</script")
                                and (tail[8].isspace() or tail[8] == "/")
                            ):
                                pop_to_opener("script")
                                advance_cur(1)
                                rollback_seam()
                                continue

            if strip_chat_delimiters:
                chat_start = _match_pending_chat_gt(out, non_ws_idx, lt_stack, openers)
                if chat_start is not None:
                    truncate_out(chat_start)
                    advance_cur(1)
                    rollback_seam()
                    continue

            push_char(">")
            if counts["untrusted_tag"] > 0 and openers and openers[-1][0] == "script":
                utag_start = next(
                    (st for k, st in reversed(openers) if k == "untrusted_tag"),
                    -1,
                )
                if utag_start != -1 and (
                    not unclosed_utag_script_gts or unclosed_utag_script_gts[-1][0] != utag_start
                ):
                    unclosed_utag_script_gts.append((utag_start, len(out)))
            advance_cur(1)
            continue

        if strip_chat_delimiters and ch == "[":
            m_inst = match_regex_at_cur(_CHAT_TEMPLATE_DELIMITER_RE, window_len=64)
            if m_inst is not None:
                advance_cur(match_end_len(m_inst))
                rollback_seam()
                continue
            push_char("[")
            advance_cur(1)
            continue

        if strip_chat_delimiters and ch == "]":
            inst_start = _match_pending_chat_inst(out, non_ws_idx, lb_stack, openers)
            if inst_start is not None:
                truncate_out(inst_start)
                advance_cur(1)
                rollback_seam()
                continue
            push_char("]")
            advance_cur(1)
            continue

        push_char(ch)
        advance_cur(1)

    for utag_start, gt_end in reversed(unclosed_utag_script_gts):
        if counts["untrusted_tag"] <= 0:
            break
        if 0 <= utag_start < gt_end <= len(out):
            delta = gt_end - utag_start
            del out[utag_start:gt_end]
            shifted_openers: list[tuple[str, int]] = []
            for kind, st in openers:
                if utag_start <= st < gt_end:
                    counts[kind] -= 1
                elif st >= gt_end:
                    shifted_openers.append((kind, st - delta))
                else:
                    shifted_openers.append((kind, st))
            openers = shifted_openers
    if counts["untrusted_tag"] > 0:
        first_utag = next((st for k, st in openers if k == "untrusted_tag"), -1)
        if 0 <= first_utag < len(out):
            del out[first_utag:]

    return "".join(out), has_adversarial_comment


def _has_adversarial_html_comment_linear(folded: str) -> bool:
    """Detects adversarial directives inside closed/reconstituted HTML comments in O(n) time."""
    if "!" not in folded or "--" not in folded or ">" not in folded:
        return False
    if (
        "<!--" not in folded or ("-->" not in folded and "--!>" not in folded)
    ) and "<script" not in folded.lower():
        return False
    _, has_adv = _strip_tokens_linear(
        folded,
        strip_comments=True,
        strip_scripts=True,
        check_adversarial_comments=True,
    )
    return has_adv


_HTML_EVENT_HANDLER_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "onabort",
        "onactivate",
        "onafterprint",
        "onafterscriptexecute",
        "onanimationcancel",
        "onanimationend",
        "onanimationiteration",
        "onanimationstart",
        "onauxclick",
        "onbeforeactivate",
        "onbeforecopy",
        "onbeforecut",
        "onbeforedeactivate",
        "onbeforeinput",
        "onbeforematch",
        "onbeforepaste",
        "onbeforeprint",
        "onbeforescriptexecute",
        "onbeforetoggle",
        "onbeforeunload",
        "onbegin",
        "onblur",
        "onbounce",
        "oncancel",
        "oncanplay",
        "oncanplaythrough",
        "onchange",
        "onclick",
        "onclose",
        "oncommand",
        "oncontentvisibilityautostatechange",
        "oncontextlost",
        "oncontextmenu",
        "oncontextrestored",
        "oncopy",
        "oncuechange",
        "oncut",
        "ondblclick",
        "ondeactivate",
        "ondrag",
        "ondragend",
        "ondragenter",
        "ondragleave",
        "ondragover",
        "ondragstart",
        "ondrop",
        "ondurationchange",
        "onemptied",
        "onend",
        "onended",
        "onerror",
        "onfinish",
        "onfocus",
        "onfocusin",
        "onfocusout",
        "onformdata",
        "onfullscreenchange",
        "onfullscreenerror",
        "ongotpointercapture",
        "onhashchange",
        "oninput",
        "oninvalid",
        "onkeydown",
        "onkeypress",
        "onkeyup",
        "onload",
        "onloadeddata",
        "onloadedmetadata",
        "onloadstart",
        "onlostpointercapture",
        "onmessage",
        "onmessageerror",
        "onmousedown",
        "onmouseenter",
        "onmouseleave",
        "onmousemove",
        "onmouseout",
        "onmouseover",
        "onmouseup",
        "onmousewheel",
        "onmove",
        "onoffline",
        "ononline",
        "onpagehide",
        "onpagereveal",
        "onpageshow",
        "onpageswap",
        "onpaste",
        "onpause",
        "onplay",
        "onplaying",
        "onpointercancel",
        "onpointerdown",
        "onpointerenter",
        "onpointerleave",
        "onpointermove",
        "onpointerout",
        "onpointerover",
        "onpointerrawupdate",
        "onpointerup",
        "onpopstate",
        "onprogress",
        "onratechange",
        "onrepeat",
        "onreset",
        "onresize",
        "onscroll",
        "onscrollend",
        "onscrollsnapchange",
        "onscrollsnapchanging",
        "onsearch",
        "onsecuritypolicyviolation",
        "onseeked",
        "onseeking",
        "onselect",
        "onselectionchange",
        "onselectstart",
        "onshow",
        "onslotchange",
        "onstalled",
        "onstart",
        "onstorage",
        "onsubmit",
        "onsuspend",
        "ontimeupdate",
        "ontoggle",
        "ontouchcancel",
        "ontouchend",
        "ontouchmove",
        "ontouchstart",
        "ontransitioncancel",
        "ontransitionend",
        "ontransitionrun",
        "ontransitionstart",
        "onunhandledrejection",
        "onunload",
        "onvolumechange",
        "onwaiting",
        "onwebkitanimationend",
        "onwebkitanimationiteration",
        "onwebkitanimationstart",
        "onwebkittransitionend",
        "onwheel",
    }
)
_HTML_WS: Final[str] = " \t\n\r\f"
_HTML_WS_RE: Final[re.Pattern[str]] = re.compile(r"[ \t\n\r\f]+")
_HTML_TAG_NAME_SKIP_RE: Final[re.Pattern[str]] = re.compile(r"[^ \t\n\r\f/>]+")
_HTML_ATTR_NAME_SKIP_RE: Final[re.Pattern[str]] = re.compile(r"[^ \t\n\r\f/=>]+")
_HTML_ATTR_UNQUOTED_SKIP_RE: Final[re.Pattern[str]] = re.compile(r"[^ \t\n\r\f>]+")

_HTML_STATE_DATA: Final[int] = 0
_HTML_STATE_TAG_NAME: Final[int] = 1
_HTML_STATE_BEFORE_ATTR_NAME: Final[int] = 2
_HTML_STATE_ATTR_NAME: Final[int] = 3
_HTML_STATE_AFTER_ATTR_NAME: Final[int] = 4
_HTML_STATE_BEFORE_ATTR_VALUE: Final[int] = 5
_HTML_STATE_ATTR_VALUE_DOUBLE: Final[int] = 6
_HTML_STATE_ATTR_VALUE_SINGLE: Final[int] = 7
_HTML_STATE_ATTR_VALUE_UNQUOTED: Final[int] = 8
_HTML_STATE_AFTER_ATTR_VALUE_QUOTED: Final[int] = 9
_HTML_STATE_SELF_CLOSING_START_TAG: Final[int] = 10


def _has_inline_html_event_handler_linear(folded: str) -> bool:
    """Scans for inline HTML event handlers via WHATWG HTML5 tag-attribute state machine in O(n)."""
    if "<" not in folded or "=" not in folded or "on" not in folded.lower():
        return False
    n = len(folded)
    pos = 0
    state = _HTML_STATE_DATA
    attr_name = ""
    attr_start = 0

    while pos < n:
        if state == _HTML_STATE_DATA:
            lt = folded.find("<", pos)
            if lt == -1:
                return False
            pos = lt + 1
            if pos < n:
                ch = folded[pos]
                if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
                    pos += 1
                    state = _HTML_STATE_TAG_NAME
            continue

        if state == _HTML_STATE_ATTR_VALUE_DOUBLE:
            q_end = folded.find('"', pos)
            if q_end == -1:
                return False
            pos = q_end + 1
            state = _HTML_STATE_AFTER_ATTR_VALUE_QUOTED
            continue

        if state == _HTML_STATE_ATTR_VALUE_SINGLE:
            q_end = folded.find("'", pos)
            if q_end == -1:
                return False
            pos = q_end + 1
            state = _HTML_STATE_AFTER_ATTR_VALUE_QUOTED
            continue

        if state == _HTML_STATE_TAG_NAME:
            m = _HTML_TAG_NAME_SKIP_RE.match(folded, pos)
            if m is not None:
                pos = m.end()
                if pos >= n:
                    return False
            ch = folded[pos]
            if ch in _HTML_WS:
                pos += 1
                state = _HTML_STATE_BEFORE_ATTR_NAME
            elif ch == "/":
                pos += 1
                state = _HTML_STATE_SELF_CLOSING_START_TAG
            else:
                pos += 1
                state = _HTML_STATE_DATA
            continue

        if state == _HTML_STATE_BEFORE_ATTR_NAME:
            m_ws = _HTML_WS_RE.match(folded, pos)
            if m_ws is not None:
                pos = m_ws.end()
                if pos >= n:
                    return False
            ch = folded[pos]
            if ch == "/":
                pos += 1
                state = _HTML_STATE_SELF_CLOSING_START_TAG
            elif ch == ">":
                pos += 1
                state = _HTML_STATE_DATA
            else:
                attr_start = pos
                pos += 1
                state = _HTML_STATE_ATTR_NAME
            continue

        if state == _HTML_STATE_ATTR_NAME:
            m = _HTML_ATTR_NAME_SKIP_RE.match(folded, pos)
            if m is not None:
                pos = m.end()
                if pos >= n:
                    return False
            ch = folded[pos]
            if ch in _HTML_WS:
                attr_name = folded[attr_start:pos].lower()
                pos += 1
                state = _HTML_STATE_AFTER_ATTR_NAME
            elif ch == "/":
                pos += 1
                state = _HTML_STATE_SELF_CLOSING_START_TAG
            elif ch == "=":
                attr_name = folded[attr_start:pos].lower()
                if attr_name in _HTML_EVENT_HANDLER_ATTRS:
                    return True
                pos += 1
                state = _HTML_STATE_BEFORE_ATTR_VALUE
            else:
                pos += 1
                state = _HTML_STATE_DATA
            continue

        if state == _HTML_STATE_AFTER_ATTR_NAME:
            m_ws = _HTML_WS_RE.match(folded, pos)
            if m_ws is not None:
                pos = m_ws.end()
                if pos >= n:
                    return False
            ch = folded[pos]
            if ch == "/":
                pos += 1
                state = _HTML_STATE_SELF_CLOSING_START_TAG
            elif ch == "=":
                if attr_name in _HTML_EVENT_HANDLER_ATTRS:
                    return True
                pos += 1
                state = _HTML_STATE_BEFORE_ATTR_VALUE
            elif ch == ">":
                pos += 1
                state = _HTML_STATE_DATA
            else:
                state = _HTML_STATE_BEFORE_ATTR_NAME
            continue

        if state == _HTML_STATE_BEFORE_ATTR_VALUE:
            m_ws = _HTML_WS_RE.match(folded, pos)
            if m_ws is not None:
                pos = m_ws.end()
                if pos >= n:
                    return False
            ch = folded[pos]
            if ch == '"':
                pos += 1
                state = _HTML_STATE_ATTR_VALUE_DOUBLE
            elif ch == "'":
                pos += 1
                state = _HTML_STATE_ATTR_VALUE_SINGLE
            elif ch == ">":
                pos += 1
                state = _HTML_STATE_DATA
            else:
                state = _HTML_STATE_ATTR_VALUE_UNQUOTED
            continue

        if state == _HTML_STATE_ATTR_VALUE_UNQUOTED:
            m = _HTML_ATTR_UNQUOTED_SKIP_RE.match(folded, pos)
            if m is not None:
                pos = m.end()
                if pos >= n:
                    return False
            ch = folded[pos]
            if ch in _HTML_WS:
                pos += 1
                state = _HTML_STATE_BEFORE_ATTR_NAME
            else:
                pos += 1
                state = _HTML_STATE_DATA
            continue

        if state == _HTML_STATE_AFTER_ATTR_VALUE_QUOTED:
            ch = folded[pos]
            if ch in _HTML_WS:
                pos += 1
                state = _HTML_STATE_BEFORE_ATTR_NAME
            elif ch == "/":
                pos += 1
                state = _HTML_STATE_SELF_CLOSING_START_TAG
            elif ch == ">":
                pos += 1
                state = _HTML_STATE_DATA
            else:
                state = _HTML_STATE_BEFORE_ATTR_NAME
            continue

        ch = folded[pos]
        if ch == ">":
            pos += 1
            state = _HTML_STATE_DATA
        else:
            state = _HTML_STATE_BEFORE_ATTR_NAME

    return False
