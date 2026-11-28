"""Byte-exact source line slicer and NFKC quote verification engine (VAL-02)."""

import re
import unicodedata
from typing import Final

from context_scout.ast_engine.models import (
    HallucinatedQuoteError,
    InvalidLineBoundsError,
    OffByOneBoundsError,
    QuoteVerificationError,
    QuoteVerificationResult,
    SlicerErrorCode,
    TruncatedLineError,
)
from context_scout.db.repositories import CodeFileRecord
from context_scout.schemas.contracts import CodeSlicePointer, DocCitationSpan, VerbatimCodeSlice


class DeterministicSlicer:
    """Byte-exact source line slicer and NFKC quote verification engine (VAL-02)."""

    MIN_QUOTE_OVERLAP_RATIO: Final[float] = 1.0
    DEFAULT_SKELETONIZE_CLASS_LINE_THRESHOLD: Final[int] = 450

    def __init__(
        self,
        skeletonize_class_line_threshold: int = DEFAULT_SKELETONIZE_CLASS_LINE_THRESHOLD,
    ) -> None:
        self.skeletonize_class_line_threshold = skeletonize_class_line_threshold

    def slice_lines(
        self,
        raw_content: str,
        start_line: int,
        end_line: int,
        line_number_width: int = 4,
    ) -> str:
        """Extracts verbatim lines with 1-indexed numbers and fixed width.

        Format: '{line_no:4d} | {line_content}'
        """
        lines = raw_content.splitlines()
        total_lines = len(lines)

        self.validate_bounds(start_line, end_line, total_lines)

        selected = lines[start_line - 1 : end_line]
        formatted = [
            f"{line_no:{line_number_width}d} | {line}"
            for line_no, line in enumerate(selected, start=start_line)
        ]
        return "\n".join(formatted)

    def validate_bounds(self, start_line: int, end_line: int, total_lines: int) -> None:
        """Validates 1-indexed line numbers and flags off-by-one errors."""
        if start_line == 0:
            raise OffByOneBoundsError(
                "0-indexed start_line detected; line numbers must be 1-indexed"
            )
        if start_line < 1:
            raise InvalidLineBoundsError(f"start_line ({start_line}) must be >= 1")
        if end_line < start_line:
            raise InvalidLineBoundsError(
                f"end_line ({end_line}) cannot be less than start_line ({start_line})"
            )
        if total_lines == 0:
            raise InvalidLineBoundsError("Cannot slice empty file")
        if start_line > total_lines:
            raise InvalidLineBoundsError(
                f"start_line ({start_line}) exceeds file line count ({total_lines})"
            )
        if end_line == total_lines + 1:
            raise OffByOneBoundsError(
                f"end_line ({end_line}) exceeds file line count ({total_lines}) by 1"
            )
        if end_line > total_lines:
            raise InvalidLineBoundsError(
                f"end_line ({end_line}) exceeds file line count ({total_lines})"
            )

    def slice_pointer(
        self,
        pointer: CodeSlicePointer,
        file_record: CodeFileRecord,
        auto_expanded_by_ast: bool = False,
        query_text: str | None = None,
    ) -> VerbatimCodeSlice:
        """Hydrates a CodeSlicePointer with byte-exact code and git commit/blob provenance."""
        verbatim_code = self.slice_lines(
            raw_content=file_record.raw_content,
            start_line=pointer.start_line,
            end_line=pointer.end_line,
        )
        active_query = f"{query_text or ''} {pointer.selection_reason or ''}".strip()
        lang_lower = (file_record.language or "").lower()
        if (pointer.end_line - pointer.start_line) > self.skeletonize_class_line_threshold:
            if lang_lower == "python":
                verbatim_code = self._skeletonize_large_python_class(
                    raw_content=file_record.raw_content,
                    start_line=pointer.start_line,
                    end_line=pointer.end_line,
                    query_text=active_query,
                    is_reference_only=(pointer.modification_intent.value == "reference_only"),
                )
            elif lang_lower in {
                "typescript",
                "tsx",
                "javascript",
                "jsx",
                "csharp",
                "java",
                "kotlin",
                "go",
                "rust",
            }:
                verbatim_code = self._skeletonize_large_brace_class(
                    raw_content=file_record.raw_content,
                    start_line=pointer.start_line,
                    end_line=pointer.end_line,
                    query_text=active_query,
                    file_path=file_record.file_path,
                    is_reference_only=(pointer.modification_intent.value == "reference_only"),
                )
        commit_sha = file_record.git_commit_sha or "0000000"
        if len(commit_sha) < 7:
            commit_sha = commit_sha.ljust(7, "0")

        blob_sha = file_record.git_blob_sha or "0000000"
        if len(blob_sha) < 7:
            blob_sha = blob_sha.ljust(7, "0")

        return VerbatimCodeSlice(
            repo_name=pointer.repo_name,
            file_path=pointer.file_path,
            start_line=pointer.start_line,
            end_line=pointer.end_line,
            qualified_symbol=pointer.qualified_symbol,
            selection_reason=pointer.selection_reason,
            modification_intent=pointer.modification_intent,
            git_commit_sha=commit_sha,
            git_blob_sha=blob_sha,
            verbatim_code_with_lines=verbatim_code,
            auto_expanded_by_ast=auto_expanded_by_ast,
        )

    def _skeletonize_large_python_class(
        self,
        raw_content: str,
        start_line: int,
        end_line: int,
        query_text: str,
        line_number_width: int = 4,
        is_reference_only: bool = False,
    ) -> str:
        """Preserves class header, __init__, and top query-matched methods while eliding others."""
        all_lines = raw_content.splitlines()
        selected = all_lines[start_line - 1 : end_line]
        if not selected or not selected[0].lstrip().startswith("class "):
            return self.slice_lines(raw_content, start_line, end_line, line_number_width)

        class_indent = len(selected[0]) - len(selected[0].lstrip())
        method_indent = class_indent + 4

        method_spans: list[tuple[str, int, int]] = []
        current_name: str | None = None
        current_start: int = 0

        for idx, line in enumerate(selected):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent == method_indent and (
                stripped.startswith("def ") or stripped.startswith("async def ")
            ):
                if current_name is not None:
                    method_spans.append((current_name, current_start, idx - 1))
                m_match = re.match(r"(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
                current_name = m_match.group(1) if m_match else "method"
                current_start = idx

        if current_name is not None:
            method_spans.append((current_name, current_start, len(selected) - 1))

        if len(method_spans) <= 3:
            return self.slice_lines(raw_content, start_line, end_line, line_number_width)

        _skeleton_stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "into",
            "that",
            "this",
            "how",
            "see",
            "are",
            "was",
            "were",
            "can",
            "all",
            "any",
            "not",
            "use",
            "using",
            "inspect",
            "located",
            "symbol",
            "user",
            "task",
            "selected",
            "primary",
            "match",
            "file",
            "line",
            "lines",
            "method",
            "class",
            "function",
            "object",
            "src",
        }
        q_terms = {
            w.lower()
            for w in re.findall(r"[A-Za-z0-9_]{3,}", query_text)
            if w.lower() not in _skeleton_stopwords
        }
        scored_methods: list[tuple[int, str]] = []
        matched_methods: set[str] = set()
        for m_name, m_start, m_end in method_spans:
            if m_name == "__init__":
                continue
            m_body = "\n".join(selected[m_start : m_end + 1]).lower()
            score = 0
            m_lower = m_name.lower()
            m_parts = set(m_lower.split("_"))
            if m_lower in q_terms:
                score += 10
                matched_methods.add(m_name)
            for qt in q_terms:
                if len(qt) >= 5 and (qt == m_lower or qt in m_parts):
                    score += 5
                    matched_methods.add(m_name)
                elif len(qt) >= 5 and re.search(rf"\b{re.escape(qt)}\s*\(", m_body):
                    score += 3
                    matched_methods.add(m_name)
                elif qt in m_body:
                    score += 1
            scored_methods.append((score, m_name))

        scored_methods.sort(key=lambda item: (-item[0], item[1]))
        if is_reference_only:
            keep_names: set[str] = set()
        else:
            bounded_matched = {name for _, name in scored_methods[:8] if name in matched_methods}
            keep_names = {name for _, name in scored_methods[:3]} | bounded_matched
        init_span = next((m for m in method_spans if m[0] == "__init__"), None)
        if init_span is not None and (init_span[2] - init_span[1]) <= 40:
            keep_names.add("__init__")

        output_lines: list[str] = []
        first_method_idx = method_spans[0][1]
        for idx in range(first_method_idx):
            line_no = start_line + idx
            output_lines.append(f"{line_no:{line_number_width}d} | {selected[idx]}")

        for m_name, m_start, m_end in method_spans:
            if m_name in keep_names or (m_end - m_start) <= 1:
                for idx in range(m_start, m_end + 1):
                    line_no = start_line + idx
                    output_lines.append(f"{line_no:{line_number_width}d} | {selected[idx]}")
            else:
                # Find the end line of the signature (delimiters balanced and line contains :)
                sig_end = m_start
                open_delims = 0
                for idx in range(m_start, m_end + 1):
                    ln = selected[idx]
                    cleaned_ln, _ = self._strip_comments_and_strings(ln, file_path="a.py")
                    for ch in cleaned_ln:
                        if ch in "([{":
                            open_delims += 1
                        elif ch in ")]}":
                            open_delims = max(0, open_delims - 1)
                    sig_end = idx
                    if open_delims == 0 and ":" in cleaned_ln:
                        break

                for idx in range(m_start, sig_end):
                    line_no = start_line + idx
                    output_lines.append(f"{line_no:{line_number_width}d} | {selected[idx]}")

                sig_line_no = start_line + sig_end
                sig_text = selected[sig_end].rstrip()
                if sig_end < m_end:
                    elided_start = start_line + sig_end + 1
                    elided_end = start_line + m_end
                    colon_idx = sig_text.rfind(":")
                    if colon_idx != -1:
                        before_colon = sig_text[: colon_idx + 1]
                        after_colon = sig_text[colon_idx + 1 :].strip()
                        if after_colon.startswith("#"):
                            comment_extra = f" {after_colon}"
                        elif after_colon:
                            comment_extra = f" # {after_colon}"
                        else:
                            comment_extra = ""
                        output_lines.append(
                            f"{sig_line_no:{line_number_width}d} | {before_colon} "
                            f"...  # [lines {elided_start}-{elided_end} elided]{comment_extra}"
                        )
                    else:
                        output_lines.append(
                            f"{sig_line_no:{line_number_width}d} | {sig_text} "
                            f"...  # [lines {elided_start}-{elided_end} elided]"
                        )
                else:
                    output_lines.append(f"{sig_line_no:{line_number_width}d} | {sig_text}")

        return "\n".join(output_lines)

    def _skeletonize_large_brace_class(
        self,
        raw_content: str,
        start_line: int,
        end_line: int,
        query_text: str,
        *,
        file_path: str = "",
        line_number_width: int = 4,
        is_reference_only: bool = False,
    ) -> str:
        """Preserves brace-class header, ctor, and query-matched methods cleanly."""
        all_lines = raw_content.splitlines()
        selected = all_lines[start_line - 1 : end_line]
        if not selected:
            return self.slice_lines(raw_content, start_line, end_line, line_number_width)

        def fmt(idx_in_sel: int) -> str:
            return f"{start_line + idx_in_sel:>{line_number_width}} | {selected[idx_in_sel]}"

        depth = 0
        class_open_idx: int | None = None
        for idx, raw_line in enumerate(selected[:15]):
            cleaned, _ = self._strip_comments_and_strings(raw_line, file_path=file_path)
            if "{" in cleaned:
                class_open_idx = idx
                break
        if class_open_idx is None:
            return self.slice_lines(raw_content, start_line, end_line, line_number_width)

        blocks: list[tuple[int, int, str]] = []
        fields_indices: list[int] = []
        depth = 0
        member_start: int | None = None
        entered_body = False
        for idx, raw_line in enumerate(selected):
            cleaned, _ = self._strip_comments_and_strings(raw_line, file_path=file_path)
            prev_depth = depth
            depth += cleaned.count("{") - cleaned.count("}")
            if idx <= class_open_idx:
                if idx == class_open_idx and depth > 0:
                    entered_body = True
                continue
            if prev_depth <= 1 and depth >= 2 and member_start is None:
                member_start = idx
            elif member_start is not None and depth <= 1:
                blocks.append((member_start, idx, selected[member_start]))
                member_start = None
            elif prev_depth == 1 and depth == 1 and member_start is None and entered_body:
                fields_indices.append(idx)

        if len(blocks) < 3:
            return self.slice_lines(raw_content, start_line, end_line, line_number_width)

        query_terms = {
            tok.lower()
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query_text)
            if tok.lower() not in {"class", "const", "async", "public", "private", "method"}
        }
        keep_indices: set[int] = set(range(class_open_idx + 1))
        keep_indices.update(fields_indices)
        keep_indices.add(len(selected) - 1)

        scored: list[tuple[int, int]] = []
        for b_idx, (b_start, b_end, sig_line) in enumerate(blocks):
            block_text = "\n".join(selected[b_start : b_end + 1]).lower()
            is_ctor = bool(re.search(r"\b(constructor|__init__|new)\s*\(", sig_line))
            hits = sum(1 for qt in query_terms if qt in block_text)
            if is_ctor:
                keep_indices.update(range(b_start, b_end + 1))
            else:
                scored.append((hits, b_idx))

        max_kept = 2 if is_reference_only else 4
        matched_blocks: set[int] = set()
        for hits, b_idx in sorted(scored, key=lambda x: (-x[0], x[1])):
            if hits > 0 and len(matched_blocks) < max_kept:
                matched_blocks.add(b_idx)
        if not matched_blocks and not is_reference_only and scored:
            matched_blocks.add(scored[0][1])

        output: list[str] = []
        elided_block_ranges: dict[int, tuple[int, int]] = {}
        for b_idx, (b_start, b_end, _sig) in enumerate(blocks):
            if b_idx in matched_blocks:
                keep_indices.update(range(b_start, b_end + 1))
            elif b_start not in keep_indices:
                elided_block_ranges[b_start] = (b_start, b_end)

        i = 0
        n = len(selected)
        while i < n:
            if i in keep_indices:
                output.append(fmt(i))
                i += 1
            elif i in elided_block_ranges:
                b_start, b_end = elided_block_ranges[i]
                open_idx = b_start
                while open_idx <= b_end and "{" not in selected[open_idx]:
                    output.append(fmt(open_idx))
                    open_idx += 1
                if open_idx <= b_end:
                    output.append(fmt(open_idx))
                if b_end > open_idx + 1:
                    output.append("     ... |   /* ... elided ... */")
                if b_end > open_idx:
                    output.append(fmt(b_end))
                i = b_end + 1
            else:
                i += 1

        return "\n".join(output)

    def normalize_text(self, text: str) -> str:
        """Applies NFKC Unicode normalization and collapses whitespace."""
        nfkc = unicodedata.normalize("NFKC", text)
        return re.sub(r"\s+", " ", nfkc).strip()

    def tokenize(self, text: str) -> list[str]:
        """Deterministic tokenization over words and punctuation."""
        return re.findall(r"\w+|[^\w\s]", text)

    @staticmethod
    def _is_prose_apostrophe(text: str, idx: int) -> bool:
        """Return True when ``text[idx] == "'"`` is an English contraction/possessive apostrophe."""
        if idx <= 0 or not text[idx - 1].isalnum():
            return False
        n = len(text)
        if idx + 1 < n and text[idx + 1].isalnum():
            return True
        return text[idx - 1] in "sS" and (
            idx + 1 == n or text[idx + 1].isspace() or text[idx + 1] in '.,;:!?)}]"-'
        )

    @staticmethod
    def _strip_comments_and_strings(
        code: str,
        file_path: str = "",
        *,
        is_doc_prose: bool = False,
    ) -> tuple[str, bool]:
        """Strips comments and string literals across Python, TypeScript, and C#.

        When ``is_doc_prose=True``, prose hashtags (``#42``, ``C#``) and English
        contraction/possessive apostrophes (``Brewer's``, ``O'Brien``, ``patients'``)
        are treated as prose characters rather than comment/string delimiters.

        Returns:
            (sanitized_code, has_unclosed_quote_or_comment)
        """
        ext = file_path.lower()
        is_rust = ext.endswith(".rs")
        is_hash_comment_lang = not is_doc_prose and not ext.endswith(
            (".ts", ".tsx", ".js", ".jsx", ".cs", ".java", ".c", ".cpp", ".rs", ".go")
        )

        chars: list[str] = []
        i = 0
        n = len(code)
        has_unclosed_quote = False

        while i < n:
            # Python triple-double quotes \"\"\"
            if code[i : i + 3] == '"""':
                end = code.find('"""', i + 3)
                if end == -1:
                    has_unclosed_quote = True
                    break
                i = end + 3
                chars.append(" ")
                continue
            # Python triple-single quotes '''
            if code[i : i + 3] == "'''":
                end = code.find("'''", i + 3)
                if end == -1:
                    has_unclosed_quote = True
                    break
                i = end + 3
                chars.append(" ")
                continue
            # C# / TS / C block comments /* ... */
            if not is_hash_comment_lang and code[i : i + 2] == "/*":
                end = code.find("*/", i + 2)
                if end == -1:
                    has_unclosed_quote = True
                    break
                i = end + 2
                chars.append(" ")
                continue
            # Single-line comment // (TS, JS, C#, Java, C, C++)
            if not is_hash_comment_lang and code[i : i + 2] == "//":
                end = code.find("\n", i + 2)
                if end == -1:
                    break
                i = end
                continue
            # Single-line comment # (Python, Terraform, Shell)
            if code[i] == "#" and is_hash_comment_lang:
                end = code.find("\n", i + 1)
                if end == -1:
                    break
                i = end
                continue
            # Double-quoted string literal "..."
            if code[i] == '"':
                j = i + 1
                closed = False
                while j < n:
                    if code[j] == "\\":
                        j += 2
                        continue
                    if code[j] == '"':
                        closed = True
                        j += 1
                        break
                    if code[j] == "\n":
                        break
                    j += 1
                if not closed:
                    has_unclosed_quote = True
                    break
                i = j
                chars.append(" ")
                continue
            # Single-quoted string literal '...' (or prose apostrophe / Rust lifetime)
            if code[i] == "'":
                if is_doc_prose and DeterministicSlicer._is_prose_apostrophe(code, i):
                    chars.append(code[i])
                    i += 1
                    continue
                if is_rust:
                    # Rust char literal is 'x' or '\x'; anything like 'a, 'py, 'static, '_
                    # without a closing ' right after a single char/escape is a lifetime/label.
                    is_char_lit = (i + 2 < n and code[i + 1] != "\\" and code[i + 2] == "'") or (
                        i + 3 < n
                        and code[i + 1] == "\\"
                        and (code[i + 3] == "'" or (i + 5 < n and code[i + 2] == "x"))
                    )
                    if not is_char_lit:
                        chars.append(" ")
                        i += 1
                        while i < n and (code[i].isalnum() or code[i] == "_"):
                            i += 1
                        continue
                j = i + 1
                closed = False
                while j < n:
                    if code[j] == "\\":
                        j += 2
                        continue
                    if code[j] == "'":
                        if (
                            is_doc_prose
                            and j > i + 1
                            and code[j - 1].isalnum()
                            and j + 1 < n
                            and code[j + 1].isalnum()
                        ):
                            j += 1
                            continue
                        closed = True
                        j += 1
                        break
                    if code[j] == "\n":
                        break
                    j += 1
                if not closed:
                    has_unclosed_quote = True
                    break
                i = j
                chars.append(" ")
                continue
            # Backtick template literal `...`
            if code[i] == "`":
                j = i + 1
                closed = False
                while j < n:
                    if code[j] == "\\":
                        j += 2
                        continue
                    if code[j] == "`":
                        closed = True
                        j += 1
                        break
                    j += 1
                if not closed:
                    has_unclosed_quote = True
                    break
                i = j
                chars.append(" ")
                continue

            chars.append(code[i])
            i += 1

        return "".join(chars), has_unclosed_quote

    def _has_unclosed_delimiters(
        self,
        text: str,
        file_path: str = "",
        *,
        is_doc_prose: bool = False,
    ) -> bool:
        cleaned, unclosed_quote = self._strip_comments_and_strings(
            text, file_path=file_path, is_doc_prose=is_doc_prose
        )
        if unclosed_quote:
            return True
        # Strip standard half-open mathematical interval notation (e.g., "[0, T)", "(0, 1]")
        cleaned = re.sub(
            r"(?:(?:\bin\b|[∈=:])\s*)?[\[\(]\s*[^\]\)\(\[\n,;]{1,20}\s*,\s*[^\]\)\(\[\n,;]{1,20}\s*[\]\)]",
            "",
            cleaned,
        )
        counts = {
            "(": cleaned.count("(") - cleaned.count(")"),
            "[": cleaned.count("[") - cleaned.count("]"),
            "{": cleaned.count("{") - cleaned.count("}"),
        }
        return any(diff > 0 for diff in counts.values())

    def verify_quote(
        self,
        quote: str,
        source_text: str,
        min_overlap_ratio: float = MIN_QUOTE_OVERLAP_RATIO,
        strict: bool = False,
        reject_absent_tokens: bool = False,
        is_doc_prose: bool = False,
    ) -> QuoteVerificationResult:
        """Verifies quote against source text using NFKC normalization and token overlap.

        Implements VAL-02 verification.
        """
        if not quote.strip():
            raise QuoteVerificationError(
                SlicerErrorCode.QUOTE_MISMATCH, "Empty quote cannot be verified"
            )

        norm_quote = self.normalize_text(quote)
        norm_source = self.normalize_text(source_text)

        # Check for unclosed syntax delimiters (truncated statements)
        if self._has_unclosed_delimiters(norm_quote, is_doc_prose=is_doc_prose):
            res = QuoteVerificationResult(
                is_valid=False,
                token_overlap_ratio=0.5,
                normalized_quote=norm_quote,
                error_code=SlicerErrorCode.TRUNCATED_QUOTE,
                error_reason="Quote is truncated mid-line with unclosed delimiters",
                unmatched_tokens=[],
            )
            if strict:
                raise TruncatedLineError(res.error_reason or "Truncated quote")
            return res

        # 1. Exact substring fast path
        if norm_quote in norm_source:
            return QuoteVerificationResult(
                is_valid=True,
                token_overlap_ratio=1.0,
                normalized_quote=norm_quote,
                matched_source_segment=norm_quote,
            )

        # 2. SequenceMatcher REMOVED (M4 Requirement 3: Zero Tolerance Substring)
        # Token substitution, omission, negation change, or homoglyph is strictly rejected.
        q_tokens = self.tokenize(norm_quote)
        s_tokens_set = set(self.tokenize(norm_source))

        if not q_tokens:
            raise QuoteVerificationError(
                SlicerErrorCode.QUOTE_MISMATCH, "No tokens extracted from quote"
            )

        absent_tokens = [t for t in q_tokens if t not in s_tokens_set]
        if absent_tokens:
            error_reason = (
                f"Quote failed verification: contains {len(absent_tokens)} hallucinated "
                f"or substituted tokens not matched in source text: {absent_tokens[:3]}"
            )
        else:
            error_reason = (
                "Quote failed verification: quote is not an exact contiguous substring of source "
                "text (tokens exist in source but sequence is altered or contains "
                "token omission / omitted tokens)"
            )

        if strict:
            raise HallucinatedQuoteError(error_reason)

        return QuoteVerificationResult(
            is_valid=False,
            token_overlap_ratio=0.0,
            normalized_quote=norm_quote,
            error_code=SlicerErrorCode.HALLUCINATED_QUOTE,
            error_reason=error_reason,
            unmatched_tokens=absent_tokens or q_tokens,
        )

    def verify_doc_citation(
        self,
        citation: DocCitationSpan,
        chunk_content: str,
        min_overlap_ratio: float = MIN_QUOTE_OVERLAP_RATIO,
        reject_absent_tokens: bool = True,
    ) -> QuoteVerificationResult:
        """Verifies a DocCitationSpan against verbatim chunk content."""
        return self.verify_quote(
            quote=citation.verbatim_quote,
            source_text=chunk_content,
            min_overlap_ratio=min_overlap_ratio,
            reject_absent_tokens=reject_absent_tokens,
            is_doc_prose=True,
        )


__all__ = [
    "DeterministicSlicer",
    "HallucinatedQuoteError",
    "InvalidLineBoundsError",
    "OffByOneBoundsError",
    "QuoteVerificationError",
    "QuoteVerificationResult",
    "TruncatedLineError",
]
