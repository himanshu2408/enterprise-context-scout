"""Polyglot Tree-sitter Side-Effect Extractor for Pillar 5 (M2-03)."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final
from uuid import NAMESPACE_DNS, UUID, uuid5

import tree_sitter_language_pack as tslp
import yaml
from tree_sitter import Node, Parser

from context_scout.ast_engine.extractor import ParseDiagnostics, _node_text, inspect_tree_errors
from context_scout.db.schema import (
    ASTSymbolRecord,
    CodeFileRecord,
    EffectCategory,
    SideEffectRecord,
    SideEffectSeverity,
)
from context_scout.schemas.contracts import DEFAULT_TENANT_ID

logger = logging.getLogger(__name__)

_PARSER_CACHE: dict[str, Parser] = {}

_LANG_TO_TSLP: Final[dict[str, str]] = {
    "python": "python",
    "py": "python",
    "typescript": "typescript",
    "ts": "typescript",
    "javascript": "javascript",
    "js": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "tsx": "tsx",
    "jsx": "tsx",
    "react_redux": "tsx",
    "go": "go",
    "golang": "go",
    "java": "java",
    "java_spring": "java",
    "kotlin": "kotlin",
    "kt": "kotlin",
    "kts": "kotlin",
    "csharp": "csharp",
    "cs": "csharp",
    "rust": "rust",
    "rs": "rust",
    "sql": "sql",
    "sql_dbt": "sql",
    "dbt": "sql",
    "terraform": "hcl",
    "hcl": "hcl",
    "tf": "hcl",
    "yaml": "yaml",
    "yml": "yaml",
    "api_specs": "yaml",
}

_EXT_TO_LANG: Final[dict[str, str]] = {
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".tsx": "tsx",
    ".jsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "csharp",
    ".rs": "rust",
    ".sql": "sql",
    ".tf": "hcl",
    ".hcl": "hcl",
    ".yaml": "yaml",
    ".yml": "yaml",
}

_COMMENT_NODE_TYPES: Final[frozenset[str]] = frozenset(
    {"comment", "line_comment", "block_comment", "multiline_comment"}
)

_CALL_NODE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "call",
        "call_expression",
        "method_invocation",
        "invocation_expression",
        "macro_invocation",
        "object_creation_expression",
        "new_expression",
    }
)

_ANNOTATION_NODE_TYPES: Final[frozenset[str]] = frozenset(
    {"decorator", "marker_annotation", "annotation", "attribute", "attribute_item"}
)

_ENCLOSING_DEF_TYPES: Final[frozenset[str]] = frozenset(
    {
        "function_definition",
        "class_definition",
        "function_declaration",
        "method_declaration",
        "method_definition",
        "class_declaration",
        "arrow_function",
        "function_item",
        "impl_item",
        "struct_item",
    }
)

_SQL_WRITE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:INSERT\s+(?:IGNORE\s+)?INTO|UPDATE|DELETE\s+FROM|MERGE\s+INTO|TRUNCATE\s+(?:TABLE\s+)?)"
    r"\s+([A-Za-z_][A-Za-z0-9_.\"`]*)",
    re.IGNORECASE,
)
_SQL_WRITE_INLINE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:INSERT\s+(?:IGNORE\s+)?INTO|UPDATE|DELETE\s+FROM|MERGE\s+INTO)\s+"
    r"([A-Za-z_][A-Za-z0-9_.\"`]*)",
    re.IGNORECASE,
)
_STRING_LIT_RE: Final[re.Pattern[str]] = re.compile(r"""['"]([^'"\r\n]{1,180})['"]""")
_TEMPLATE_ARG_RE: Final[re.Pattern[str]] = re.compile(
    r"""(?:template(?:_id|_name)?|mjml|hbs|view)\s*[:=]\s*['"]([^'"\r\n]+)['"]""",
    re.IGNORECASE,
)
_CRON_EXPR_RE: Final[re.Pattern[str]] = re.compile(
    r"""['"]([0-9*/,-]+\s+[0-9*/,-]+\s+[0-9*/,-]+\s+[0-9*/,-]+\s+[0-9*/,-A-Za-z?]+)['"]"""
)

_CATEGORY_TO_SEVERITY: Final[dict[EffectCategory, SideEffectSeverity]] = {
    "email": "external_comms",
    "sms": "external_comms",
    "push_notification": "external_comms",
    "webhook": "external_io",
    "pubsub_kafka_event": "async_messaging",
    "cron_job": "scheduled_job",
    "feature_flag": "runtime_toggle",
    "db_write": "state_mutation",
}


def _get_parser(lang_key: str) -> Parser | None:
    ts_name = _LANG_TO_TSLP.get(lang_key.lower())
    if ts_name is None:
        return None
    cached = _PARSER_CACHE.get(ts_name)
    if cached is not None:
        return cached
    try:
        parser = tslp.get_parser(ts_name)  # type: ignore[arg-type]
    except Exception as exc:
        logger.debug("Tree-sitter parser lookup failed for '%s': %s", ts_name, exc)
        return None
    _PARSER_CACHE[ts_name] = parser
    return parser


def _resolve_lang(language: str, file_path: str) -> str:
    low = language.strip().lower()
    if low in _LANG_TO_TSLP:
        return _LANG_TO_TSLP[low]
    lower_path = file_path.lower()
    for ext, mapped in _EXT_TO_LANG.items():
        if lower_path.endswith(ext):
            return mapped
    return low


def _is_docstring_expr_stmt(node: Node, parent: Node | None) -> bool:
    if node.type != "expression_statement":
        return False
    if parent is None or parent.type not in {"module", "block", "class_body"}:
        return False
    named = [c for c in node.children if c.is_named]
    return len(named) == 1 and named[0].type in {"string", "string_literal"}


def _extract_first_string_arg(text: str) -> str | None:
    paren_idx = text.find("(")
    sub = text[paren_idx:] if paren_idx != -1 else text
    m = _STRING_LIT_RE.search(sub)
    return m.group(1).strip() if m else None


def _extract_named_or_first_arg(text: str, *param_names: str) -> str | None:
    paren_idx = text.find("(")
    sub = text[paren_idx:] if paren_idx != -1 else text
    for pname in param_names:
        m = re.search(
            rf"""\b{re.escape(pname)}\s*[:=]\s*['"]([^'"\r\n]+)['"]""",
            sub,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
    return _extract_first_string_arg(sub)


def _extract_callee(node: Node, code_bytes: bytes) -> str:
    for field in ("function", "name"):
        child = node.child_by_field_name(field)
        if child is not None:
            return _node_text(child, code_bytes).strip()
    if node.type == "method_invocation":
        obj = node.child_by_field_name("object")
        name = node.child_by_field_name("name")
        if obj is not None and name is not None:
            return f"{_node_text(obj, code_bytes).strip()}.{_node_text(name, code_bytes).strip()}"
        if name is not None:
            return _node_text(name, code_bytes).strip()
    if node.type == "macro_invocation":
        macro = node.child_by_field_name("macro")
        if macro is not None:
            return _node_text(macro, code_bytes).strip()
    txt = _node_text(node, code_bytes).strip()
    paren = txt.find("(")
    return txt[:paren].strip() if paren > 0 else txt[:120].strip()


def _extract_enclosing_name(node: Node, code_bytes: bytes) -> str:
    curr = node.parent
    parts: list[str] = []
    while curr is not None:
        if curr.type in _ENCLOSING_DEF_TYPES:
            name_node = curr.child_by_field_name("name")
            if name_node is None:
                for child in curr.children:
                    if child.type in {"identifier", "type_identifier"}:
                        name_node = child
                        break
            if name_node is not None:
                ident = _node_text(name_node, code_bytes).strip()
                if ident:
                    parts.append(ident)
        curr = curr.parent
    if not parts:
        return "<module>"
    return ".".join(reversed(parts))


def _match_symbol(
    line_num: int,
    fallback_qname: str,
    symbols: Sequence[ASTSymbolRecord],
) -> tuple[UUID | None, str]:
    best: ASTSymbolRecord | None = None
    best_span = 1_000_000_000
    for sym in symbols:
        if sym.start_line <= line_num <= sym.end_line:
            span = sym.end_line - sym.start_line
            if span < best_span:
                best = sym
                best_span = span
    if best is not None:
        return best.symbol_id, best.qualified_name
    return None, fallback_qname


def _classify_call_side_effect(
    callee: str,
    call_text: str,
) -> tuple[EffectCategory, str, str, str | None, str] | None:
    """Classifies a call expression into (category, target, driver, template_ref, trigger)."""
    cal_low = callee.lower()
    txt_low = call_text.lower()

    # 1. Email
    if any(
        tok in cal_low
        for tok in (
            "sendmail",
            "send_mail",
            "send_email",
            "sendemail",
            "transporter.sendmail",
            "javamailsender",
            "mailsender.send",
            "emails.send",
            "sendgrid",
            "ses.send_email",
            "ses_client.send_email",
            "mailgun",
        )
    ) or ("sendgrid" in txt_low and ".send(" in txt_low):
        target = (
            _extract_named_or_first_arg(
                call_text, "to", "recipient", "to_email", "template_id", "subject"
            )
            or callee
        )
        tpl = None
        m_tpl = _TEMPLATE_ARG_RE.search(call_text)
        if m_tpl:
            tpl = m_tpl.group(1).strip()
        driver = (
            "sendgrid"
            if "sendgrid" in txt_low
            else "aws_ses"
            if "ses" in cal_low
            else "nodemailer"
            if "transporter" in cal_low or "nodemailer" in txt_low
            else "javamail"
            if "javamail" in cal_low or "mailsender" in cal_low
            else "resend"
            if "resend" in cal_low
            else "smtp"
        )
        return ("email", target, driver, tpl, callee)

    # 2. SMS
    if any(
        tok in cal_low
        for tok in (
            "send_sms",
            "sendsms",
            "twilio",
            "messages.create",
            "vonage.sms",
            "plivo",
        )
    ) or ("sns" in cal_low and "publish" in cal_low and "phonenumber" in txt_low):
        target = (
            _extract_named_or_first_arg(call_text, "to", "phone_number", "PhoneNumber") or callee
        )
        driver = (
            "twilio"
            if "twilio" in txt_low or "messages.create" in cal_low
            else "aws_sns_sms"
            if "sns" in cal_low
            else "sms_client"
        )
        return ("sms", target, driver, None, callee)

    # 3. Push Notification
    if any(
        tok in cal_low
        for tok in (
            "send_push",
            "sendpush",
            "fcm",
            "messaging.send",
            "send_multicast",
            "send_each_for_multicast",
            "apns",
            "send_push_notifications",
            "onesignal",
            "webpush",
        )
    ):
        target = _extract_named_or_first_arg(call_text, "topic", "token", "device_token") or callee
        driver = (
            "firebase_fcm"
            if "fcm" in txt_low or "messaging." in cal_low or "multicast" in cal_low
            else "apns"
            if "apns" in txt_low
            else "expo_push"
            if "expo" in txt_low or "send_push_notifications" in cal_low
            else "push_gateway"
        )
        return ("push_notification", target, driver, None, callee)

    # 4. Outbound Webhook
    if any(
        tok in cal_low
        for tok in (
            "send_webhook",
            "dispatch_webhook",
            "trigger_webhook",
            "deliver_webhook",
            "post_webhook",
            "notify_slack",
            "slack_webhook",
        )
    ) or (
        any(
            tok in cal_low
            for tok in (
                ".post",
                "postforentity",
                "postforobject",
                "fetch",
                "axios",
                "webclient",
                "resttemplate",
            )
        )
        and any(
            marker in txt_low
            for marker in (
                "webhook",
                "/webhooks",
                "hooks.",
                "/hooks/",
                "hooks.slack.com",
                "discord.com/api/webhooks",
                "events.pagerduty.com",
                "x-hub-signature",
                "x-webhook-signature",
                "callback_url",
            )
        )
    ):
        target = (
            _extract_named_or_first_arg(call_text, "url", "webhook_url", "callback_url") or callee
        )
        driver = "slack_webhook" if "slack" in txt_low else "http_webhook"
        return ("webhook", target, driver, None, callee)

    # 5. Pub/Sub / Kafka / Event Bus
    if any(
        tok in cal_low
        for tok in (
            "publisher.publish",
            "pubsub.publish",
            "kafkatemplate.send",
            "producer.send",
            "producer.produce",
            "send_message",
            "basic_publish",
            "publish_event",
            "emit_event",
            "nats.publish",
            "put_events",
            "eventbus.publish",
        )
    ) or ("sns" in cal_low and "publish" in cal_low):
        target = (
            _extract_named_or_first_arg(
                call_text,
                "topic",
                "topic_path",
                "TopicArn",
                "QueueUrl",
                "routing_key",
                "channel",
                "subject",
                "event_name",
            )
            or callee
        )
        driver = (
            "gcp_pubsub"
            if "pubsub" in txt_low or "topic_path" in txt_low
            else "kafka"
            if "kafka" in txt_low or "produce" in cal_low
            else "aws_sqs"
            if "sqs" in txt_low or "queueurl" in txt_low
            else "aws_sns"
            if "sns" in txt_low or "topicarn" in txt_low
            else "rabbitmq"
            if "basic_publish" in cal_low
            else "event_bus"
        )
        return ("pubsub_kafka_event", target, driver, None, callee)

    # 6. Cron / Scheduled Jobs (imperative registration)
    if any(
        tok in cal_low
        for tok in (
            "add_periodic_task",
            "scheduler.add_job",
            "cron.schedule",
            "cron.addfunc",
            "recurringjob.addorupdate",
            "recurringjob.trigger",
        )
    ) or cal_low.endswith(".crontab"):
        m_cron = _CRON_EXPR_RE.search(call_text)
        target = m_cron.group(1) if m_cron else (_extract_first_string_arg(call_text) or callee)
        driver = (
            "celery_beat"
            if "periodic_task" in cal_low or "crontab" in cal_low
            else "apscheduler"
            if "add_job" in cal_low
            else "hangfire"
            if "recurringjob" in cal_low
            else "cron"
        )
        return ("cron_job", target, driver, None, callee)

    # 7. Feature Flags
    if any(
        tok in cal_low
        for tok in (
            "ldclient.variation",
            "boolvariation",
            "stringvariation",
            "is_feature_enabled",
            "isfeatureenabled",
            "check_gate",
            "has_feature",
            "get_boolean_value",
            "getbooleanvalue",
            "unleash.is_enabled",
            "feature_flag",
        )
    ) or ("flagsmith" in txt_low and "feature" in cal_low):
        target = _extract_named_or_first_arg(call_text, "flag_key", "key", "feature") or callee
        driver = (
            "launchdarkly"
            if "ldclient" in cal_low or "variation" in cal_low
            else "unleash"
            if "unleash" in txt_low
            else "openfeature"
            if "booleanvalue" in cal_low or "boolean_value" in cal_low
            else "statsig"
            if "check_gate" in cal_low
            else "posthog"
            if "posthog" in txt_low
            else "feature_flag"
        )
        return ("feature_flag", target, driver, None, callee)

    # 8. Database Writes (ORM or parameterized SQL DML inside execute/query)
    if any(
        tok in cal_low
        for tok in (
            ".execute",
            ".executemany",
            ".exec",
            ".query",
            ".raw",
            "executeupdate",
            "executesql",
        )
    ):
        for m_str in _STRING_LIT_RE.finditer(call_text):
            sql_cand = m_str.group(1)
            m_dml = _SQL_WRITE_INLINE_RE.search(sql_cand)
            if m_dml:
                tbl = m_dml.group(1).strip("`\"'")
                return ("db_write", tbl, "sql_dml", None, callee)

    if any(
        tok in cal_low
        for tok in (
            "session.add",
            "session.delete",
            "session.merge",
            "bulk_save_objects",
            "bulk_create",
            "bulk_update",
            ".objects.create",
            ".objects.update",
            ".save",
            "repository.save",
            "repo.save",
            "gorm.create",
            "db.create",
            "db.save",
        )
    ) or re.search(r"\bprisma\.[A-Za-z0-9_]+\.(?:create|update|upsert|delete)\b", cal_low):
        target = _extract_first_string_arg(call_text) or callee
        m_prisma = re.search(
            r"\bprisma\.([A-Za-z0-9_]+)\.(?:create|update|upsert|delete)\b", cal_low
        )
        if m_prisma:
            target = m_prisma.group(1)
        driver = (
            "prisma"
            if m_prisma
            else "sqlalchemy"
            if "session." in cal_low
            else "django_orm"
            if ".objects." in cal_low
            else "orm_write"
        )
        return ("db_write", target, driver, None, callee)

    return None


def _classify_annotation_side_effect(
    ann_text: str,
) -> tuple[EffectCategory, str, str, str | None, str] | None:
    low = ann_text.lower()
    if any(
        tok in low
        for tok in ("@scheduled", "@periodic_task", "scheduled(", "[recurringjob", "cron =")
    ):
        m_cron = _CRON_EXPR_RE.search(ann_text)
        target = (
            m_cron.group(1) if m_cron else (_extract_first_string_arg(ann_text) or ann_text[:64])
        )
        driver = "spring_scheduled" if "@scheduled" in low else "celery_beat"
        return ("cron_job", target, driver, None, ann_text.splitlines()[0].strip()[:120])
    if any(
        tok in low
        for tok in ("@kafkalistener", "@pubsublistener", "@rabbitlistener", "@sqslistener")
    ):
        target = _extract_named_or_first_arg(ann_text, "topics", "topic", "queues") or ann_text[:64]
        driver = "kafka" if "kafka" in low else "pubsub" if "pubsub" in low else "mq_listener"
        return (
            "pubsub_kafka_event",
            target,
            driver,
            None,
            ann_text.splitlines()[0].strip()[:120],
        )
    return None


class SideEffectExtractor:
    """Tree-sitter AST-backed extractor cataloging Pillar 5 side effects with fail-closed bounds."""

    def __init__(self) -> None:
        self.last_parse_diagnostics: ParseDiagnostics | None = None

    def extract_from_code_file(
        self,
        code_file: CodeFileRecord,
        symbols: Sequence[ASTSymbolRecord] = (),
    ) -> list[SideEffectRecord]:
        return self.extract_from_source(
            code=code_file.raw_content,
            language=code_file.language,
            file_path=code_file.file_path,
            domain_id=code_file.domain_id,
            repo_name=code_file.repo_name,
            allowed_scopes=code_file.allowed_scopes,
            file_id=code_file.file_id,
            symbols=symbols,
            tenant_id=code_file.tenant_id,
        )

    def extract_from_source(
        self,
        *,
        code: str,
        language: str,
        file_path: str,
        domain_id: str,
        repo_name: str,
        allowed_scopes: Sequence[str],
        file_id: UUID | None = None,
        symbols: Sequence[ASTSymbolRecord] = (),
        tenant_id: UUID = DEFAULT_TENANT_ID,
    ) -> list[SideEffectRecord]:
        """Extracts side-effect records cleanly; abstains with [] on Tree-sitter syntax errors."""
        self.last_parse_diagnostics = None
        if not code.strip():
            return []

        norm_lang = _resolve_lang(language, file_path)
        parser = _get_parser(norm_lang)
        if parser is None:
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="unsupported",
                abstained=True,
                reason=f"No Tree-sitter grammar registered for language '{language}'",
            )
            return []

        code_bytes = code.encode("utf-8", errors="replace")
        tree = parser.parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            reason = (
                f"Syntax error in '{file_path or '<memory>'}' ({norm_lang}): "
                f"error_nodes={err_cnt}, missing_nodes={miss_cnt}"
            )
            logger.warning("%s; abstaining from side-effect extraction", reason)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend=f"tree-sitter-{norm_lang}",
                has_parse_error=True,
                abstained=True,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason=reason,
            )
            return []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend=f"tree-sitter-{norm_lang}")

        scopes_list = [s for s in allowed_scopes if s and s.strip()] or [f"repo:{repo_name}:read"]
        now = datetime.now(UTC)
        results: list[SideEffectRecord] = []
        seen_keys: set[tuple[str, str, int, str]] = set()

        def _emit(
            *,
            category: EffectCategory,
            target: str,
            driver: str,
            start_line: int,
            end_line: int,
            qualified_symbol: str,
            symbol_id: UUID | None,
            template_ref: str | None = None,
            trigger: str = "",
            meta: dict[str, Any] | None = None,
        ) -> None:
            clean_target = target.strip() or driver or category
            dedupe = (category, clean_target, start_line, qualified_symbol)
            if dedupe in seen_keys:
                return
            seen_keys.add(dedupe)
            se_id = uuid5(
                NAMESPACE_DNS,
                f"{tenant_id}:{domain_id}:{repo_name}:{file_path}:{qualified_symbol}:{category}:{clean_target}:{start_line}",
            )
            results.append(
                SideEffectRecord(
                    side_effect_id=se_id,
                    tenant_id=tenant_id,
                    domain_id=domain_id,
                    repo_name=repo_name,
                    file_id=file_id,
                    symbol_id=symbol_id,
                    file_path=file_path,
                    qualified_symbol=qualified_symbol,
                    start_line=max(1, start_line),
                    end_line=max(max(1, start_line), end_line),
                    effect_category=category,
                    target_identifier=clean_target,
                    provider_or_driver=driver,
                    trigger_condition=trigger,
                    template_or_schema_ref=template_ref,
                    severity_tier=_CATEGORY_TO_SEVERITY[category],
                    confidence=1.0,
                    metadata_json=meta or {},
                    allowed_scopes=list(scopes_list),
                    indexed_at=now,
                )
            )

        if norm_lang == "yaml":
            self._extract_yaml_side_effects(
                code=code,
                file_path=file_path,
                symbols=symbols,
                emit=_emit,
            )
            return results

        if norm_lang == "hcl":
            self._extract_hcl_side_effects(
                root=tree.root_node,
                code_bytes=code_bytes,
                symbols=symbols,
                emit=_emit,
            )
            return results

        if norm_lang == "sql":
            self._extract_sql_side_effects(
                root=tree.root_node,
                code_bytes=code_bytes,
                symbols=symbols,
                emit=_emit,
            )
            return results

        # General AST traversal across Python, TS/JS/TSX, Go, Java, Kotlin, C#, Rust
        stack: list[Node] = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type in _COMMENT_NODE_TYPES or _is_docstring_expr_stmt(node, node.parent):
                continue

            if node.type in _ANNOTATION_NODE_TYPES:
                ann_txt = _node_text(node, code_bytes).strip()
                classified = _classify_annotation_side_effect(ann_txt)
                if classified is not None:
                    cat, tgt, drv, tpl, trig = classified
                    s_ln = node.start_point[0] + 1
                    e_ln = node.end_point[0] + 1
                    fallback_q = _extract_enclosing_name(node, code_bytes)
                    sym_id, qsym = _match_symbol(s_ln, fallback_q, symbols)
                    _emit(
                        category=cat,
                        target=tgt,
                        driver=drv,
                        start_line=s_ln,
                        end_line=e_ln,
                        qualified_symbol=qsym,
                        symbol_id=sym_id,
                        template_ref=tpl,
                        trigger=trig,
                    )

            if node.type in _CALL_NODE_TYPES:
                callee = _extract_callee(node, code_bytes)
                call_text = _node_text(node, code_bytes)
                classified = _classify_call_side_effect(callee, call_text)
                if classified is not None:
                    cat, tgt, drv, tpl, trig = classified
                    s_ln = node.start_point[0] + 1
                    e_ln = node.end_point[0] + 1
                    fallback_q = _extract_enclosing_name(node, code_bytes)
                    sym_id, qsym = _match_symbol(s_ln, fallback_q, symbols)
                    _emit(
                        category=cat,
                        target=tgt,
                        driver=drv,
                        start_line=s_ln,
                        end_line=e_ln,
                        qualified_symbol=qsym,
                        symbol_id=sym_id,
                        template_ref=tpl,
                        trigger=trig,
                    )

            stack.extend(reversed(node.children))

        return results

    def _extract_sql_side_effects(
        self,
        *,
        root: Node,
        code_bytes: bytes,
        symbols: Sequence[ASTSymbolRecord],
        emit: Any,
    ) -> None:
        stack: list[Node] = [root]
        while stack:
            node = stack.pop()
            if node.type in _COMMENT_NODE_TYPES:
                continue
            if node.type in {"statement", "insert", "update", "delete"}:
                stmt_text = _node_text(node, code_bytes).strip()
                m = _SQL_WRITE_RE.match(stmt_text)
                if m:
                    tbl = m.group(1).strip("`\"'")
                    s_ln = node.start_point[0] + 1
                    e_ln = node.end_point[0] + 1
                    sym_id, qsym = _match_symbol(s_ln, f"sql.{tbl}", symbols)
                    emit(
                        category="db_write",
                        target=tbl,
                        driver="sql_statement",
                        start_line=s_ln,
                        end_line=e_ln,
                        qualified_symbol=qsym,
                        symbol_id=sym_id,
                        trigger=stmt_text.splitlines()[0][:120],
                    )
                    continue
            stack.extend(reversed(node.children))

    def _extract_hcl_side_effects(
        self,
        *,
        root: Node,
        code_bytes: bytes,
        symbols: Sequence[ASTSymbolRecord],
        emit: Any,
    ) -> None:
        top_body = next((c for c in root.children if c.type == "body"), None)
        if top_body is None:
            return
        for block in top_body.children:
            if block.type != "block":
                continue
            b_txt = _node_text(block, code_bytes)
            m_hdr = re.match(
                r'^\s*resource\s+"([^"]+)"\s+"([^"]+)"',
                b_txt,
            )
            if not m_hdr:
                continue
            res_type, res_name = m_hdr.group(1), m_hdr.group(2)
            s_ln = block.start_point[0] + 1
            e_ln = block.end_point[0] + 1
            sym_id, qsym = _match_symbol(s_ln, f"{res_type}.{res_name}", symbols)
            if res_type in {
                "google_pubsub_topic",
                "aws_sns_topic",
                "aws_sqs_queue",
                "confluent_kafka_topic",
            }:
                m_name = re.search(r'name\s*=\s*"([^"]+)"', b_txt)
                target = m_name.group(1) if m_name else res_name
                emit(
                    category="pubsub_kafka_event",
                    target=target,
                    driver=res_type,
                    start_line=s_ln,
                    end_line=e_ln,
                    qualified_symbol=qsym,
                    symbol_id=sym_id,
                    trigger=f"terraform resource {res_type}.{res_name}",
                )
            elif res_type in {"google_cloud_scheduler_job", "aws_cloudwatch_event_rule"}:
                m_sched = re.search(r'(?:schedule|schedule_expression)\s*=\s*"([^"]+)"', b_txt)
                target = m_sched.group(1) if m_sched else res_name
                emit(
                    category="cron_job",
                    target=target,
                    driver=res_type,
                    start_line=s_ln,
                    end_line=e_ln,
                    qualified_symbol=qsym,
                    symbol_id=sym_id,
                    trigger=f"terraform resource {res_type}.{res_name}",
                )
            elif res_type.startswith("launchdarkly_"):
                m_key = re.search(r'key\s*=\s*"([^"]+)"', b_txt)
                target = m_key.group(1) if m_key else res_name
                emit(
                    category="feature_flag",
                    target=target,
                    driver=res_type,
                    start_line=s_ln,
                    end_line=e_ln,
                    qualified_symbol=qsym,
                    symbol_id=sym_id,
                    trigger=f"terraform resource {res_type}.{res_name}",
                )

    def _extract_yaml_side_effects(
        self,
        *,
        code: str,
        file_path: str,
        symbols: Sequence[ASTSymbolRecord],
        emit: Any,
    ) -> None:
        try:
            docs = list(yaml.safe_load_all(code))
        except yaml.YAMLError as exc:
            logger.debug("YAML parse failed for '%s': %s", file_path, exc)
            return

        total_lines = max(1, len(code.splitlines()))
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            kind = str(doc.get("kind") or "").strip().lower()
            spec = doc.get("spec")
            meta = doc.get("metadata")
            name = (
                str(meta.get("name")) if isinstance(meta, dict) and meta.get("name") else file_path
            )
            if kind == "cronjob" and isinstance(spec, dict) and spec.get("schedule"):
                sched = str(spec["schedule"])
                line_num = self._find_line(code, sched)
                sym_id, qsym = _match_symbol(line_num, f"cronjob.{name}", symbols)
                emit(
                    category="cron_job",
                    target=sched,
                    driver="k8s_cronjob",
                    start_line=line_num,
                    end_line=min(total_lines, line_num + 2),
                    qualified_symbol=qsym,
                    symbol_id=sym_id,
                    trigger=f"k8s CronJob {name}",
                )
            on_block = doc.get("on") if "on" in doc else doc.get(True)
            if isinstance(on_block, dict) and isinstance(on_block.get("schedule"), list):
                for entry in on_block["schedule"]:
                    if isinstance(entry, dict) and entry.get("cron"):
                        cron_expr = str(entry["cron"])
                        line_num = self._find_line(code, cron_expr)
                        sym_id, qsym = _match_symbol(line_num, f"workflow.{name}", symbols)
                        emit(
                            category="cron_job",
                            target=cron_expr,
                            driver="github_actions_schedule",
                            start_line=line_num,
                            end_line=line_num,
                            qualified_symbol=qsym,
                            symbol_id=sym_id,
                            trigger=f"workflow schedule {name}",
                        )

    @staticmethod
    def _find_line(code: str, needle: str) -> int:
        for idx, line in enumerate(code.splitlines(), start=1):
            if needle in line:
                return idx
        return 1


__all__ = ["SideEffectExtractor"]
