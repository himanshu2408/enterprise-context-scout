"""Gitattributes down-weighting and CODEOWNERS / Backstage ownership graph extraction."""

from __future__ import annotations

import logging
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import yaml

from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol

logger = logging.getLogger(__name__)

WEIGHT_NORMAL: Final[float] = 1.0
WEIGHT_VENDORED: Final[float] = 0.25
WEIGHT_GENERATED: Final[float] = 0.15
WEIGHT_GENERATED_AND_VENDORED: Final[float] = 0.10

_GENERATED_ATTRS: Final[frozenset[str]] = frozenset(
    {"linguist-generated", "gitlab-generated", "generated"}
)
_VENDORED_ATTRS: Final[frozenset[str]] = frozenset(
    {"linguist-vendored", "gitlab-vendored", "vendored"}
)

CODEOWNERS_FILENAMES: Final[tuple[str, ...]] = (
    ".github/CODEOWNERS",
    ".gitlab/CODEOWNERS",
    "CODEOWNERS",
    "docs/CODEOWNERS",
)
CATALOG_INFO_FILENAMES: Final[frozenset[str]] = frozenset({"catalog-info.yaml", "catalog-info.yml"})
_BACKSTAGE_KINDS: Final[frozenset[str]] = frozenset(
    {"Component", "System", "API", "Resource", "Domain", "Group", "Location"}
)
_SECTION_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"^\^?\[([^\[\]]+)\](?:\[(\d+)\])?(.*)$")


def normalize_rel_path(path: str) -> str:
    """Normalizes repository-relative POSIX paths while preserving dotfiles."""
    cleaned = path.replace("\\", "/").strip()
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lstrip("/")


_normalize_rel_path = normalize_rel_path


def _compile_git_glob(
    pattern: str, *, dir_prefix: str = "", for_codeowners: bool = False
) -> re.Pattern[str]:
    """Compiles a .gitattributes / CODEOWNERS glob pattern into an anchored O(n) regex."""
    raw = pattern.strip()
    is_dir_only = raw.endswith("/")
    if is_dir_only:
        raw = raw.rstrip("/")

    anchored = raw.startswith("/") or ("/" in raw)
    if raw.startswith("/"):
        raw = raw.lstrip("/")

    regex_parts: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch == "\\":
            if i + 1 < n:
                regex_parts.append(re.escape(raw[i + 1]))
                i += 2
            else:
                regex_parts.append(re.escape("\\"))
                i += 1
            continue

        if ch == "*":
            if i + 1 < n and raw[i + 1] == "*":
                # Handle /**/ (zero or more directory segments) and leading **/
                if i + 2 < n and raw[i + 2] == "/":
                    regex_parts.append("(?:.*/)?")
                    i += 3
                else:
                    regex_parts.append(".*")
                    i += 2
            else:
                regex_parts.append("[^/]*")
                i += 1
            continue

        if ch == "?":
            regex_parts.append("[^/]")
            i += 1
            continue

        if ch == "[":
            end_idx = raw.find("]", i + 1)
            if end_idx != -1:
                body = raw[i + 1 : end_idx]
                if body.startswith("!"):
                    body = "^" + body[1:]
                regex_parts.append(f"[{body}]")
                i = end_idx + 1
                continue

        regex_parts.append(re.escape(ch))
        i += 1

    core = "".join(regex_parts)
    prefix_re = re.escape(dir_prefix.rstrip("/") + "/") if dir_prefix else ""

    if is_dir_only:
        expr = f"^{prefix_re}{core}/.+$" if anchored else f"^{prefix_re}(?:.*/)?{core}/.+$"
    elif anchored:
        # In CODEOWNERS, `/docs` can also match files under `/docs/...` if directory-shaped.
        expr = f"^{prefix_re}{core}(?:/.*)?$" if for_codeowners else f"^{prefix_re}{core}$"
    else:
        expr = (
            f"^{prefix_re}(?:.*/)?{core}(?:/.*)?$"
            if for_codeowners
            else f"^{prefix_re}(?:.*/)?{core}$"
        )
    return re.compile(expr)


@dataclass(frozen=True)
class FileWeightClassification:
    """Result of .gitattributes generated/vendored classification for a file path."""

    file_path: str
    ranking_weight: float
    is_generated: bool
    is_vendored: bool


@dataclass(frozen=True)
class _GitAttrRule:
    pattern: str
    compiled: re.Pattern[str]
    generated: bool | None
    vendored: bool | None
    source_dir: str


def _parse_attr_tokens(line: str) -> tuple[str, list[str]] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        tokens = stripped.split()
    if len(tokens) < 2:
        return None
    return tokens[0], tokens[1:]


class GitAttributesMatcher:
    """Parses root and nested .gitattributes rules to compute retrieval ranking weights."""

    def __init__(self, rules: Sequence[_GitAttrRule] = ()) -> None:
        self._rules = tuple(rules)

    @classmethod
    def from_repo_files(cls, files: Mapping[str, str]) -> GitAttributesMatcher:
        attr_paths: list[str] = []
        for raw_path in files:
            norm = _normalize_rel_path(raw_path)
            if norm == ".gitattributes" or norm.endswith("/.gitattributes"):
                attr_paths.append(norm)

        # Sort shallow-to-deep so nested .gitattributes override root rules cleanly.
        attr_paths.sort(key=lambda p: (p.count("/"), p))
        rules: list[_GitAttrRule] = []
        for attr_path in attr_paths:
            source_dir = attr_path.rsplit("/", 1)[0] if "/" in attr_path else ""
            raw_content = files.get(attr_path)
            if raw_content is None:
                # Match original key if unnormalized
                for orig_k, orig_v in files.items():
                    if _normalize_rel_path(orig_k) == attr_path:
                        raw_content = orig_v
                        break
            if not raw_content:
                continue
            for line in raw_content.splitlines():
                parsed = _parse_attr_tokens(line)
                if parsed is None:
                    continue
                glob_pat, attr_tokens = parsed
                gen_val: bool | None = None
                vend_val: bool | None = None
                for tok in attr_tokens:
                    name, state = _interpret_attr_token(tok)
                    if name in _GENERATED_ATTRS:
                        gen_val = state
                    elif name in _VENDORED_ATTRS:
                        vend_val = state
                if gen_val is None and vend_val is None:
                    continue
                compiled = _compile_git_glob(glob_pat, dir_prefix=source_dir, for_codeowners=False)
                rules.append(
                    _GitAttrRule(
                        pattern=glob_pat,
                        compiled=compiled,
                        generated=gen_val,
                        vendored=vend_val,
                        source_dir=source_dir,
                    )
                )
        return cls(rules)

    @property
    def has_rules(self) -> bool:
        return bool(self._rules)

    def classify_file(self, file_path: str) -> FileWeightClassification:
        norm = _normalize_rel_path(file_path)
        is_generated = False
        is_vendored = False
        for rule in self._rules:
            if rule.source_dir and not norm.startswith(f"{rule.source_dir}/"):
                continue
            if rule.compiled.match(norm):
                if rule.generated is not None:
                    is_generated = rule.generated
                if rule.vendored is not None:
                    is_vendored = rule.vendored

        if is_generated and is_vendored:
            weight = WEIGHT_GENERATED_AND_VENDORED
        elif is_generated:
            weight = WEIGHT_GENERATED
        elif is_vendored:
            weight = WEIGHT_VENDORED
        else:
            weight = WEIGHT_NORMAL

        return FileWeightClassification(
            file_path=norm,
            ranking_weight=weight,
            is_generated=is_generated,
            is_vendored=is_vendored,
        )


def _interpret_attr_token(token: str) -> tuple[str, bool | None]:
    tok = token.strip()
    if tok.startswith("-"):
        return tok[1:].lower(), False
    if tok.startswith("!"):
        return tok[1:].lower(), False
    if "=" in tok:
        key, val = tok.split("=", 1)
        val_l = val.strip().lower()
        if val_l in {"true", "1", "yes"}:
            return key.strip().lower(), True
        if val_l in {"false", "0", "no"}:
            return key.strip().lower(), False
        return key.strip().lower(), None
    return tok.lower(), True


def normalize_owner_id(raw_owner: str) -> tuple[str, str, str | None]:
    """Normalizes a CODEOWNERS or Backstage owner into (canonical_id, slug, org_opt)."""
    cleaned = raw_owner.strip()
    if not cleaned:
        raise ValueError("Owner identifier cannot be empty.")

    if cleaned.startswith("@"):
        handle = cleaned[1:].strip()
        if "/" in handle:
            org, slug = handle.split("/", 1)
            org_l = org.strip().lower()
            slug_l = slug.strip().lower()
            return f"team:{org_l}/{slug_l}", slug_l, org_l
        slug_l = handle.lower()
        return f"team:{slug_l}", slug_l, None

    if ":" in cleaned and "/" in cleaned:
        kind_part, rest = cleaned.split(":", 1)
        kind_l = kind_part.strip().lower()
        _ns, name_part = rest.split("/", 1)
        name_l = name_part.strip().lower()
        if kind_l == "user":
            return f"user:{name_l}", name_l, None
        return f"team:{name_l}", name_l, None

    if ":" in cleaned:
        kind_part, name_part = cleaned.split(":", 1)
        kind_l = kind_part.strip().lower()
        name_l = name_part.strip().lower()
        if kind_l == "user":
            return f"user:{name_l}", name_l, None
        return f"team:{name_l}", name_l, None

    if "@" in cleaned:
        email_l = cleaned.lower()
        return f"owner:{email_l}", email_l, None

    slug_l = cleaned.lower()
    return f"team:{slug_l}", slug_l, None


@dataclass(frozen=True)
class CodeOwnersRule:
    """Single parsed CODEOWNERS matching rule."""

    pattern: str
    owners: tuple[str, ...]
    section: str = ""
    line_number: int = 1


def _split_codeowners_line(line: str) -> list[str]:
    """Splits a CODEOWNERS line while respecting backslash-escaped spaces in paths."""
    tokens: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch == "\\" and i + 1 < n and line[i + 1] == " ":
            buf.append(" ")
            i += 2
            continue
        if ch.isspace():
            if buf:
                tokens.append("".join(buf))
                buf.clear()
            while i < n and line[i].isspace():
                i += 1
            continue
        if ch == "#" and not buf:
            break
        buf.append(ch)
        i += 1
    if buf:
        tokens.append("".join(buf))
    return tokens


class CodeOwnersMatcher:
    """Matches repository paths against GitHub and GitLab CODEOWNERS rules."""

    def __init__(self, rules: Sequence[tuple[CodeOwnersRule, re.Pattern[str]]] = ()) -> None:
        self._rules = tuple(rules)

    @classmethod
    def parse(cls, content: str) -> CodeOwnersMatcher:
        parsed_rules: list[tuple[CodeOwnersRule, re.Pattern[str]]] = []
        current_section = ""
        section_default_owners: tuple[str, ...] = ()

        for lineno, raw_line in enumerate(content.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            sec_match = _SECTION_HEADER_RE.match(stripped)
            if sec_match:
                current_section = sec_match.group(1).strip().lower()
                trailing = sec_match.group(3).strip()
                if trailing:
                    sec_tokens = [t for t in _split_codeowners_line(trailing) if t]
                    section_default_owners = tuple(
                        normalize_owner_id(tok)[0] for tok in sec_tokens if not tok.startswith("#")
                    )
                else:
                    section_default_owners = ()
                continue

            tokens = _split_codeowners_line(stripped)
            if not tokens:
                continue
            pattern = tokens[0]
            raw_owners = [t for t in tokens[1:] if t and not t.startswith("#")]
            if raw_owners:
                owners = tuple(normalize_owner_id(tok)[0] for tok in raw_owners)
            elif current_section and section_default_owners:
                owners = section_default_owners
            else:
                # Explicit empty owners unassign prior matches for this pattern.
                owners = ()

            compiled = _compile_git_glob(pattern, dir_prefix="", for_codeowners=True)
            parsed_rules.append(
                (
                    CodeOwnersRule(
                        pattern=pattern,
                        owners=owners,
                        section=current_section,
                        line_number=lineno,
                    ),
                    compiled,
                )
            )
        return cls(parsed_rules)

    @classmethod
    def from_repo_files(cls, files: Mapping[str, str]) -> CodeOwnersMatcher:
        normalized_map = {_normalize_rel_path(k): v for k, v in files.items()}
        for candidate in CODEOWNERS_FILENAMES:
            if candidate in normalized_map:
                return cls.parse(normalized_map[candidate])
        return cls()

    @property
    def has_rules(self) -> bool:
        return bool(self._rules)

    def match_owners(self, file_path: str) -> tuple[str, ...]:
        """Returns matching normalized owner IDs (last rule wins per GitLab/GitHub section)."""
        norm = _normalize_rel_path(file_path)
        owners_by_section: dict[str, tuple[str, ...]] = {}
        section_order: list[str] = []

        for rule, compiled in self._rules:
            if compiled.match(norm):
                if rule.section not in owners_by_section:
                    section_order.append(rule.section)
                owners_by_section[rule.section] = rule.owners

        merged: list[str] = []
        seen: set[str] = set()
        for sec in section_order:
            for owner in owners_by_section.get(sec, ()):
                if owner not in seen:
                    seen.add(owner)
                    merged.append(owner)
        return tuple(merged)


@dataclass(frozen=True)
class BackstageEntity:
    """Parsed Backstage catalog-info.yaml entity."""

    kind: str
    name: str
    namespace: str
    base_dir: str
    source_path: str
    owner: str | None = None
    system: str | None = None
    domain: str | None = None
    subcomponent_of: str | None = None
    provides_apis: tuple[str, ...] = ()
    consumes_apis: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class CatalogParseResult:
    """Result envelope for parsing a Backstage catalog-info.yaml document stream."""

    entities: tuple[BackstageEntity, ...]
    abstained: bool = False
    error: str | None = None


def normalize_backstage_ref(raw_ref: str, *, default_kind: str) -> tuple[str, str]:
    """Parses `[<kind>:][<namespace>/]<name>` and returns (kind_lower, entity_name_lower)."""
    cleaned = raw_ref.strip()
    if ":" in cleaned:
        kind_part, rest = cleaned.split(":", 1)
        kind = kind_part.strip().lower()
    else:
        kind = default_kind.lower()
        rest = cleaned

    if "/" in rest:
        _, name = rest.split("/", 1)
    else:
        name = rest
    return kind, name.strip().lower()


def parse_catalog_info_yaml(
    content: str, *, source_path: str = "catalog-info.yaml"
) -> CatalogParseResult:
    """Parses multi-document Backstage catalog-info.yaml, failing closed on invalid YAML."""
    norm_path = _normalize_rel_path(source_path)
    base_dir = norm_path.rsplit("/", 1)[0] if "/" in norm_path else ""
    try:
        docs = list(yaml.safe_load_all(content))
    except yaml.YAMLError as exc:
        logger.warning(
            "Malformed Backstage catalog YAML at %s; failing closed (%s: %s)",
            norm_path,
            type(exc).__name__,
            exc,
        )
        return CatalogParseResult(entities=(), abstained=True, error=str(exc))

    entities: list[BackstageEntity] = []
    for doc in docs:
        if doc is None:
            continue
        if not isinstance(doc, dict):
            logger.warning("Skipping non-mapping Backstage catalog document in %s", norm_path)
            return CatalogParseResult(
                entities=(), abstained=True, error="Catalog document must be a YAML mapping."
            )
        kind = str(doc.get("kind") or "").strip()
        if kind not in _BACKSTAGE_KINDS:
            continue
        meta = doc.get("metadata")
        if not isinstance(meta, dict) or not str(meta.get("name") or "").strip():
            logger.warning("Missing metadata.name in Backstage %s in %s", kind, norm_path)
            return CatalogParseResult(
                entities=(), abstained=True, error=f"Missing metadata.name for {kind}."
            )

        name = str(meta["name"]).strip().lower()
        namespace = str(meta.get("namespace") or "default").strip().lower()
        description = str(meta.get("description") or "").strip()
        raw_spec = doc.get("spec")
        spec: dict[str, Any] = raw_spec if isinstance(raw_spec, dict) else {}

        raw_owner = spec.get("owner")
        owner_id = normalize_owner_id(str(raw_owner))[0] if raw_owner else None

        raw_system = spec.get("system")
        system_name = (
            normalize_backstage_ref(str(raw_system), default_kind="system")[1]
            if raw_system
            else None
        )
        raw_domain = spec.get("domain")
        domain_name = (
            normalize_backstage_ref(str(raw_domain), default_kind="domain")[1]
            if raw_domain
            else None
        )
        raw_subcomp = spec.get("subcomponentOf")
        subcomp_name = (
            normalize_backstage_ref(str(raw_subcomp), default_kind="component")[1]
            if raw_subcomp
            else None
        )
        provides_apis = tuple(
            normalize_backstage_ref(str(api), default_kind="api")[1]
            for api in (spec.get("providesApis") or [])
            if str(api).strip()
        )
        consumes_apis = tuple(
            normalize_backstage_ref(str(api), default_kind="api")[1]
            for api in (spec.get("consumesApis") or [])
            if str(api).strip()
        )

        entities.append(
            BackstageEntity(
                kind=kind,
                name=name,
                namespace=namespace,
                base_dir=base_dir,
                source_path=norm_path,
                owner=owner_id,
                system=system_name,
                domain=domain_name,
                subcomponent_of=subcomp_name,
                provides_apis=provides_apis,
                consumes_apis=consumes_apis,
                description=description,
            )
        )

    return CatalogParseResult(entities=tuple(entities), abstained=False, error=None)


@dataclass(frozen=True)
class OwnershipGraphResult:
    """Extracted ownership nodes (team, system, component, api) and graph edges."""

    symbols: tuple[ParsedSymbol, ...]
    edges: tuple[ParsedEdge, ...]
    file_owners: dict[str, tuple[str, ...]]
    file_systems: dict[str, str]
    abstained: bool = False
    diagnostics: tuple[str, ...] = ()


def _team_slug(canonical_owner_id: str) -> str:
    if ":" in canonical_owner_id:
        _, rest = canonical_owner_id.split(":", 1)
    else:
        rest = canonical_owner_id
    return rest.rsplit("/", 1)[-1].lower()


def _canonicalize_team_aliases(owner_ids: Sequence[str]) -> dict[str, str]:
    """Deduplicates `@org/team` and `group:default/team` when only one org uses that slug."""
    orgs_by_slug: dict[str, set[str]] = {}
    bare_owners: set[str] = set()
    for oid in owner_ids:
        if oid.startswith("team:"):
            rest = oid[len("team:") :]
            if "/" in rest:
                org, slug = rest.split("/", 1)
                orgs_by_slug.setdefault(slug, set()).add(org)
            else:
                bare_owners.add(rest)

    alias_map: dict[str, str] = {}
    for oid in owner_ids:
        if oid.startswith("team:"):
            rest = oid[len("team:") :]
            if "/" in rest:
                _, slug = rest.split("/", 1)
                if len(orgs_by_slug.get(slug, set())) == 1 and slug in bare_owners:
                    alias_map[oid] = f"team:{next(iter(orgs_by_slug[slug]))}/{slug}"
                    alias_map[f"team:{slug}"] = alias_map[oid]
                else:
                    alias_map[oid] = oid
            else:
                orgs = orgs_by_slug.get(rest, set())
                if len(orgs) == 1:
                    alias_map[oid] = f"team:{next(iter(orgs))}/{rest}"
                else:
                    alias_map[oid] = oid
        else:
            alias_map[oid] = oid
    return alias_map


class OwnershipGraphExtractor:
    """Extracts CODEOWNERS and Backstage catalog-info.yaml into OWNED_BY_TEAM / PART_OF_SYSTEM."""

    def extract(
        self,
        *,
        repo_name: str,
        files: Mapping[str, str],
        code_symbols_by_path: Mapping[str, Sequence[str]] | None = None,
    ) -> OwnershipGraphResult:
        normalized_files = {_normalize_rel_path(k): v for k, v in files.items()}
        codeowners = CodeOwnersMatcher.from_repo_files(normalized_files)

        all_entities: list[BackstageEntity] = []
        diagnostics: list[str] = []
        abstained = False
        for rel_path, content in sorted(normalized_files.items()):
            basename = rel_path.rsplit("/", 1)[-1]
            if basename in CATALOG_INFO_FILENAMES:
                parsed = parse_catalog_info_yaml(content, source_path=rel_path)
                if parsed.abstained:
                    abstained = True
                    if parsed.error:
                        diagnostics.append(f"{rel_path}: {parsed.error}")
                    continue
                all_entities.extend(parsed.entities)

        raw_owner_ids: list[str] = []
        file_raw_owners: dict[str, tuple[str, ...]] = {}
        for rel_path in sorted(normalized_files):
            matched = codeowners.match_owners(rel_path) if codeowners.has_rules else ()
            if matched:
                file_raw_owners[rel_path] = matched
                raw_owner_ids.extend(matched)

        for entity in all_entities:
            if entity.owner:
                raw_owner_ids.append(entity.owner)
            if entity.kind == "Group":
                raw_owner_ids.append(f"team:{entity.name}")

        alias_map = _canonicalize_team_aliases(raw_owner_ids)
        symbols_by_qn: dict[str, ParsedSymbol] = {}
        edges_seen: set[tuple[str, str, str]] = set()
        edges: list[ParsedEdge] = []

        def _ensure_symbol(
            *,
            qualified_name: str,
            name: str,
            symbol_kind: str,
            signature: str,
            summary: str,
        ) -> None:
            if qualified_name in symbols_by_qn:
                return
            symbols_by_qn[qualified_name] = ParsedSymbol(
                name=name,
                qualified_name=qualified_name,
                symbol_kind=symbol_kind,
                start_line=1,
                end_line=1,
                start_byte=0,
                end_byte=max(1, len(signature)),
                signature=signature,
                signature_skeleton=signature,
                docstring=summary,
                contextual_summary=summary,
                repo_name=repo_name,
                repo_qualified_id=f"{repo_name}::{qualified_name}",
                bare_name=name,
            )

        def _ensure_team_symbol(raw_owner: str) -> str:
            canonical = alias_map.get(raw_owner, raw_owner)
            ident = canonical.split(":", 1)[-1]
            qn = f"ownership.team.{ident.replace('/', '.')}"
            _ensure_symbol(
                qualified_name=qn,
                name=ident,
                symbol_kind="team",
                signature=f"team {canonical}",
                summary=f"Repository ownership team {canonical}",
            )
            return qn

        def _ensure_system_symbol(system_name: str, description: str = "") -> str:
            qn = f"ownership.system.{system_name}"
            _ensure_symbol(
                qualified_name=qn,
                name=system_name,
                symbol_kind="system",
                signature=f"system {system_name}",
                summary=description or f"Backstage system {system_name}",
            )
            return qn

        def _emit_edge(src_qn: str, tgt_qn: str, edge_kind: str) -> None:
            key = (src_qn, tgt_qn, edge_kind)
            if key in edges_seen:
                return
            edges_seen.add(key)
            edges.append(
                ParsedEdge(
                    source_qualified_name=src_qn,
                    target_qualified_name=tgt_qn,
                    edge_kind=edge_kind,
                    target_repo_qualified_id=f"{repo_name}::{tgt_qn}",
                    is_cross_repo=False,
                )
            )

        # 1. Materialize Backstage entities & edges
        components_by_dir: dict[str, list[BackstageEntity]] = {}
        for entity in all_entities:
            if entity.kind == "System":
                sys_qn = _ensure_system_symbol(entity.name, entity.description)
                if entity.owner:
                    team_qn = _ensure_team_symbol(entity.owner)
                    _emit_edge(sys_qn, team_qn, "OWNED_BY_TEAM")
            elif entity.kind == "Group":
                _ensure_team_symbol(f"team:{entity.name}")
            elif entity.kind in {"Component", "API", "Resource"}:
                kind_lower = "api" if entity.kind == "API" else "component"
                comp_qn = f"ownership.{kind_lower}.{entity.name}"
                _ensure_symbol(
                    qualified_name=comp_qn,
                    name=entity.name,
                    symbol_kind=kind_lower,
                    signature=f"{entity.kind.lower()} {entity.name}",
                    summary=entity.description
                    or f"Backstage {entity.kind} {entity.name} ({entity.source_path})",
                )
                if entity.kind == "Component":
                    components_by_dir.setdefault(entity.base_dir, []).append(entity)
                if entity.owner:
                    team_qn = _ensure_team_symbol(entity.owner)
                    _emit_edge(comp_qn, team_qn, "OWNED_BY_TEAM")
                if entity.system:
                    sys_qn = _ensure_system_symbol(entity.system)
                    _emit_edge(comp_qn, sys_qn, "PART_OF_SYSTEM")

        # 2. Link code files and their symbols to CODEOWNERS teams + Backstage systems
        file_owners: dict[str, tuple[str, ...]] = {}
        file_systems: dict[str, str] = {}
        sorted_dirs = sorted(components_by_dir.keys(), key=len, reverse=True)

        for rel_path in sorted(normalized_files):
            matched_component: BackstageEntity | None = None
            for base_dir in sorted_dirs:
                if not base_dir or rel_path == base_dir or rel_path.startswith(f"{base_dir}/"):
                    cands = components_by_dir[base_dir]
                    if cands:
                        matched_component = cands[0]
                        break

            raw_owners = file_raw_owners.get(rel_path, ())
            canonical_owners = tuple(dict.fromkeys(alias_map.get(o, o) for o in raw_owners))
            if not canonical_owners and matched_component and matched_component.owner:
                canonical_owners = (
                    alias_map.get(matched_component.owner, matched_component.owner),
                )

            if canonical_owners:
                file_owners[rel_path] = canonical_owners

            system_qn: str | None = None
            if matched_component and matched_component.system:
                file_systems[rel_path] = matched_component.system
                system_qn = _ensure_system_symbol(matched_component.system)

            code_qns = (
                list(code_symbols_by_path.get(rel_path, ()))
                if code_symbols_by_path is not None
                else []
            )
            if not code_qns:
                continue

            team_qns = [_ensure_team_symbol(o) for o in canonical_owners]
            for sym_qn in code_qns:
                for t_qn in team_qns:
                    _emit_edge(sym_qn, t_qn, "OWNED_BY_TEAM")
                if system_qn is not None:
                    _emit_edge(sym_qn, system_qn, "PART_OF_SYSTEM")

        return OwnershipGraphResult(
            symbols=tuple(symbols_by_qn.values()),
            edges=tuple(edges),
            file_owners=file_owners,
            file_systems=file_systems,
            abstained=abstained,
            diagnostics=tuple(diagnostics),
        )


def has_ownership_or_attr_files(paths: Sequence[str] | Mapping[str, Any]) -> bool:
    """Fast check returning True when .gitattributes, CODEOWNERS, or catalog exists."""
    for raw in paths:
        norm = _normalize_rel_path(str(raw))
        base = norm.rsplit("/", 1)[-1]
        if (
            base == ".gitattributes"
            or norm in CODEOWNERS_FILENAMES
            or base in CATALOG_INFO_FILENAMES
        ):
            return True
    return False


__all__ = [
    "CATALOG_INFO_FILENAMES",
    "CODEOWNERS_FILENAMES",
    "WEIGHT_GENERATED",
    "WEIGHT_GENERATED_AND_VENDORED",
    "WEIGHT_NORMAL",
    "WEIGHT_VENDORED",
    "BackstageEntity",
    "CatalogParseResult",
    "CodeOwnersMatcher",
    "CodeOwnersRule",
    "FileWeightClassification",
    "GitAttributesMatcher",
    "OwnershipGraphExtractor",
    "OwnershipGraphResult",
    "has_ownership_or_attr_files",
    "normalize_backstage_ref",
    "normalize_owner_id",
    "normalize_rel_path",
    "parse_catalog_info_yaml",
]
