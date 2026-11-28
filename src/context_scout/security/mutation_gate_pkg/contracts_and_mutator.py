"""AST mutation gate contracts, AST site counter, and isolated in-memory mutator."""

from __future__ import annotations

import ast
import contextlib
import copy
import sys
import types
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

DEFAULT_MIN_MUTATION_KILL_RATE: Final[float] = 0.85
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class MutantSpec:
    """Identifies a single concrete AST mutation site inside a target module."""

    module_relpath: str
    scope_name: str
    class_name: str | None
    function_name: str
    site_index: int
    line_no: int
    mutation_kind: str
    description: str


@dataclass(frozen=True)
class MutantOutcome:
    """Outcome of running the verification probe against one AST mutant."""

    spec: MutantSpec
    killed: bool
    kill_reason: str = ""
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class TargetMutationReport:
    """Aggregated mutation kill statistics for one required M3-03 scope."""

    scope_name: str
    total_mutants: int
    killed_mutants: int
    survived_mutants: int
    kill_rate: float
    min_kill_rate: float
    passed: bool
    elapsed_ms: float
    survivors: tuple[MutantOutcome, ...] = ()


@dataclass(frozen=True)
class MutationGateSummary:
    """Overall M3-03 mutation gate report across all 4 required scopes."""

    total_mutants: int
    killed_mutants: int
    survived_mutants: int
    overall_kill_rate: float
    min_kill_rate: float
    passed: bool
    elapsed_ms: float
    scope_reports: dict[str, TargetMutationReport] = field(default_factory=dict)


_CMP_FLIPS: Final[dict[type[ast.cmpop], type[ast.cmpop]]] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}


def _is_logger_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "logger"
    )


def _should_skip_node(node: ast.AST) -> bool:
    if _is_logger_call(node):
        return True
    return isinstance(node, ast.Expr) and _is_logger_call(node.value)


class _SiteCounterVisitor(ast.NodeVisitor):
    """Collects deterministic, non-trivial AST mutation candidates inside a function."""

    def __init__(self) -> None:
        self.candidates: list[tuple[int, int, str, str]] = []
        self._idx = 0
        self._in_return_or_assign = False
        self._in_nested_def = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._in_nested_def > 0:
            return
        self._in_nested_def += 1
        for stmt in node.body:
            self.visit(stmt)
        self._in_nested_def -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._in_nested_def > 0:
            return
        self._in_nested_def += 1
        for stmt in node.body:
            self.visit(stmt)
        self._in_nested_def -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Expr(self, node: ast.Expr) -> None:
        if _should_skip_node(node):
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_logger_call(node):
            return
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        prev = self._in_return_or_assign
        self._in_return_or_assign = True
        self.generic_visit(node)
        self._in_return_or_assign = prev

    def visit_Assign(self, node: ast.Assign) -> None:
        prev = self._in_return_or_assign
        self._in_return_or_assign = True
        self.generic_visit(node)
        self._in_return_or_assign = prev

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.ops) == 1 and type(node.ops[0]) in _CMP_FLIPS:
            op_name = type(node.ops[0]).__name__
            flip_name = _CMP_FLIPS[type(node.ops[0])].__name__
            self.candidates.append(
                (
                    self._idx,
                    getattr(node, "lineno", 0),
                    "compare_flip",
                    f"{op_name} -> {flip_name}",
                )
            )
            self._idx += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if isinstance(node.op, (ast.And, ast.Or)):
            op_name = type(node.op).__name__
            flip_name = "Or" if isinstance(node.op, ast.And) else "And"
            self.candidates.append(
                (
                    self._idx,
                    getattr(node, "lineno", 0),
                    "boolop_flip",
                    f"{op_name} -> {flip_name}",
                )
            )
            self._idx += 1
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.Not):
            self.candidates.append(
                (
                    self._idx,
                    getattr(node, "lineno", 0),
                    "unary_not_drop",
                    "drop unary not",
                )
            )
            self._idx += 1
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if self._in_return_or_assign and isinstance(node.value, bool):
            self.candidates.append(
                (
                    self._idx,
                    getattr(node, "lineno", 0),
                    "bool_const_flip",
                    f"{node.value!r} -> {not node.value!r}",
                )
            )
            self._idx += 1


class _SingleSiteMutator(ast.NodeTransformer):
    """Applies exactly one AST mutation identified by target `site_index`."""

    def __init__(self, target_index: int) -> None:
        self._target_index = target_index
        self._idx = 0
        self._in_return_or_assign = False
        self._in_nested_def = 0
        self.applied = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if self._in_nested_def > 0:
            return node
        self._in_nested_def += 1
        node.body = [self.visit(stmt) for stmt in node.body]
        self._in_nested_def -= 1
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        if self._in_nested_def > 0:
            return node
        self._in_nested_def += 1
        node.body = [self.visit(stmt) for stmt in node.body]
        self._in_nested_def -= 1
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        if _should_skip_node(node):
            return node
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if _is_logger_call(node):
            return node
        return self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> ast.AST:
        prev = self._in_return_or_assign
        self._in_return_or_assign = True
        res = self.generic_visit(node)
        self._in_return_or_assign = prev
        return res

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        prev = self._in_return_or_assign
        self._in_return_or_assign = True
        res = self.generic_visit(node)
        self._in_return_or_assign = prev
        return res

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if len(node.ops) == 1 and type(node.ops[0]) in _CMP_FLIPS:
            cur = self._idx
            self._idx += 1
            node = self.generic_visit(node)  # type: ignore[assignment]
            if cur == self._target_index:
                flipped = _CMP_FLIPS[type(node.ops[0])]()
                node.ops = [flipped]
                self.applied = True
            return node
        return self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        if isinstance(node.op, (ast.And, ast.Or)):
            cur = self._idx
            self._idx += 1
            node = self.generic_visit(node)  # type: ignore[assignment]
            if cur == self._target_index:
                node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
                self.applied = True
            return node
        return self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        if isinstance(node.op, ast.Not):
            cur = self._idx
            self._idx += 1
            node = self.generic_visit(node)  # type: ignore[assignment]
            if cur == self._target_index:
                self.applied = True
                return node.operand
            return node
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self._in_return_or_assign and isinstance(node.value, bool):
            cur = self._idx
            self._idx += 1
            if cur == self._target_index:
                self.applied = True
                return ast.copy_location(ast.Constant(value=not node.value), node)
        return node


def _find_function_node(
    tree: ast.Module, class_name: str | None, function_name: str
) -> ast.FunctionDef | None:
    if class_name is None:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return node
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == function_name:
                    return item
    return None


@contextlib.contextmanager
def _isolated_module(relpath: str) -> Any:
    abs_path = _REPO_ROOT / relpath
    source = abs_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(abs_path))
    mod_name = f"_mutgate_{abs_path.stem}_{uuid.uuid4().hex[:8]}"
    mod = types.ModuleType(mod_name)
    mod.__file__ = str(abs_path)
    sys.modules[mod_name] = mod
    try:
        exec(compile(tree, str(abs_path), "exec"), mod.__dict__)
        yield mod, tree, abs_path
    finally:
        sys.modules.pop(mod_name, None)


def _swap_mutated_function(
    mod: types.ModuleType,
    abs_path: Path,
    orig_fn: ast.FunctionDef,
    class_name: str | None,
    site_index: int,
) -> tuple[Any, Callable[[], None]]:
    mutated_fn = copy.deepcopy(orig_fn)
    mutator = _SingleSiteMutator(site_index)
    mutated_fn = mutator.visit(mutated_fn)
    assert mutator.applied, f"Failed to apply mutation site {site_index} in {orig_fn.name}"

    wrapper = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations", asname=None)],
                level=0,
            ),
            mutated_fn,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(wrapper)
    local_ns: dict[str, Any] = {}
    exec(compile(wrapper, str(abs_path), "exec"), mod.__dict__, local_ns)
    new_attr = local_ns[orig_fn.name]

    if class_name is None:
        prev_attr = getattr(mod, orig_fn.name)
        setattr(mod, orig_fn.name, new_attr)

        def _restore_mod() -> None:
            setattr(mod, orig_fn.name, prev_attr)

        return new_attr, _restore_mod

    cls_obj = getattr(mod, class_name)
    prev_attr = cls_obj.__dict__[orig_fn.name]
    setattr(cls_obj, orig_fn.name, new_attr)

    def _restore_cls() -> None:
        setattr(cls_obj, orig_fn.name, prev_attr)

    return new_attr, _restore_cls


@dataclass(frozen=True)
class _TargetModuleConfig:
    scope_name: str
    relpath: str
    targets: tuple[tuple[str | None, str], ...]
    probe_fn: Callable[[types.ModuleType], None]
