import heapq
import sys
from collections import defaultdict
from enum import IntEnum
from functools import cached_property
from typing import Iterable, Self, cast

import libcst as cst
from libcst import MetadataWrapper
from libcst.metadata import (
    Assignment,
    BuiltinAssignment,
    ScopeProvider,
)

from fenris.app.scaffold.collector import ClsVar, Collector, FunctionDef


class Resolver:
    def __init__(self, collector: Collector):
        self._collector = collector

    @cached_property
    def module(self) -> cst.Module:
        self._collector.maybe_collect()
        class_def = cst.ClassDef(
            name=cst.Name(value=f"MyAwesome{self._collector.target_name}"),
            bases=[cst.Arg(value=cst.Name(value=self._collector.target_name))],
            body=cst.IndentedBlock(body=[*self._resolve_fn_defs()]),
            leading_lines=[cst.EmptyLine(), cst.EmptyLine()],
        )
        return cst.Module(body=[*self._resolve_imports(), class_def])

    def _required_cls_vars(self) -> Iterable[ClsVar]:
        for cls_var in self._collector.cls_vars:
            if cls_var.is_required:
                yield cls_var

    def _abstract_fn_defs(self) -> Iterable[FunctionDef]:
        for fn_def in self._collector.fn_defs:
            if fn_def.is_abstract:
                yield fn_def

    def _collected_assignments(self) -> Iterable[tuple[Assignment, str | None]]:
        for fn_def in self._abstract_fn_defs():
            for access in fn_def.accesses:
                if isinstance(access.node, cst.BaseString):
                    continue

                if len(access.referents) == 0:
                    raise NameError(access.node.value)

                if len(access.referents) > 1:
                    raise ValueError(f"Reference {access.node.value} is ambiguous.")

                assignment = next(iter(access.referents))
                if isinstance(assignment, BuiltinAssignment):
                    continue

                if isinstance(assignment, Assignment):
                    yield assignment, fn_def.module

    def _resolve_imports(self) -> Iterable[cst.SimpleStatementLine]:
        # Create an assignment and use the same code path as for any other
        # assignment. Not necessarily elegant, but convenient.
        wrapper = MetadataWrapper(
            cst.Module(
                body=[
                    cst.ensure_type(
                        cst.parse_statement(
                            f"from {self._collector.target_module} import "
                            f"{self._collector.target_name}"
                        ),
                        cst.SimpleStatementLine,
                    )
                ]
            )
        )
        scope = next(iter(set(wrapper.resolve(ScopeProvider).values())))
        return tuple(
            _resolve_assignments(
                (
                    *self._collected_assignments(),
                    *((a, None) for a in scope.assignments),
                )
            )
        )

    def _resolve_fn_defs(self) -> Iterable[cst.FunctionDef]:
        def key(_fn_def: FunctionDef) -> tuple[int, int, int]:
            return (
                int(not _fn_def.is_property),
                -_fn_def.mro_index,
                _fn_def.method_index,
            )

        for idx, fn_def in enumerate(sorted(self._abstract_fn_defs(), key=key)):
            yield fn_def.node.with_changes(
                decorators=fn_def.node.decorators[:-1],
                body=_fn_body_not_implemented_error(),
                leading_lines=[cst.EmptyLine()] if idx > 0 else [],
            )


def _fn_body_not_implemented_error() -> cst.IndentedBlock:
    return cst.IndentedBlock(
        body=[
            cst.SimpleStatementLine(
                body=[cst.Raise(exc=cst.Call(func=cst.Name("NotImplementedError")))]
            )
        ]
    )


class _Group(IntEnum):
    STDLIB = 1
    EXTLIB = 2
    PROJECT = 3

    @classmethod
    def from_module(cls, module: str) -> Self:
        base = module.split(".")[0]
        if base in sys.stdlib_module_names:
            return cls(1)
        if base == __name__.split(".")[0]:
            return cls(3)
        return cls(2)


def _resolve_assignment(
    assignment: Assignment,
    src_module: str | None,
) -> tuple[str | None, str] | None:

    node = assignment.node
    if not isinstance(node, cst.Import | cst.ImportFrom):
        if src_module is None:
            raise ValueError(
                "src_module None not allowed when assignment node is not an import "
                "statement."
            )
        return src_module, assignment.name

    if isinstance(node.names, cst.ImportStar):
        raise ValueError("Please, do not use wildcard imports.")

    if getattr(node, "relative", False):
        raise NotImplementedError("No support for relative imports.")

    def codegen(_node: cst.CSTNode) -> str:
        if isinstance(_node, cst.ImportAlias):
            _node = _node.with_changes(comma=cst.MaybeSentinel.DEFAULT)
        return cst.Module([]).code_for_node(_node)

    module = None
    if hasattr(node, "module"):
        # Is not None because we already crashed if relative import
        module = codegen(cast(cst.Attribute | cst.Name, node.module))

    for alias in node.names:
        if alias.asname is not None:
            name = alias.asname.name
        else:
            name = alias.name  # type: ignore[assignment]

        if not isinstance(name, cst.Name | cst.Attribute):
            raise TypeError(f"AsName.name has unexpected type {type(name)}.")

        if codegen(name) == assignment.name:
            return module, codegen(alias)
    return None


def _resolve_assignments(
    assignments: Iterable[tuple[Assignment, str | None]],
) -> Iterable[cst.SimpleStatementLine]:

    unique: dict[str | None, set[str]] = defaultdict(set)
    for assignment, src_module in assignments:
        resolved = _resolve_assignment(assignment, src_module)
        if resolved is not None:
            from_module, alias = resolved
            unique[from_module].add(alias)

    heap: list[tuple[int, str, str]] = []

    def push(_module: str, _stmt: str) -> None:
        heapq.heappush(heap, (_Group.from_module(_module), _module, _stmt))

    for from_module, aliases in unique.items():
        if from_module is None:
            for alias in aliases:
                push(alias.split(" ")[0], f"import {alias}")
        else:
            push(from_module, f"from {from_module} import {', '.join(sorted(aliases))}")

    prev_group = None
    while heap:
        group, _, stmt = heapq.heappop(heap)
        node = cst.ensure_type(cst.parse_statement(stmt), cst.SimpleStatementLine)
        if prev_group and group != prev_group:
            node = node.with_changes(leading_lines=[cst.EmptyLine()])
        prev_group = group
        yield node
