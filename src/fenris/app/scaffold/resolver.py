"""Resolve a minimal new cst.Module from collected target data.

Resolves required imports from the accesses associated with required
class variables and abstract methods. Organizes a default implementation
without overriding method bodies or assign targets.

Functions
---------
resolve
"""

import heapq
import sys
from collections import defaultdict
from collections.abc import Iterable
from enum import IntEnum
from typing import Self, cast

import libcst as cst
from libcst import MetadataWrapper
from libcst.metadata import (
    Assignment,
    BuiltinAssignment,
    ScopeProvider,
)

from fenris import ROOT_PACKAGE
from fenris.app.scaffold.collector import ClsVar, FunctionDef, TargetData


def resolve(target: TargetData) -> cst.Module:
    return cst.Module(
        body=[
            *_resolve_imports(target),
            cst.ClassDef(
                name=cst.Name(value=f"MyAwesome{target.name}"),
                bases=[cst.Arg(value=cst.Name(value=target.name))],
                body=cst.IndentedBlock(body=[*_resolve_class_body(target)]),
                leading_lines=[cst.EmptyLine(), cst.EmptyLine()],
            ),
        ]
    )


def _resolve_imports(target: TargetData) -> Iterable[cst.SimpleStatementLine]:
    assignments: list[tuple[Assignment, str | None]] = []
    # Create an assignment and use the same code path as for any other
    # assignment. Not necessarily elegant, but convenient.
    wrapper = MetadataWrapper(
        cst.Module(
            body=[
                cst.ensure_type(
                    cst.parse_statement(f"from {target.module} import {target.name}"),
                    cst.SimpleStatementLine,
                )
            ]
        )
    )
    scope = next(iter(set(wrapper.resolve(ScopeProvider).values())))
    if scope is not None:
        # I do love static analysis tools, but they can be annoying,
        # scope will never be None...
        for assignment in scope.assignments:
            # Definitely not a BuiltinAssignment
            assignments.append((cast(Assignment, assignment), None))

    nodes: Iterable[ClsVar | FunctionDef] = (
        *target.required_cls_vars,
        *target.abstract_fn_defs,
    )
    for node in nodes:
        for access in node.accesses:
            if isinstance(access.node, cst.BaseString):
                continue

            if len(access.referents) == 0:
                raise NameError(access.node.value)

            if len(access.referents) > 1:
                raise ValueError(f"Reference {access.node.value} is ambiguous.")

            assignment = next(iter(access.referents))
            if isinstance(assignment, BuiltinAssignment):
                continue
            assignments.append((cast(Assignment, assignment), node.module))

    return _resolve_assignments(assignments)


def _resolve_class_body(
    target: TargetData,
) -> Iterable[cst.SimpleStatementLine | cst.FunctionDef]:

    cls_vars = tuple(
        cst.SimpleStatementLine(body=[cls_var.node])
        for cls_var in sorted(
            target.required_cls_vars,
            key=lambda c: (-c.mro_index, c.index),
        )
    )
    yield from cls_vars

    for idx, fn_def in enumerate(
        sorted(
            target.abstract_fn_defs,
            key=lambda w: (int(not w.is_property), -w.mro_index, w.index),
        )
    ):
        yield fn_def.node.with_changes(
            decorators=fn_def.node.decorators[:-1],
            leading_lines=[] if idx == 0 and not cls_vars else [cst.EmptyLine()],
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
        if base == ROOT_PACKAGE:
            return cls(3)
        return cls(2)


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
