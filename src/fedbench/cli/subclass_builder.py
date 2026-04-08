import inspect
import sys
from abc import ABC
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Self, cast

import libcst as cst
import libcst.matchers as m
from libcst.metadata import (
    Access,
    Assignment,
    BuiltinAssignment,
    ExpressionContext,
    ExpressionContextProvider,
    MetadataWrapper,
    ScopeProvider,
)


class _ImportGroup(IntEnum):
    STDLIB = 1
    EXTLIB = 2
    PROJECT = 3

    @classmethod
    def get_for_module(cls, module: str) -> Self:
        base = module.split(".")[0]
        if base in sys.stdlib_module_names:
            return cls(1)
        if base == __name__.split(".")[0]:
            return cls(3)
        return cls(2)


@dataclass(frozen=True)
class _ImportWrapper:
    node: cst.SimpleStatementLine
    group: _ImportGroup
    module: str


@dataclass(frozen=True)
class _FunctionDefWrapper:
    node: cst.FunctionDef = field(repr=False)
    is_property: bool
    is_abstract: bool
    mro_index: int
    method_index: int
    module: str
    accesses: list[Access] = field(default_factory=list)


class AbstractMethodCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (ScopeProvider, ExpressionContextProvider)

    def __init__(self, parent_cls: type) -> None:
        super().__init__()
        self._mro = tuple(c for c in parent_cls.mro() if c not in (ABC, object))
        self._curr_modulename: str = ""
        self._collected: bool = False
        self._cls_stack: list[str] = []
        self._fn_stack: list[str] = []
        self._fn_defs: dict[str, list[_FunctionDefWrapper | None]] = defaultdict(
            lambda: [None for _ in self._mro]
        )

    def __iter__(self) -> Iterator[_FunctionDefWrapper]:
        for fn_defs in self._fn_defs.values():
            fn_def = next(fn for fn in fn_defs if fn is not None)
            if fn_def.is_abstract:
                yield fn_def

    @property
    def parent_cls(self) -> type:
        return self._mro[0]

    @property
    def _curr_mro_index(self) -> int:
        if self._cls_stack:
            cls_name = self._cls_stack[-1]
            for index, cls in enumerate(self._mro):
                if cls.__name__ == cls_name:
                    return index
        return -1

    def visit_Import(self, node: cst.Import) -> bool:
        return False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._cls_stack.append(node.name.value)
        return self._curr_mro_index >= 0

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self._cls_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if not self._cls_stack:
            return False

        self._fn_stack.append(node.name.value)

        is_abstract = False
        is_property = False

        for dec in node.decorators:
            if m.matches(dec.decorator, m.Name("property")):
                is_property = True
                break
        try:
            is_abstract = m.matches(
                node.decorators[-1].decorator,  # Always innermost dec
                m.Name("abstractmethod")
                | m.Attribute(value=m.Name("abc"), attr=m.Name("abstractmethod")),
            )
        except IndexError:
            pass

        fn_def = _FunctionDefWrapper(
            node,
            is_property,
            is_abstract,
            self._curr_mro_index,
            len(self._fn_defs),
            self._curr_modulename,
        )
        self._fn_defs[node.name.value][self._curr_mro_index] = fn_def

        # Visit decorators, params and returns in order to associate accesses
        # with the current FunctionDef.
        for dec in node.decorators[:-1]:
            dec.visit(self)

        node.params.visit(self)
        if node.returns:
            node.returns.visit(self)

        return False

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        if not self._cls_stack:
            return
        self._fn_stack.pop()

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        return self._visit_attr_or_name(node)

    def visit_Name(self, node: cst.Name) -> bool:
        return self._visit_attr_or_name(node)

    def _visit_attr_or_name(self, node: cst.Attribute | cst.Name) -> bool:
        # Relevant accesses from decorators, params and returns all belong to the
        # class scope. The goal here is to associate those accesses with a specific
        # FunctionDef, to know which accesses are required to build an appropriate
        # scope for the generated subclass.

        if self._curr_mro_index < 0 or not self._fn_stack:
            return False

        expr_context = self.get_metadata(ExpressionContextProvider, node, None)
        if expr_context != ExpressionContext.LOAD:
            return False

        scope = self.get_metadata(ScopeProvider, node)
        if scope is None:
            return False

        fn_def = cast(
            _FunctionDefWrapper, self._fn_defs[self._fn_stack[-1]][self._curr_mro_index]
        )
        for access in scope.accesses[node]:
            if access.node is node:
                fn_def.accesses.append(access)

        return True

    def maybe_collect(self) -> Iterable[_FunctionDefWrapper]:
        if not self._collected:
            self._collect()
            self._collected = True
        yield from self

    def _collect(self) -> None:
        processed = set()
        for cls in self._mro:
            src_file = inspect.getsourcefile(cls)
            if src_file is None:
                raise ValueError(
                    f"Failed to find source file for class {cls.__name__}."
                )
            if src_file in processed:
                continue

            with Path(src_file).resolve().open() as f:
                code = f.read()

            self._curr_modulename = cls.__module__
            wrapper = MetadataWrapper(cst.parse_module(code))
            wrapper.visit(self)
            processed.add(src_file)
        self._curr_modulename = ""


class Builder:
    def __init__(self, collector: AbstractMethodCollector):
        self._collector = collector
        self._imports: tuple[_ImportWrapper, ...] | None = None
        self._fn_defs: tuple[_FunctionDefWrapper, ...] | None = None
        self._name: str | None = None

    def with_name(self, name: str) -> Self:
        self._name = name
        return self

    def build(self) -> cst.Module:
        self._maybe_collect()
        imports: list[cst.SimpleStatementLine] = []
        fn_defs: list[cst.FunctionDef] = []

        prev_imp_group: _ImportGroup | None = None
        for imp in cast(tuple[_ImportWrapper, ...], self._imports):
            imports.append(
                imp.node.with_changes(
                    leading_lines=[cst.EmptyLine()]
                    if prev_imp_group and imp.group != prev_imp_group
                    else []
                )
            )
            prev_imp_group = imp.group

        for idx, fn_def in enumerate(
            cast(tuple[_FunctionDefWrapper, ...], self._fn_defs)
        ):
            fn_defs.append(
                fn_def.node.with_changes(
                    decorators=fn_def.node.decorators[:-1],
                    body=cst.IndentedBlock(
                        body=[
                            cst.SimpleStatementLine(
                                body=[
                                    cst.Raise(
                                        exc=cst.Call(
                                            func=cst.Name("NotImplementedError")
                                        )
                                    )
                                ]
                            )
                        ]
                    ),
                    leading_lines=[cst.EmptyLine()] if idx > 0 else [],
                )
            )
        parent_cls_name = self._collector.parent_cls.__name__
        name = self._name or f"MyAwesome{parent_cls_name}"
        tree = cst.Module(
            body=[
                *imports,
                cst.ClassDef(
                    name=cst.Name(value=name),
                    bases=[cst.Arg(value=cst.Name(value=parent_cls_name))],
                    body=cst.IndentedBlock(body=fn_defs),
                    leading_lines=[cst.EmptyLine(), cst.EmptyLine()],
                ),
            ]
        )
        self._name = None
        return tree

    def _maybe_collect(self) -> None:
        if self._imports is not None:
            return

        def import_key(imp: _ImportWrapper) -> tuple[int, str]:
            return imp.group.value, imp.module

        def fn_def_key(fn_def: _FunctionDefWrapper) -> tuple[int, int, int]:
            return int(not fn_def.is_property), -fn_def.mro_index, fn_def.method_index

        imports = list(_resolve_imports(self._collector.maybe_collect()))
        imports.append(
            _ImportWrapper(
                cst.ensure_type(
                    cst.parse_statement(
                        f"from {self._collector.parent_cls.__module__} "
                        f"import {self._collector.parent_cls.__name__}"
                    ),
                    cst.SimpleStatementLine,
                ),
                _ImportGroup.PROJECT,
                self._collector.parent_cls.__module__,
            )
        )
        self._imports = tuple(sorted(imports, key=import_key))
        self._fn_defs = tuple((sorted(self._collector, key=fn_def_key)))


def _resolve_imports(
    fn_defs: Iterable[_FunctionDefWrapper],
) -> Iterable[_ImportWrapper]:

    imports: dict[str | None, set[str]] = defaultdict(set[str])
    for fn_def in fn_defs:
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
                module, aliases = _resolve_assignment(assignment, fn_def.module)
                if aliases:
                    imports[module].add(aliases)

    return _create_import_wrappers(imports)


def _resolve_assignment(
    assignment: Assignment, src_module: str
) -> tuple[str | None, str]:

    def codegen(_node: cst.CSTNode) -> str:
        if isinstance(_node, cst.ImportAlias):
            _node = _node.with_changes(comma=cst.MaybeSentinel.DEFAULT)
        return cst.Module([]).code_for_node(_node)

    node = assignment.node
    if isinstance(node, cst.Import | cst.ImportFrom):
        if isinstance(node.names, cst.ImportStar):
            raise ValueError("Please, do not use wildcard imports.")

        if getattr(node, "relative", False):
            raise NotImplementedError("No support for relative imports.")

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
        return None, ""

    return src_module, assignment.name


def _create_import_wrappers(
    imports: dict[str | None, set[str]],
) -> Iterable[_ImportWrapper]:

    for module, aliases in imports.items():
        if module is None:
            for alias in aliases:
                _module = alias.split(" ")[0]
                yield _ImportWrapper(
                    cst.ensure_type(
                        cst.parse_statement(f"import {alias}"), cst.SimpleStatementLine
                    ),
                    _ImportGroup.get_for_module(_module),
                    _module,
                )
        else:
            yield _ImportWrapper(
                cst.ensure_type(
                    cst.parse_statement(
                        f"from {module} import {', '.join(sorted(aliases))}"
                    ),
                    cst.SimpleStatementLine,
                ),
                _ImportGroup.get_for_module(module),
                module,
            )
