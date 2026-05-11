"""Collect data relevant for scaffolding some target ABC.

Classes
-------
Collector
    Given some target ABC, locates source files, parses with libcst
    and collects relevant data from the resulting syntax tree.
TargetData
    Collector output.
ClsVar
    A collected class variable assignment. Wraps a libcst.AnnAssign node.
FunctionDef
    A collected function definition. Wraps a libcst.FunctionDef node.
"""

import inspect
from abc import ABC
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import libcst as cst
from libcst import MetadataWrapper
from libcst import matchers as m
from libcst.metadata import (
    Access,
    ExpressionContext,
    ExpressionContextProvider,
    ScopeProvider,
)

from fenris.app.scaffold.tags import REQUIRED_CLS_VAR


@dataclass
class _NodeWrapper[T]:
    node: T = field(repr=False)
    module: str
    mro_index: int
    index: int
    accesses: list[Access] = field(init=False, repr=False, default_factory=list)


@dataclass
class ClsVar(_NodeWrapper[cst.AnnAssign]):
    is_required: bool


@dataclass
class FunctionDef(_NodeWrapper[cst.FunctionDef]):
    is_property: bool
    is_abstract: bool


@dataclass
class TargetData:
    cls: type
    name: str
    module: str
    cls_vars: tuple[ClsVar, ...]
    fn_defs: tuple[FunctionDef, ...]

    @property
    def required_cls_vars(self) -> Iterable[ClsVar]:
        return filter(lambda n: n.is_required, self.cls_vars)

    @property
    def abstract_fn_defs(self) -> Iterable[FunctionDef]:
        return filter(lambda n: n.is_abstract, self.fn_defs)


class Collector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (ScopeProvider, ExpressionContextProvider)

    def __init__(self, target_cls: type) -> None:
        super().__init__()
        self._mro = tuple(c for c in target_cls.mro() if c not in (ABC, object))
        self._fn_defs: dict[str, list[FunctionDef | None]] = defaultdict(
            lambda: [None for _ in self._mro]
        )
        self._cls_vars: dict[str, list[ClsVar | None]] = defaultdict(
            lambda: [None for _ in self._mro]
        )
        self._curr_modulename: str = ""
        self._cls_stack: list[str] = []
        self._wrapper_stack: list[
            _NodeWrapper[cst.AnnAssign] | _NodeWrapper[cst.FunctionDef]
        ] = []

    def collect(self) -> TargetData:
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

        return TargetData(
            self._mro[0],
            self._mro[0].__name__,
            self._mro[0].__module__,
            tuple(
                next(n for n in value if n is not None)
                for value in self._cls_vars.values()
            ),
            tuple(
                next(n for n in value if n is not None)
                for value in self._fn_defs.values()
            ),
        )

    @property
    def _curr_mro_index(self) -> int:
        if self._cls_stack:
            cls_name = self._cls_stack[-1]
            for index, cls in enumerate(self._mro):
                if cls.__name__ == cls_name:
                    return index
        return -1

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._cls_stack.append(node.name.value)
        return self._curr_mro_index >= 0

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self._cls_stack.pop()

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> bool:
        scope = self.get_metadata(ScopeProvider, node)
        if not isinstance(scope, cst.metadata.ClassScope) or not m.matches(
            node.body[0], m.AnnAssign(target=m.Name())
        ):
            return False

        try:
            is_required = m.matches(
                node.leading_lines[-1],
                m.EmptyLine(comment=m.Comment(f"# {REQUIRED_CLS_VAR}")),
            )
        except IndexError:
            is_required = False

        cls_var = ClsVar(
            cst.ensure_type(node.body[0], cst.AnnAssign),
            self._curr_modulename,
            self._curr_mro_index,
            len(self._cls_vars),
            is_required,
        )
        name = cst.ensure_type(cls_var.node.target, cst.Name).value
        self._cls_vars[name][self._curr_mro_index] = cls_var
        self._wrapper_stack.append(cls_var)
        return True

    def leave_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> None:
        scope = self.get_metadata(ScopeProvider, node)
        if not isinstance(scope, cst.metadata.ClassScope) or not m.matches(
            node.body[0], m.AnnAssign(target=m.Name())
        ):
            return
        self._wrapper_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        scope = self.get_metadata(ScopeProvider, node)
        if not isinstance(scope, cst.metadata.ClassScope):
            return False

        # We have hit a method that could need an implementation
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

        fn_def = FunctionDef(
            node,
            self._curr_modulename,
            self._curr_mro_index,
            len(self._fn_defs),
            is_property,
            is_abstract,
        )
        self._fn_defs[node.name.value][self._curr_mro_index] = fn_def
        self._wrapper_stack.append(fn_def)

        # Visit decorators, params and returns in order to associate accesses
        # with the current FunctionDef.
        for dec in node.decorators[:-1]:
            dec.visit(self)

        node.params.visit(self)
        if node.returns:
            node.returns.visit(self)

        return False

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        scope = self.get_metadata(ScopeProvider, node)
        if isinstance(scope, cst.metadata.ClassScope):
            self._wrapper_stack.pop()

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        return self._visit_attr_or_name(node)

    def visit_Name(self, node: cst.Name) -> bool:
        return self._visit_attr_or_name(node)

    def _visit_attr_or_name(self, node: cst.Attribute | cst.Name) -> bool:
        # The goal here is to associate accesses with a specific node.
        # Why? Because all accesses we are interested in belong to some class
        # scope, but we may very well not be interested in all accesses from
        # said scope.

        if not self._wrapper_stack:
            return False

        expr_context = self.get_metadata(ExpressionContextProvider, node, None)
        if expr_context != ExpressionContext.LOAD:
            return False

        scope = self.get_metadata(ScopeProvider, node)
        if scope is None:
            return False

        node_wrapper = self._wrapper_stack[-1]
        for access in scope.accesses[node]:
            if access.node is node:
                node_wrapper.accesses.append(access)

        # If it is an attribute we must descend
        return True
