import inspect
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class _FunctionDefWrapper:
    node: cst.FunctionDef = field(repr=False)
    module: str
    is_property: bool
    is_abstract: bool
    mro_index: int
    method_index: int
    accesses: list[Access] = field(default_factory=list)


class SubclassBuilder(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (ScopeProvider, ExpressionContextProvider)

    def __init__(self, parent_cls: type) -> None:
        super().__init__()
        self._mro = tuple(c for c in parent_cls.mro() if c not in (ABC, object))
        self._curr_modulename: str = ""
        self._cls_stack: list[str] = []
        self._fn_stack: list[str] = []
        self._fn_defs: dict[str, list[_FunctionDefWrapper | None]] = defaultdict(
            lambda: [None for _ in self._mro]
        )

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

        wrapper = _FunctionDefWrapper(
            node,
            self._curr_modulename,
            is_property,
            is_abstract,
            self._curr_mro_index,
            len(self._fn_defs),
        )
        self._fn_defs[node.name.value][self._curr_mro_index] = wrapper

        # Visit decorators, params and returns in order to associate accesses
        # with the current FunctionDef.
        for dec in node.decorators[:-1]:
            dec.visit(self)

        node.params.visit(self)
        if node.returns:
            node.returns.visit(self)

        return False

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
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
        wrapper = self._fn_defs[self._fn_stack[-1]][self._curr_mro_index]

        for access in scope.accesses[node]:
            if access.node is node:
                wrapper.accesses.append(access)

        return True

    def build(self) -> cst.Module:
        self._maybe_parse_and_visit()

        new_imports: dict[str, set[str]] = defaultdict(set[str])
        imports: dict[cst.Import | cst.ImportFrom, set[str]] = defaultdict(set[str])
        fn_defs = []

        for name, wrappers in self._fn_defs.items():
            wrapper = next(w for w in wrappers if w is not None)
            if not wrapper.is_abstract:
                continue

            fn_defs.append(
                wrapper.node.with_changes(
                    decorators=wrapper.node.decorators[:-1],
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
                )
            )

            for access in wrapper.accesses:
                if len(access.referents) == 0:
                    raise NameError(access.node.value)

                if len(access.referents) > 1:
                    raise ValueError(f"Reference {access.node.value} is ambiguous.")

                assignment = next(iter(access.referents))
                if isinstance(assignment, BuiltinAssignment):
                    continue

                if isinstance(assignment, Assignment):
                    node = assignment.node
                    if isinstance(node, cst.Import | cst.ImportFrom):
                        try:
                            relative = node.relative  # ImportFrom
                        except AttributeError:
                            relative = False  # Import
                        if relative:
                            raise NotImplementedError(
                                "No support for resolving relative imports."
                            )
                        imports[node].add(assignment.name)
                    else:
                        new_imports[wrapper.module].add(assignment.name)

        print(new_imports)
        return cst.Module(
            body=[
                *[cst.SimpleStatementLine([imp]) for imp in imports],
                cst.ClassDef(
                    name=cst.Name(value="MyTest"),
                    bases=[cst.Arg(value=cst.Name(value=self.parent_cls.__name__))],
                    body=cst.IndentedBlock(body=[fn_def for fn_def in fn_defs]),
                    leading_lines=[cst.EmptyLine(), cst.EmptyLine()],
                ),
            ]
        )

    def _maybe_parse_and_visit(self):
        if self._curr_modulename:
            return

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


if __name__ == "__main__":
    from fedbench.core.algorithm import Coordinator, SingleStepCoordinator, Synthesizer

    print(SubclassBuilder(Synthesizer).build().code)
    print(SubclassBuilder(Coordinator).build().code)
    print(SubclassBuilder(SingleStepCoordinator).build().code)
