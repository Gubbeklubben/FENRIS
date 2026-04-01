from __future__ import annotations

import inspect
from collections.abc import Iterable
from pathlib import Path

import libcst as cst

from fedbench.core.component import Component


class Collector(cst.CSTVisitor):
    def __init__(self, cls: type[Component]) -> None:
        super().__init__()
        if not issubclass(cls, Component):
            raise TypeError(f"{cls.__name__} is not a subclass of Component")

        self._cls = cls
        self._parents = []

        for c in self._cls.mro():
            if c is not self._cls:
                self._parents.append(c)
            if c is Component:
                break

        self._visit_classes = {c.__name__ for c in (self._cls, *self._parents)}
        self._imports: set[cst.Import | cst.ImportFrom] = set()
        self._abstract: dict[str, cst.FunctionDef] = {}
        self._concrete: dict[str, cst.FunctionDef] = {}
        self._required_refs: set[str] = set()
        self._collected: bool = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._cls})"

    def visit_Import(self, node: cst.Import) -> bool:
        self._imports.add(node)
        return True

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        self._imports.add(node)
        return True

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        return node.name.value in self._visit_classes

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        for dec in node.decorators:
            if dec.decorator.value == "abstractmethod":
                self._abstract[node.name.value] = node
                return True

        self._concrete[node.name.value] = node
        return False

    def visit_Name(self, node: cst.Name) -> bool:
        self._required_refs.add(node.value)
        return True

    def build_tree(self, name: str) -> cst.Module:
        if not self._collected:
            for src_file in (
                inspect.getsourcefile(c) for c in (self._cls, *self._parents)
            ):
                with Path(src_file).resolve().open() as f:
                    code = f.read()
                    tree = cst.parse_module(code)
                    tree.visit(self)
            self._collected = True

        return cst.Module(
            body=[
                *(
                    cst.SimpleStatementLine(body=[imp])
                    for imp in self._required_imports()
                ),
                cst.EmptyLine(),
                cst.ClassDef(
                    name=cst.Name(value=name),
                    bases=[cst.Arg(value=cst.Name(value=self._cls.__name__))],
                    body=cst.IndentedBlock(body=[*self._required_methods()]),
                ),
            ]
        )

    def _required_imports(self) -> Iterable[cst.Import | cst.ImportFrom]:
        for imp in self._imports:
            aliases = []
            for alias in imp.names:
                if alias.name.value not in self._required_refs:
                    continue
                aliases.append(alias)

            if aliases:
                match imp:
                    case cst.ImportFrom():
                        yield cst.ImportFrom(module=imp.module, names=aliases)
                    case cst.Import():
                        yield cst.Import(names=aliases)

    def _required_methods(self) -> Iterable[cst.FunctionDef]:
        for name in self._abstract:
            if name not in self._concrete:
                yield self._abstract[name]


def scaffold(name: str, cls: type[Component]) -> str:
    collector = Collector(cls)
    tree = collector.build_tree(name)
    # transform: remove decorators, sort imports and methods...
    return tree.code


if __name__ == "__main__":
    from fedbench.core.algorithm import SingleStepCoordinator

    print(scaffold("MyCoord", SingleStepCoordinator))
