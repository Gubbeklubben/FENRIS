from dataclasses import dataclass
from typing import ClassVar, Self

import typer

from fedbench.runtime.registry import Group
from fedbench.scaffold.subclass_builder import AbstractMethodCollector, Builder


@dataclass(frozen=True)
class _Component:
    arg_syntax: ClassVar[str] = "group:identifier"
    group: Group
    identifier: str

    @classmethod
    def parse(cls, arg: str) -> Self:
        try:
            group_keyword, identifier = arg.split(":")
        except ValueError:
            raise typer.BadParameter("Syntax error, missing ':'.")
        try:
            # Because Group has custom __new__ and __init__ with an additional arg,
            # mypy incorrectly thinks that __call__ also need that arg.
            group = Group(group_keyword)  # type: ignore[call-arg]
        except ValueError:
            raise typer.BadParameter(f"Invalid group: {group_keyword}.")

        if not identifier.isidentifier():
            raise typer.BadParameter(
                f"Syntax error, '{identifier}' is not a valid identifier."
            )
        return cls(group, identifier)

    @classmethod
    def default(cls) -> Self:
        return cls(Group.SYNTHESIZERS, "my_synthesizer")


def fully_qualified_name(project_name: str, component: _Component) -> str:
    return f"{module_name(project_name, component)}:{class_name(component)}"


def module_name(project_name: str, component: _Component) -> str:
    return f"{project_name}.{component.group.value}.{component.identifier.lower()}"


def class_name(component: _Component) -> str:
    return "".join(w.capitalize() for w in component.identifier.split("_"))


def codegen(component: _Component) -> str:
    collector = AbstractMethodCollector(component.group.base)
    builder = Builder(collector)
    return builder.with_name(class_name(component)).build().code
