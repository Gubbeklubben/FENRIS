from dataclasses import dataclass
from typing import ClassVar, Self

import typer

from fedbench.cli.plugin._util import parse_identifier
from fedbench.runtime.registry import Group
from fedbench.scaffold.subclass_builder import AbstractMethodCollector, Builder


@dataclass(frozen=True)
class _Component:
    parser_syntax: ClassVar[str] = "group:[module.]*name"
    group: Group
    module: tuple[str, ...]
    class_name: str

    @property
    def name(self) -> str:
        return self.module[-1]

    @classmethod
    def parse(cls, arg: str) -> Self:
        try:
            group_keyword, module = arg.split(":", maxsplit=1)
        except ValueError:
            raise typer.BadParameter("Syntax error, missing ':'.")
        try:
            # Because Group has custom __new__ and __init__ with an additional arg,
            # mypy incorrectly thinks that __call__ also need that arg.
            group = Group(group_keyword)  # type: ignore[call-arg]
        except ValueError:
            raise typer.BadParameter(f"Invalid group: {group_keyword}.")

        module_parts = tuple(parse_identifier(m.lower()) for m in module.split("."))
        if len(module_parts) == 1:  # Enforce some structure
            module_parts = (group.value, module[0])
        class_name = parse_identifier(_to_cap_words(module_parts[-1]))

        return cls(group, module_parts, class_name)

    @classmethod
    def default(cls) -> Self:
        return cls(
            Group.SYNTHESIZERS,
            (Group.SYNTHESIZERS.value, "my_synthesizer"),
            "MySynthesizer",
        )


def codegen(component: _Component) -> str:
    collector = AbstractMethodCollector(component.group.base)
    builder = Builder(collector)
    return builder.with_name(component.class_name).build().code


def _to_cap_words(identifier: str) -> str:
    return "".join(w.capitalize() for w in identifier.split("_"))
