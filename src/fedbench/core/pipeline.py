from typing import Protocol

from fedbench.core.runcontext import RunContext


class Command(Protocol):
    @property
    def __name__(self) -> str:
        pass

    def __call__(self, ctx: RunContext) -> None:
        pass
