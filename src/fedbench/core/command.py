from typing import Protocol

from fedbench.core.runcontext import RunContext


class Command(Protocol):
    def __call__(self, ctx: RunContext) -> None:
        pass
