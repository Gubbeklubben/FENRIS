from typing import Protocol

from fedbench.app.run.runcontext import RunContext


class Command(Protocol):
    def __call__(self, ctx: RunContext) -> None:
        pass
