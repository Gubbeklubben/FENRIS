from typing import Protocol

from fenris.app.run.runcontext import RunContext


class Command(Protocol):
    def __call__(self, ctx: RunContext) -> None:
        pass
