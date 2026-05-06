from dataclasses import dataclass

from fenris.core.data import TableSchema
from fenris.core.payload import Payload


@dataclass(frozen=True)
class GlobalInitContext:
    schema: TableSchema
    seed: int


@dataclass(frozen=True)
class TrainContext:
    global_init_artifacts: Payload | None
    client_storage: Payload | None
    seed: int


@dataclass(frozen=True)
class SampleContext:
    global_init_artifacts: Payload | None
    client_storage: Payload | None
    schema: TableSchema
    seed: int
    num_rows: int
