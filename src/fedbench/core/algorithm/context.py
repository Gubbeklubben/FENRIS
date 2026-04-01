from dataclasses import dataclass

from fedbench.core.data import TableSchema
from fedbench.core.payload import Payload


@dataclass(frozen=True)
class GlobalInitContext:
    schema: TableSchema
    seed: int


@dataclass(frozen=True)
class TrainContext:
    global_init_artifacts: Payload | None
    client_cache: Payload | None


@dataclass(frozen=True)
class SampleContext:
    global_init_artifacts: Payload | None
    client_cache: Payload | None
    schema: TableSchema
    seed: int
    num_rows: int
