from dataclasses import dataclass

from fenris.core.data import TableSchema
from fenris.core.payload import Payload


@dataclass(frozen=True)
class GlobalInitContext:
    """Context passed to `Synthesizer.global_init`.

    Attributes
    ----------
    schema : TableSchema
        Schema classifying the dataset that will be used for the run.
    seed : int
        Seed for stochastic operations during preprocessing.
    """

    schema: TableSchema
    seed: int


@dataclass(frozen=True)
class TrainContext:
    """Context passed to `Synthesizer.train`.

    Attributes
    ----------
    global_init_artifacts : Payload or None
        Synthesizer-side artifacts produced by `Synthesizer.global_init`, or
        ``None`` if global initialization produced no synthesizer artifacts.
    client_storage : Payload or None
        Persistent per-client key/value store. The synthesizer may read from
        and write to this across training rounds. ``None`` on the first round.
    seed : int
        Seed for stochastic operations during training.
    """

    global_init_artifacts: Payload | None
    client_storage: Payload | None
    seed: int


@dataclass(frozen=True)
class SampleContext:
    """Context passed to `Synthesizer.sample`.

    Attributes
    ----------
    global_init_artifacts : Payload or None
        Synthesizer-side artifacts produced by `Synthesizer.global_init`, or
        ``None`` if global initialization produced no synthesizer artifacts.
    client_storage : Payload or None
        Persistent per-client store accumulated during training, or ``None``
        if the synthesizer wrote nothing to storage.
    schema : TableSchema
        Schema of the table to synthesize.
    seed : int
        Seed for stochastic operations during sampling.
    num_rows : int
        Number of synthetic rows to generate.
    """

    global_init_artifacts: Payload | None
    client_storage: Payload | None
    schema: TableSchema
    seed: int
    num_rows: int
