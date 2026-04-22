import pickle
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, overload

from flwr.app import Array, ArrayRecord, ConfigRecord, MetricRecord, RecordDict

from fedbench.core.payload import ArraysTarget, Objects, Payload
from fedbench.flwr.rdict import RDictNamespaceView


def count_rdict_bytes(rdict: RecordDict) -> int:
    """
    Count the uncompressed model-parameter payload bytes in a RecordDict.

    Counts only array_records (which carry both real tensors and
    pickle-serialized objects). Excludes metric_records and config_records,
    which carry only scalar values and JSON strings, not model parameters.

    Each Array.data is the raw bytes as stored; shape[0] equals len(data)
    for both numpy arrays (raw buffer) and pickle objects (uint8 encoding).
    """
    total = 0
    for record in rdict.array_records.values():
        for arr in record.values():
            total += len(arr.data)
    return total


class ObjectSerde(ABC):
    """Serialize/deserialize the contents of a payload's objects attribute."""

    @property
    @abstractmethod
    def stype(self) -> str:
        """String indicating the serialization mechanism."""
        pass

    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        pass


class Pickle(ObjectSerde):
    def __init__(self, disabled: bool = False) -> None:
        self._disabled = disabled

    @property
    def stype(self) -> str:
        return "pickle"

    def serialize(self, obj: Any) -> bytes:
        if self._disabled:
            raise TypeError(
                f"Pickle disabled, can not serialize object of type {type(obj)}."
            )
        return pickle.dumps(obj)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)


class FlwrSerde:
    def __init__(
        self,
        object_serde: ObjectSerde,
        default_arrays_target: ArraysTarget | None = None,
    ) -> None:

        self._object_serde = object_serde
        self._default_arrays_target = default_arrays_target or ArraysTarget.NUMPY

    @overload
    def to_flwr(self, payload: Payload) -> RecordDict: ...

    @overload
    def to_flwr(self, payload: Payload, target: None) -> RecordDict: ...

    @overload
    def to_flwr[T: (RecordDict, RDictNamespaceView)](
        self, payload: Payload, target: T
    ) -> T: ...

    def to_flwr(
        self, payload: Payload, target: RecordDict | RDictNamespaceView | None = None
    ) -> RecordDict | RDictNamespaceView:

        out = target if target is not None else RecordDict()

        for key, arrays in payload.arrays.items():
            out[key] = ArrayRecord(arrays)

        for key, objects in payload.objects.items():
            out[key] = ArrayRecord(self._serialize_objects(objects))

        for key, metrics in payload.metrics.items():
            out[key] = MetricRecord(metrics)

        for key, extras in payload.extras.items():
            out[key] = ConfigRecord(extras)

        return out

    def from_flwr(
        self,
        rdict: RecordDict | RDictNamespaceView,
        arrays_target: ArraysTarget | None = None,
    ) -> Payload:

        arrays_target = arrays_target or self._default_arrays_target
        payload = Payload()

        for key, arrays in rdict.array_records.items():
            if not arrays:
                payload.arrays[key] = {}
            elif list(arrays.values())[0].stype == self._object_serde.stype:
                payload.objects[key] = self._deserialize_objects(arrays)
            else:
                if arrays_target == ArraysTarget.TORCH:
                    payload.arrays[key] = arrays.to_torch_state_dict()
                else:
                    payload.arrays[key] = arrays.to_numpy_ndarrays()

        for key, metrics in rdict.metric_records.items():
            payload.metrics[key] = dict(metrics)

        for key, extras in rdict.config_records.items():
            payload.extras[key] = dict(extras)

        return payload

    @contextmanager
    def use_deserialized(
        self,
        target: RecordDict | RDictNamespaceView,
        arrays_target: ArraysTarget | None = None,
    ) -> Generator[Payload, None, None]:

        payload = self.from_flwr(target, arrays_target)
        try:
            yield payload
        finally:
            target.update(self.to_flwr(payload))

    def _serialize_objects(self, objects: Objects) -> dict[str, Array]:
        arrays = {}
        for key, value in objects.items():
            data = self._object_serde.serialize(value)
            # Send the bytes as a 1d uint8 ndarray
            arr = Array(
                dtype="uint8",
                shape=(len(data),),
                stype=self._object_serde.stype,
                # raise err
                data=data,
            )
            arrays[key] = arr
        return arrays

    def _deserialize_objects(self, arrays: ArrayRecord) -> Objects:
        objects = {}
        for key, value in arrays.items():
            objects[key] = self._object_serde.deserialize(value.data)
        return objects
