import pickle
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, overload

from flwr.app import (
    Array,
    ArrayRecord,
    ConfigRecord,
    MetricRecord,
    RecordDict,
)

from fedbench.core.update import Objects, Update
from fedbench.flwr.rdict import RDictNamespaceView


class ObjectSerde(ABC):
    """Serialize/deserialize the contents of an Update's objects attribute."""

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
        default_arrays_map: dict[str, str] | None = None,
    ) -> None:

        self._object_serde = object_serde
        self._default_arrays_map = default_arrays_map or {}

    @overload
    def to_flwr(self, update: Update) -> RecordDict: ...

    @overload
    def to_flwr(self, update: Update, target: None) -> RecordDict: ...

    @overload
    def to_flwr[T: (RecordDict, RDictNamespaceView)](
        self, update: Update, target: T
    ) -> T: ...

    def to_flwr(
        self, update: Update, target: RecordDict | RDictNamespaceView | None = None
    ) -> RecordDict | RDictNamespaceView:

        out = target if target is not None else RecordDict()

        for key, arrays in update.arrays.items():
            out[key] = ArrayRecord(arrays)

        for key, objects in update.objects.items():
            out[key] = ArrayRecord(self._serialize_objects(objects))

        for key, metrics in update.metrics.items():
            out[key] = MetricRecord(metrics)

        for key, extras in update.extras.items():
            out[key] = ConfigRecord(extras)

        return out

    def from_flwr(
        self,
        rdict: RecordDict | RDictNamespaceView,
        arrays_to_ml_framework_map: dict[str, str] | None = None,
    ) -> Update:

        arrays_map = arrays_to_ml_framework_map or self._default_arrays_map
        update = Update()

        for key, arrays in rdict.array_records.items():
            if list(arrays.values())[0].stype == self._object_serde.stype:
                update.objects[key] = self._deserialize_objects(arrays)
            else:
                ml_framework = arrays_map.get(key, "numpy")
                if ml_framework == "torch":
                    update.arrays[key] = arrays.to_torch_state_dict()
                else:
                    update.arrays[key] = arrays.to_numpy_ndarrays()

        for key, metrics in rdict.metric_records.items():
            update.metrics[key] = dict(metrics)

        for key, extras in rdict.config_records.items():
            update.extras[key] = dict(extras)

        return update

    @contextmanager
    def use_deserialized(
        self,
        target: RecordDict | RDictNamespaceView,
        arrays_to_ml_framework_map: dict[str, str] | None = None,
    ) -> Generator[Update, None, None]:

        update = self.from_flwr(target, arrays_to_ml_framework_map)
        try:
            yield update
        finally:
            target.update(self.to_flwr(update))

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
