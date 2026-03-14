import pickle
from abc import ABC, abstractmethod
from typing import Any

from flwr.common import (
    Array,
    ArrayRecord,
    ConfigRecord,
    MetricRecord,
    RecordDict,
)

from fedbench.core.update import Objects, Update


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
    def __init__(self, object_serde: ObjectSerde) -> None:
        self._object_serde = object_serde

    def to_flwr(self, update: Update) -> RecordDict:
        rdict = RecordDict()

        for key, arrays in update.arrays.items():
            rdict[key] = ArrayRecord(arrays)

        for key, objects in update.objects.items():
            rdict[key] = ArrayRecord(self._serialize_objects(objects))

        for key, metrics in update.metrics.items():
            rdict[key] = MetricRecord(metrics)

        for key, extras in update.extras.items():
            rdict[key] = ConfigRecord(extras)

        return rdict

    def from_flwr(
        self,
        rdict: RecordDict,
        arrays_to_ml_framework_map: dict[str, str] | None = None,
    ) -> Update:

        arrays_to_ml_framework_map = arrays_to_ml_framework_map or {}
        update = Update()

        for key, arrays in rdict.array_records.items():
            if list(arrays.values())[0].stype == self._object_serde.stype:
                update.objects[key] = self._deserialize_objects(arrays)
            else:
                ml_framework = arrays_to_ml_framework_map.get(key, "numpy")
                if ml_framework == "torch":
                    update.arrays[key] = arrays.to_torch_state_dict()
                else:
                    update.arrays[key] = arrays.to_numpy_ndarrays()

        for key, metrics in rdict.metric_records.items():
            update.metrics[key] = dict(metrics)

        for key, extras in rdict.config_records.items():
            update.extras[key] = dict(extras)

        return update

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
