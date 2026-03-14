import pickle
from typing import Protocol

from flwr.common import (
    Array,
    ArrayRecord,
    ConfigRecord,
    MetricRecord,
    RecordDict,
)

from fedbench.core.update import Objects, Update


class FlwrSerializer(Protocol):
    def __call__(self, update: Update) -> RecordDict:
        pass


class FlwrDeserializer(Protocol):
    def __call__(
        self,
        rdict: RecordDict,
        arrays_to_ml_framework_map: dict[str, str] | None = None,
    ) -> Update:
        pass


def to_flwr_pickle(update: Update) -> RecordDict:
    rdict = RecordDict()

    for key, arrays in update.arrays.items():
        rdict[key] = ArrayRecord(arrays)

    for key, objects in update.objects.items():
        # noinspection PyUnnecessaryCast
        rdict[key] = ArrayRecord(_pickle_objects(objects))

    for key, metrics in update.metrics.items():
        rdict[key] = MetricRecord(metrics)

    for key, extras in update.extras.items():
        rdict[key] = ConfigRecord(extras)

    return rdict


def from_flwr_pickle(
    rdict: RecordDict,
    arrays_to_ml_framework_map: dict[str, str] | None = None,
) -> Update:

    arrays_to_ml_framework_map = arrays_to_ml_framework_map or {}
    update = Update()

    for key, arrays in rdict.array_records.items():
        if list(arrays.values())[0].stype == "pickle":
            objects = _unpickle_arrays(arrays)
            update.objects[key] = objects
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


def to_flwr_no_pickle(update: Update) -> RecordDict:
    if update.objects:
        raise RuntimeError(
            "Pickle is disabled, but update has non-empty objects field."
        )
    return to_flwr_pickle(update)


def _pickle_objects(objects: Objects) -> dict[str, Array]:
    arrays = {}
    for key, value in objects.items():
        data = pickle.dumps(value)
        # Send the bytes as a 1d uint8 ndarray
        arr = Array(
            dtype="uint8",
            shape=(len(data),),
            stype="pickle",  # Will, and should make f.ex. arr.numpy() raise err
            data=data,
        )
        arrays[key] = arr
    return arrays


def _unpickle_arrays(arrays: ArrayRecord) -> Objects:
    objects = {}
    for key, value in arrays.items():
        objects[key] = pickle.loads(value.data)
    return objects
