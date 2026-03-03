import pickle
from typing import Protocol, cast

from flwr.common import (
    Message,
    Array,
    ArrayRecord,
    MetricRecord,
    ConfigRecord,
    RecordDict
)

from fedbench.core.update import Objects, Update


_METADATA_KEY = f"{__package__}.metadata"


class FlwrSerializer(Protocol):
    def __call__(
            self,
            update: Update,
            message_type: str | None = None,
            dst_node_id: int | None = None,
            reply_to: Message | None = None) -> Message:
        pass


class FlwrDeserializer(Protocol):
    def __call__(
            self,
            message: Message,
            arrays_to_ml_framework_map: dict[str, str] | None = None) -> Update:
        pass


def to_flwr_pickle(
        update: Update,
        message_type: str | None = None,
        dst_node_id: int | None = None,
        reply_to: Message | None = None) -> Message:

    if reply_to is None:
        if dst_node_id is None:
            raise ValueError("Either dst_node_id or reply_to is required.")

        if message_type is None:
            raise ValueError("message_type required when reply_to is None.")

    rdict = RecordDict()
    pickle_records = []

    for key, arrays in update.arrays.items():
        rdict[key] = ArrayRecord(arrays)

    for key, objects in update.objects.items():
        # noinspection PyUnnecessaryCast
        rdict[key] = ArrayRecord(_pickle_objects(objects))
        pickle_records.append(key)

    for key, metrics in update.metrics.items():
        rdict[key] = MetricRecord(metrics)

    for key, extras in update.extras.items():
        rdict[key] = ConfigRecord(extras)

    _inject_metadata(rdict, pickle_records)

    if reply_to is not None:
        return Message(content=rdict, reply_to=reply_to)

    # noinspection PyUnnecessaryCast
    return Message(
        content=rdict,
        message_type=cast(str, message_type),
        dst_node_id=cast(int, dst_node_id)
    )


def from_flwr_pickle(
        message: Message,
        arrays_to_ml_framework_map: dict[str, str] | None = None) -> Update:

    arrays_to_ml_framework_map = arrays_to_ml_framework_map or {}
    rdict = message.content
    update = Update()

    pickle_records = _extract_metadata(rdict)

    for key, arrays in rdict.array_records.items():
        if key in pickle_records:
            # noinspection PyUnnecessaryCast
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


def _pickle_objects(objects: Objects) -> dict[str, Array]:
    arrays = {}
    for key, value in objects.items():
        data = pickle.dumps(value)
        # Send the bytes as a 1d uint8 ndarray
        arr = Array(
            dtype="uint8",
            shape=(len(data),),
            stype="pickle",  # Will, and should make f.ex. arr.numpy() raise err
            data=data)
        arrays[key] = arr
    return arrays


def _unpickle_arrays(arrays: ArrayRecord) -> Objects:
    objects = {}
    for key, value in arrays.items():
        objects[key] = pickle.loads(value.data)
    return objects


def _inject_metadata(rdict: RecordDict, pickle_records: list[str]) -> None:
    cfg_record = ConfigRecord({"pickle-records": pickle_records})
    rdict.config_records[_METADATA_KEY] = cfg_record


def _extract_metadata(rdict: RecordDict) -> list[str]:
    cfg_record = rdict.config_records.pop(_METADATA_KEY, None)
    if cfg_record is None:
        return []
    # noinspection PyUnnecessaryCast
    return cast(list[str], cfg_record["pickle-records"])
