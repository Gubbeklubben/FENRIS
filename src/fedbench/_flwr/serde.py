import pickle
from enum import StrEnum
from typing import Any

import msgpack
from flwr.common import (
    Message,
    Array,
    ArrayRecord,
    MetricRecord,
    ConfigRecord,
    RecordDict
)

from fedbench.common import Update, Objects


class _NonArrayProtocol(StrEnum):
    PICKLE = "pickle"
    MSGPACK = "msgpack"


def _dumps(obj: Any, protocol: str) -> bytes:
    match protocol:
        case _NonArrayProtocol.PICKLE:
            return pickle.dumps(obj)
        case _NonArrayProtocol.MSGPACK:
            return msgpack.dumps(obj)
        case _:
            raise ValueError(f"Unsupported protocol: {protocol}")


def _loads(data: bytes, protocol: str) -> Any:
    match protocol:
        case _NonArrayProtocol.PICKLE:
            return pickle.loads(data)
        case _NonArrayProtocol.MSGPACK:
            return msgpack.loads(data)
        case _:
            raise ValueError(f"Unsupported protocol: {protocol}")


def objects_to_arrays(
        objects: Objects,
        protocol: str,
        allow_pickle: bool) -> dict[str, Array]:

    if protocol == _NonArrayProtocol.PICKLE and not allow_pickle:
        raise RuntimeError(f"allow_pickle is False, refusing to"
                           f"pickle {objects}")
    arrays = {}
    for key, value in objects.items():
        data = _dumps(value, protocol)
        # Send the bytes as a 1d uint8 ndarray
        arr = Array(
            dtype="uint8",
            shape=(len(data),),
            stype=protocol,  # Will, and should make f.ex. arr.numpy() raise err
            data=data)
        arrays[key] = arr
    return arrays


def arrays_to_objects(
        arrays: ArrayRecord,
        protocol: str,
        allow_pickle: bool) -> Objects:

    if protocol == _NonArrayProtocol.PICKLE and not allow_pickle:
        raise RuntimeError("allow_pickle is False, refusing to unpickle data")

    objects = {}
    for key, value in arrays.items():
        objects[key] = _loads(value.data, protocol)
    return objects


def to_flwr(
        update: Update,
        message_type: str | None = None,
        dst_node_id: int = None,
        reply_to: Message = None,
        non_array_protocol: str | None = None,
        allow_pickle: bool = False) -> Message:

    if reply_to is None:
        if dst_node_id is None:
            raise ValueError("Either dst_node_id or reply_to is required")

        if message_type is None:
            raise ValueError("message_type required when reply_to is None")

    if update.objects and not non_array_protocol:
        raise ValueError("non_array_protocol required to send non array objects")

    rdict = RecordDict()
    na_records = []

    for key, arrays in update.arrays.items():
        rdict[key] = ArrayRecord(arrays)

    for key, objects in update.objects.items():
        rdict[key] = ArrayRecord(
            objects_to_arrays(
                objects,
                non_array_protocol,
                allow_pickle)
        )
        na_records.append(key)

    for key, metrics in update.metrics.items():
        rdict[key] = MetricRecord(metrics)

    for key, extras in update.extras.items():
        rdict[key] = ConfigRecord(extras)

    if na_records:
        rdict[f"{__package__}.metadata"] = ConfigRecord({
            "na-proto": non_array_protocol,
            "na-records": na_records,
        })

    if reply_to is not None:
        return Message(content=rdict, reply_to=reply_to)

    return Message(
        message_type=message_type,
        dst_node_id=dst_node_id,
        content=rdict,
    )


def from_flwr(
        message: Message,
        arrays_decode_spec: dict[str, str] = None,
        allow_pickle: bool = False) -> Update:

    def get_arrays_decode_spec(k: str) -> str:
        if arrays_decode_spec is None:
            return "numpy"
        return arrays_decode_spec.get(k, "numpy")

    rdict = message.content
    update = Update()
    metadata = rdict.config_records.get(f"{__package__}.metadata", {})
    na_proto = metadata.get("na-proto", None)
    na_records = metadata.get("na-records", ())

    if na_records and na_proto is None:
        raise RuntimeError(f"Message contains non array records, but no"
                           f"corresponding protocol")

    for key, arrays in rdict.array_records.items():
        if key in na_records:
            objects = arrays_to_objects(
                arrays,
                na_proto,
                allow_pickle
            )
            update.objects[key] = objects
        else:
            decode_spec = get_arrays_decode_spec(key)
            if decode_spec == "torch":
                update.arrays[key] = arrays.to_torch_state_dict()
            else:
                update.arrays[key] = arrays.to_numpy_ndarrays()

    for key, metrics in rdict.metric_records.items():
        update.metrics[key] = dict(metrics)

    for key, extras in rdict.config_records.items():
        update.extras[key] = dict(extras)

    return update
