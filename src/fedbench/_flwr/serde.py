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

from fedbench.common import MessageContent


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


def _objects_to_arrays(
        objects: dict[str, Any],
        protocol: str,
        allow_pickle: bool) -> dict[str, Array]:

    if protocol == "pickle" and not allow_pickle:
        raise RuntimeError(
            f"'allow_pickle' is False, refusing to pickle {objects}"
        )
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


def _arrays_to_objects(
        arrays: ArrayRecord,
        protocol: str,
        allow_pickle: bool) -> dict[str, Any]:

    if protocol == "pickle" and not allow_pickle:
        raise RuntimeError("'allow_pickle' is False, refusing to unpickle data")

    objects = {}
    for key, value in arrays.items():
        objects[key] = _loads(value.data, protocol)
    return objects


def to_flwr_message(
        msg_content: MessageContent,
        message_type: str | None = None,
        dst_node_id: int = None,
        reply_to: Message = None,
        non_array_protocol: str | None = None,
        allow_pickle: bool = False) -> Message:

    if reply_to is None:
        if dst_node_id is None:
            raise ValueError("Either dst_node_id or reply_to is required")

        if message_type is None:
            raise ValueError("message_type required when reply_to is None'")

    rdict = RecordDict()
    na_records = []

    for key, arrays in msg_content.arrays.items():
        rdict[key] = ArrayRecord(arrays)

    for key, objects in msg_content.objects.items():
        if non_array_protocol is None:
            raise ValueError(
                "non_array_protocol is required when sending non array "
                "objects"
            )
        rdict[key] = ArrayRecord(
            _objects_to_arrays(objects, non_array_protocol, allow_pickle)
        )
        na_records.append(key)

    if na_records:
        rdict[f"{__package__}.metadata"] = ConfigRecord({
            "na-proto": non_array_protocol,
            "na-records": na_records,
        })

    for key, metrics in msg_content.metrics.items():
        rdict[key] = MetricRecord(metrics)

    for key, config in msg_content.config.items():
        rdict[key] = ConfigRecord(config)

    if reply_to is not None:
        return Message(content=rdict, reply_to=reply_to)

    return Message(
        message_type=message_type,
        dst_node_id=dst_node_id,
        content=rdict,
    )


def from_flwr_message(
        message: Message,
        arrays_decode_spec: dict[str, str] = None,
        allow_pickle: bool = False) -> MessageContent:

    def get_arrays_decode_spec(k: str) -> str:
        if arrays_decode_spec is None:
            return "numpy"
        return arrays_decode_spec.get(k, "numpy")

    rdict = message.content
    msg_content = MessageContent()

    try:
        metadata = rdict.config_records[f"{__package__}.metadata"]
    except KeyError:
        na_proto = None
        na_records = []
    else:
        na_proto = metadata["na-proto"]
        na_records = metadata["na-records"]

    for key, arrays in rdict.array_records.items():
        if key in na_records:
            objects = _arrays_to_objects(
                arrays,
                na_proto,
                allow_pickle
            )
            msg_content.add_objects(key, objects)
        else:
            decode_spec = get_arrays_decode_spec(key)
            if decode_spec == "torch":
                msg_content.add_arrays(key, arrays.to_torch_state_dict())
            else:
                msg_content.add_arrays(key, arrays.to_numpy_ndarrays())

    for key, metrics in rdict.metric_records.items():
        msg_content.add_metrics(key, dict(metrics))

    for key, config in rdict.config_records.items():
        msg_content.add_config(key, dict(config))

    return msg_content
