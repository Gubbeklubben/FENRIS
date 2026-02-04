from flwr.common import ArrayRecord

from fedbench.common import MLRuntime, ModelState


def from_array_record(record: ArrayRecord, ml_runtime: MLRuntime) -> ModelState:
    match ml_runtime:
        case MLRuntime.NUMPY:
            return record.to_numpy_ndarrays()
        case MLRuntime.TORCH:
            return record.to_torch_state_dict()