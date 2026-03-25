from typing import Iterable, cast

import torch
from torch import Tensor

from fedbench.core.algorithm import SingleStepCoordinator
from fedbench.core.logger import ELBOW, log_warning
from fedbench.core.payload import Payload


class FedTabDiffCoordinator(SingleStepCoordinator):
    def __init__(self) -> None:
        self._state: dict[str, Tensor] | None = None

    @property
    def global_state(self) -> Payload | None:
        if self._state is None:
            return None
        return Payload(arrays={"state": self._state})

    def attach_global_init_artifacts(self, artifacts: Payload) -> None:
        # noinspection PyUnnecessaryCast
        self._state = cast(dict[str, Tensor], artifacts.arrays["initial-state"])

    def aggregate_train(self, replies: Iterable[tuple[int, Payload]]) -> None:
        if not replies:
            raise ValueError("No replies, can not aggregate.")

        num_samples: list[int] = []
        state_dicts: list[dict[str, Tensor]] = []

        for _, reply in replies:
            # noinspection PyUnnecessaryCast
            num_samples.append(cast(int, reply.metrics["metrics"]["num-samples"]))
            # noinspection PyUnnecessaryCast
            state_dicts.append(cast(dict[str, Tensor], reply.arrays["state"]))

        total = sum(num_samples)
        if total <= 0:
            log_warning(str(self), f"Total number of samples: {total}")
            log_warning("", f"\t{ELBOW} Skipping aggregation.")
            return

        weights = tuple(float(n) / total for n in num_samples)
        keys = tuple(state_dicts[0].keys())
        aggr_state: dict[str, Tensor] = {}

        with torch.no_grad():
            for key in keys:
                result: Tensor | None = None

                for state_dict, weight in zip(state_dicts, weights, strict=True):
                    tensor = state_dict[key].detach().cpu()
                    if result is None:
                        result = tensor * weight
                    else:
                        result = result + tensor * weight

                aggr_state[key] = result

        self._state = aggr_state
