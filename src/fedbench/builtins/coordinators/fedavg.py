from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self, cast

import torch

from fedbench.core.algorithm import SingleStepCoordinator
from fedbench.core.payload import Arrays, ArraysTarget, Payload, PayloadSchema


@dataclass(frozen=True)
class GlobalState:
    state: Arrays

    @classmethod
    def decode(cls, payload: Payload) -> Self:
        return cls(payload.arrays["state"])

    def encode(self) -> Payload:
        return Payload(arrays={"state": self.state})


@dataclass(frozen=True)
class ClientUpdate:
    state: Arrays
    count: int

    @classmethod
    def decode(cls, payload: Payload) -> Self:
        # noinspection PyUnnecessaryCast
        return cls(
            state=payload.arrays["state"],
            count=cast(int, payload.metrics["metrics"]["count"]),
        )

    def encode(self) -> Payload:
        return Payload(
            arrays={"state": self.state},
            metrics={"metrics": {"count": self.count}},
        )


class FedAvg(SingleStepCoordinator):
    def __init__(self, weighted: bool = True) -> None:
        self._weighted = weighted
        self._state: dict[str, torch.Tensor] | None = None

    @property
    def arrays_target(self) -> ArraysTarget:
        return ArraysTarget.TORCH

    @property
    def payload_schema(self) -> PayloadSchema:
        return MappingProxyType(
            {
                "global_state": GlobalState,
                "client_update": ClientUpdate,
            }
        )

    def attach_global_init_artifacts(self, artifacts: Payload) -> None:
        # noinspection PyUnnecessaryCast
        self._state = cast(dict[str, torch.Tensor], GlobalState.decode(artifacts).state)

    def configure_train(
        self, client_ids: Iterable[int]
    ) -> Iterable[tuple[int, Payload]]:

        if self._state is None:
            raise ValueError("No global state, can not configure training round.")

        for cid in client_ids:
            yield cid, GlobalState(self._state).encode()

    def aggregate_train(self, replies: Iterable[tuple[int, Payload]]) -> None:
        if not replies:
            raise ValueError("No replies, can not aggregate.")

        count: list[int] = []
        state_dicts: list[dict[str, torch.Tensor]] = []

        for _, payload in replies:
            update = ClientUpdate.decode(payload)
            count.append(update.count)
            # noinspection PyUnnecessaryCast
            state_dicts.append(cast(dict[str, torch.Tensor], update.state))

        total = sum(count)
        if total <= 0:
            raise ValueError(f"Total count: {count}, can not aggregate.")

        weights = tuple(float(n) / total for n in count)
        keys = tuple(state_dicts[0].keys())
        aggr_state: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for key in keys:
                result: torch.Tensor | None = None

                for state_dict, weight in zip(state_dicts, weights, strict=True):
                    tensor = state_dict[key].detach().cpu()
                    if result is None:
                        result = tensor * weight
                    else:
                        result = result + tensor * weight

                aggr_state[key] = result

        self._state = aggr_state

    def publish_train_artifacts(self) -> Payload:
        if self._state is None:
            raise ValueError("No global state, can not publish training artifacts.")
        return GlobalState(self._state).encode()
