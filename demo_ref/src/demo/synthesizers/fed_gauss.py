from typing import ClassVar, cast

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from fenris.builtins.coordinators import fedavg
from fenris.core.algorithm import (
    GlobalInitArtifacts,
    GlobalInitContext,
    SampleContext,
    Synthesizer,
    TrainContext,
)
from fenris.core.payload import ArraysTarget, Payload


class FedGauss(Synthesizer):
    SUPPORTED_COORDINATORS: ClassVar[set[str]] = {"fedavg"}

    def __init__(
        self,
        local_steps: int = 40,
        learning_rate: float = 0.05,
        clip: bool = False,
    ) -> None:
        if local_steps < 1:
            raise ValueError("Expecting local_steps >= 1.")
        if not 0 < learning_rate <= 0.5:
            raise ValueError("Expecting 0 < learning_rate <= 0.5.")
        self._local_steps = local_steps
        self._learning_rate = learning_rate
        self._clip = clip

    @property
    def arrays_target(self) -> ArraysTarget:
        return ArraysTarget.TORCH

    def global_init(
        self,
        df: DataFrame,
        context: GlobalInitContext,
    ) -> GlobalInitArtifacts:
        columns = [c.name for c in context.schema.columns]
        scaler = MinMaxScaler()
        scaler.fit(df[columns].to_numpy())

        d = len(columns)
        initial_state = {
            "mu": torch.full((d,), 0.5),
            "log_sigma": torch.zeros(d),
        }

        return GlobalInitArtifacts(
            coordinator=fedavg.GlobalState(initial_state).encode(),
            synthesizer=Payload(
                objects={"objects": {"scaler": scaler}},
            ),
        )

    def train(
        self,
        request: Payload,
        df: DataFrame,
        context: TrainContext,
    ) -> Payload:
        assert isinstance(context.global_init_artifacts, Payload)
        scaler = context.global_init_artifacts.objects["objects"]["scaler"]
        state = cast(dict[str, torch.Tensor], fedavg.GlobalState.decode(request).state)
        columns = [c.name for c in context.schema.columns]

        scaled = scaler.transform(df[columns].to_numpy())
        x = torch.tensor(scaled)

        mu = state["mu"].requires_grad_()
        log_sigma = state["log_sigma"].requires_grad_()
        optimizer = torch.optim.Adam([mu, log_sigma], lr=self._learning_rate)

        for _ in range(self._local_steps):
            optimizer.zero_grad()
            dist = torch.distributions.Normal(mu, log_sigma.exp())
            nll = -dist.log_prob(x).mean()
            nll.backward()
            optimizer.step()

        return fedavg.ClientUpdate(
            state={"mu": mu.detach(), "log_sigma": log_sigma.detach()},
            count=len(df),
        ).encode()

    def sample(
        self,
        request: Payload,
        context: SampleContext,
    ) -> DataFrame:
        assert isinstance(context.global_init_artifacts, Payload)
        scaler = context.global_init_artifacts.objects["objects"]["scaler"]
        state = cast(dict[str, torch.Tensor], fedavg.GlobalState.decode(request).state)
        columns = [c.name for c in context.schema.columns]

        mu = state["mu"]
        sigma = state["log_sigma"].exp()

        rng = np.random.default_rng(context.seed)
        z = rng.normal(mu, sigma, size=(context.num_rows, len(columns)))
        x = scaler.inverse_transform(z)

        if self._clip:
            x = np.clip(x, scaler.data_min_, scaler.data_max_)

        out = pd.DataFrame(np.asarray(x), columns=columns)

        for col in context.schema.columns:
            if col.kind == "integer":
                out[col.name] = out[col.name].round().astype(int)
            elif col.kind == "binary":
                out[col.name] = out[col.name].round().clip(0, 1).astype(int)

        return out
