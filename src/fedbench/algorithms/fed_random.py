import random
from typing import Generator, Iterable, cast

from pandas import DataFrame

from fedbench.core.algorithm import (
    Algorithm,
    Coordinator,
    Synthesizer,
    coordinator_spec,
    synthesizer_spec,
)
from fedbench.core.logger import ELBOW, TEE, log_info
from fedbench.core.payload import Payload


class FedRandomCoordinator(Coordinator):
    @property
    def global_state(self) -> Payload | None:
        return Payload(objects={"objects": {"state": object()}})

    def train(
        self, client_ids: Iterable[int]
    ) -> Generator[
        Iterable[tuple[int, Payload]],
        Iterable[tuple[int, Payload]],
        None,
    ]:
        rnd = 0
        update = Payload(extras={"federation": {"client_ids": list(client_ids)}})
        dst = next(iter(client_ids))

        while True:
            rnd += 1

            log_info(str(self), "")
            log_info("", f"\t{TEE} Begin internal round: {rnd}")

            replies = yield ((dst, update),)
            src, reply = next(iter(replies))

            if "abort" in reply.extras["federation"]:
                log_info("", f"\t{ELBOW} Recv abort fm client {dst}")
                return

            # noinspection PyUnnecessaryCast
            dst = cast(int, reply.extras["federation"]["dst"])
            # noinspection PyUnnecessaryCast
            message = cast(str, reply.extras["federation"]["message"])

            log_info("", f"\t{TEE} Forwarding message: {message}")
            log_info("", f"\t{TEE} From: {src}")
            log_info("", f"\t{TEE} To: {dst}")
            log_info("", f"\t{ELBOW} End internal round: {rnd}")

            update = Payload(
                extras={
                    "federation": {
                        "client_ids": list(client_ids),
                        "message": message,
                    }
                }
            )


class FedRandomSynthesizer(Synthesizer):
    def train(self, request: Payload, data: DataFrame) -> Payload:
        try:
            # noinspection PyUnnecessaryCast
            message = cast(str, request.extras["federation"]["message"])
        except KeyError:
            pass
        else:
            log_info(str(self), f"Recv message {message}")

        # noinspection PyUnnecessaryCast
        client_ids = cast(list[int], request.extras["federation"]["client_ids"])
        dst = random.choice(client_ids)
        message = random.choice(("Hello!", "Hi!", "What?", "Ok"))
        abort = random.choice((True, False, False, False, False))

        update = Payload(extras={"federation": {"dst": dst, "message": message}})
        if abort:
            update.extras["federation"]["abort"] = abort
        return update

    def sample(self, request: Payload, num_rows: int, seed: int) -> DataFrame:
        return DataFrame()


class FedRandom(Algorithm):
    """A degenerate demo of multistep capabilities."""

    coordinator_spec = coordinator_spec(FedRandomCoordinator)
    synthesizer_spec = synthesizer_spec(FedRandomSynthesizer)
