import random
from typing import Iterable, Generator, cast

from pandas import DataFrame

from fedbench.core.algorithm import Algorithm, Synthesizer, Coordinator
from fedbench.core.logger import log_info, TEE, ELBOW
from fedbench.core.update import Update


class FedRandom(Algorithm):
    """A degenerate demo of multistep capabilities."""

    def create_coordinator(self) -> Coordinator:
        return FedRandomCoordinator()

    def create_synthesizer(self) -> Synthesizer:
        return FedRandomSynthesizer()


class FedRandomCoordinator(Coordinator):
    @property
    def global_state(self) -> Update | None:
        return Update(objects={"objects": {"state": object()}})

    def train(
            self,
            client_ids: Iterable[int]) -> Generator[Iterable[tuple[int, Update]],
                                                    Iterable[tuple[int, Update]],
                                                    None]:
        rnd = 0
        update = Update(extras={"federation": {"client_ids": list(client_ids)}})
        dst = next(iter(client_ids))

        while True:
            rnd += 1

            log_info(str(self), f"Begin round: {rnd}")
            replies = yield ((dst, update),)
            src, reply = next(iter(replies))

            if "abort" in reply.extras["federation"]:
                log_info(str(self), f"Recv abort fm client {dst}")
                return

            # noinspection PyUnnecessaryCast
            dst = cast(int, reply.extras["federation"]["dst"])
            # noinspection PyUnnecessaryCast
            message = cast(str, reply.extras["federation"]["message"])

            log_info(str(self), f"Forwarding message: {message}")
            log_info("", f"\t{TEE} From: {src}")
            log_info("", f"\t{TEE} To: {dst}")
            log_info("", f"\t{ELBOW} End round: {rnd}")

            update = Update(extras={
                "federation": {"client_ids": list(client_ids),
                               "message": message}
            })


class FedRandomSynthesizer(Synthesizer):
    def train(self, request: Update, data: DataFrame) -> Update:
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

        update = Update(extras={"federation": {"dst": dst, "message": message}})
        if abort:
            update.extras["federation"]["abort"] = abort
        return update

    def sample(self, request: Update, num_rows: int, seed: int) -> DataFrame:
        return DataFrame()