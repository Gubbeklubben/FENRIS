import math
from typing import Callable

from fedbench.config import MetricsConfig
from fedbench.core.logger import log_debug, log_error, log_info
from fedbench.core.payload import Payload


class EarlyStoppingMonitor:
    def __init__(self, config: MetricsConfig, evaluate_fn: Callable[[Payload], float]):
        self._config = config
        self._evaluate_fn = evaluate_fn

        self._early_stop_triggered = False
        self._patience_counter = 0
        self._nan_counter = 0
        self._best_value = math.inf if self._config.stop_mode == "min" else -math.inf

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @property
    def early_stop_triggered(self) -> bool:
        return self._early_stop_triggered

    def should_run(self, current_round: int, num_rounds: int) -> bool:
        """True if this round should compute the stopping metric."""

        # Check if early stopping is enabled
        if not self._config.early_stop:
            return False

        # If this is the last round, we are stopping anyway, no need to check
        if current_round == num_rounds:
            return False

        # Don't run until the configured minimum number of rounds
        if current_round < self._config.stop_min_rounds:
            return False

        # Evaluate as soon as stop_min_rounds is reached,
        # even if stop_min_rounds is not a multiple of stop_eval_every
        rounds_since_first_run = current_round - self._config.stop_min_rounds
        return rounds_since_first_run % self._config.stop_eval_every == 0

    def run(self, aggregated_state: Payload) -> None:
        value = self._evaluate_fn(aggregated_state)

        if math.isnan(value):
            # NaN counts as no improvement, but neither increments nor resets patience
            log_debug(
                str(self), "Stop metric is NaN - keeping patience counter unchanged."
            )
            self._nan_counter += 1
        elif self._is_improvement(value):
            self._best_value = value
            self._patience_counter = 0
            self._nan_counter = 0
            log_debug(
                str(self),
                "Stop metric improved - resetting patience counter.",
            )
        else:
            self._patience_counter += 1
            self._nan_counter = 0
            log_debug(
                str(self),
                "Stop metric did not improve significantly"
                " - incrementing patience counter.",
            )

        log_info(
            str(self),
            f"Stop metric {self._config.stop_metric} = {value}, "
            f"patience {self._patience_counter}/{self._config.stop_patience}",
        )

        if self._nan_counter >= 5:
            self._early_stop_triggered = True
            log_error(
                str(self),
                "Stop metric evaluated to NaN 5 times in a row - interrupting run.",
            )
            return

        if self._patience_counter >= self._config.stop_patience:
            self._early_stop_triggered = True
            log_info(str(self), "Early stop triggered.")

    def _is_improvement(self, value: float) -> bool:
        log_debug(
            str(self),
            f"Checking stop metric improvement: "
            f"stop_mode={self._config.stop_mode}, "
            f"value={value}, "
            f"best_value={self._best_value}, "
            f"stop_epsilon={self._config.stop_epsilon}",
        )
        if self._config.stop_mode == "min":
            return value < self._best_value - self._config.stop_epsilon
        else:
            return value > self._best_value + self._config.stop_epsilon
