import json
from dataclasses import dataclass
from typing import cast

from flwr.common import (
    Context,
    Message,
    MetricRecord,
    RecordDict,
)

from fedbench.core.algorithm import ComponentSpec, Synthesizer
from fedbench.core.data import PartitionedDataset
from fedbench.core.encoder import FedbenchEncoder
from fedbench.core.eval import EvaluationSuite, LocalEvalContext
from fedbench.core.logger import log_warning
from fedbench.core.update import Extras, Update
from fedbench.flwr.serde import (
    FlwrDeserializer,
    FlwrSerializer
)
from fedbench.runtime.component_factory import create_synthesizer


@dataclass(frozen=True)
class FlwrClient:
    dataset: PartitionedDataset
    synthesizer_spec: ComponentSpec[Synthesizer]
    synthesizer_artifacts: Update | None
    eval_suite: EvaluationSuite
    to_flwr: FlwrSerializer
    from_flwr: FlwrDeserializer

    def fed_init(
        self,
        seed: int,
        flwr_message: Message,
        flwr_context: Context,
    ) -> Message:

        partition_id = self._get_partition_id(flwr_context)
        train_df = self.dataset.load_train_partition(partition_id)
        synthesizer = create_synthesizer(
            spec=self.synthesizer_spec,
            artifacts=self.synthesizer_artifacts,
            client_cache=None  # TODO!
        )
        request = self.from_flwr(
            flwr_message,
            self.synthesizer_spec.arrays_to_ml_framework_map,
        )
        reply = synthesizer.fed_init(request, seed, self.dataset.schema, train_df)
        return self.to_flwr(update=reply, reply_to=flwr_message)

    def train(self, flwr_message: Message, flwr_context: Context) -> Message:
        partition_id = self._get_partition_id(flwr_context)
        train_df = self.dataset.load_train_partition(partition_id)
        synthesizer = create_synthesizer(
            spec=self.synthesizer_spec,
            artifacts=self.synthesizer_artifacts,
            client_cache=None  # TODO!
        )
        request = self.from_flwr(
            flwr_message,
            self.synthesizer_spec.arrays_to_ml_framework_map,
        )
        reply = synthesizer.train(request, train_df)
        return self.to_flwr(update=reply, reply_to=flwr_message)

    def evaluate(
        self,
        flwr_message: Message,
        flwr_context: Context,
        num_synthetic_rows: int | None,
        seed: int,
        target_column: str | None,
        sensitive_columns: tuple[str, ...] | None,
    ) -> Message:

        partition_id = self._get_partition_id(flwr_context)
        train_df = self.dataset.load_train_partition(partition_id)
        test_df = self.dataset.load_test_partition(partition_id)

        synthesizer = create_synthesizer(
            self.synthesizer_spec,
            self.synthesizer_artifacts,
            client_cache=None,
        )
        request = self.from_flwr(
            flwr_message,
            self.synthesizer_spec.arrays_to_ml_framework_map,
        )
        synthetic_df = synthesizer.sample(
            request,
            num_synthetic_rows or len(train_df),
            seed,
        )
        if synthetic_df.empty:
            log_warning(__name__, f"Recv empty sample from {synthesizer}.")
            return Message(
                content=RecordDict({"metrics": MetricRecord()}),
                reply_to=flwr_message,
            )
        eval_ctx = LocalEvalContext(
            train_df=train_df,
            test_df=test_df,
            synthetic_df=synthetic_df,
            seed=seed,
            target_column=target_column,
            sensitive_columns=sensitive_columns,
            schema=self.dataset.schema,
        )
        metrics: Extras = {}
        for key, value in self.eval_suite.local_evaluate(eval_ctx).items():
            metrics[key] = json.dumps(value, cls=FedbenchEncoder)

        update = Update(extras={"metrics": metrics})

        return self.to_flwr(
            update=update,
            reply_to=flwr_message,
        )

    @staticmethod
    def _get_partition_id(flwr_context: Context) -> int:
        # noinspection PyUnnecessaryCast
        return cast(int, flwr_context.node_config["partition-id"])
