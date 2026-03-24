import json
import math
import time
from typing import cast

from flwr.clientapp import ClientApp
from flwr.common import Context, Message, MetricRecord, RecordDict

from fedbench.core.encoder import FedbenchEncoder
from fedbench.core.eval import LocalEvalContext
from fedbench.core.logger import log_warning
from fedbench.core.update import Extras, Update
from fedbench.flwr.client.context import build_client_context
from fedbench.flwr.namespace import Namespace
from fedbench.runtime.component_factory import create_synthesizer

app = ClientApp()


def get_partition_id(flwr_context: Context) -> int:
    # noinspection PyUnnecessaryCast
    return cast(int, flwr_context.node_config["partition-id"])


@app.query("configure")
def configure(message: Message, flwr_context: Context) -> Message:
    Namespace.FRAMEWORK.view(flwr_context.state).update(
        Namespace.FRAMEWORK.view(message.content),
    )
    Namespace.GLOBAL_INIT_ARTIFACTS.view(flwr_context.state).update(
        Namespace.GLOBAL_INIT_ARTIFACTS.view(message.content),
    )
    return Message(content=RecordDict(), reply_to=message)


@app.query("fed_init")
def fed_init(message: Message, flwr_context: Context) -> Message:
    ctx = build_client_context(flwr_context.state)
    partition_id = get_partition_id(flwr_context)
    train_df = ctx.dataset.load_train_partition(partition_id)

    artifacts = ctx.serde.from_flwr(ctx.artifacts_cache)
    request = ctx.serde.from_flwr(message.content)

    with ctx.serde.use_deserialized(ctx.synthesizer_cache) as cache:
        synthesizer = create_synthesizer(
            ctx.synthesizer_factory,
            artifacts=artifacts,
            client_cache=cache,
        )
        reply = synthesizer.fed_init(
            request, ctx.config.seed.init, ctx.dataset.schema, train_df
        )

    rdict = ctx.serde.to_flwr(reply)
    return Message(content=rdict, reply_to=message)


@app.train()
def train(message: Message, flwr_context: Context) -> Message:
    ctx = build_client_context(flwr_context.state)
    partition_id = get_partition_id(flwr_context)
    train_df = ctx.dataset.load_train_partition(partition_id)

    artifacts = ctx.serde.from_flwr(ctx.artifacts_cache)
    request = ctx.serde.from_flwr(message.content)

    with ctx.serde.use_deserialized(ctx.synthesizer_cache) as cache:
        synthesizer = create_synthesizer(
            ctx.synthesizer_factory,
            artifacts=artifacts,
            client_cache=cache,
        )
        start_time = time.perf_counter_ns()
        reply = synthesizer.train(request, train_df)
        train_seconds = (time.perf_counter_ns() - start_time) / 1e9

    with ctx.serde.use_deserialized(ctx.framework_cache) as cache:
        metrics = cast(
            dict[str, float],
            cache.metrics.setdefault(
                "metrics", {"local_train_seconds": 0, "local_train_rounds": 0}
            ),
        )
        metrics["local_train_seconds"] += train_seconds
        metrics["local_train_rounds"] += 1

    rdict = ctx.serde.to_flwr(reply)
    return Message(content=rdict, reply_to=message)


@app.evaluate()
def evaluate(message: Message, flwr_context: Context) -> Message:
    ctx = build_client_context(flwr_context.state)
    partition_id = get_partition_id(flwr_context)
    train_df = ctx.dataset.load_train_partition(partition_id)
    test_df = ctx.dataset.load_test_partition(partition_id)

    artifacts = ctx.serde.from_flwr(ctx.artifacts_cache)
    request = ctx.serde.from_flwr(message.content)

    with ctx.serde.use_deserialized(ctx.synthesizer_cache) as cache:
        synthesizer = create_synthesizer(
            ctx.synthesizer_factory,
            artifacts=artifacts,
            client_cache=cache,
        )
        synthetic_df = synthesizer.sample(
            request,
            len(train_df),
            ctx.config.seed.sampling,
        )

    if synthetic_df.empty:
        log_warning(__name__, f"Recv empty sample from {synthesizer}.")
        return Message(
            content=RecordDict({"metrics": MetricRecord()}),
            reply_to=message,
        )

    # noinspection PyUnnecessaryCast
    cached_metrics = cast(
        dict[str, float],
        ctx.serde.from_flwr(ctx.framework_cache).metrics.get("metrics", {}),
    )
    local_train_seconds = cached_metrics.get("local_train_seconds", math.nan)
    local_train_rounds = cached_metrics.get("local_train_rounds", math.nan)

    eval_ctx = LocalEvalContext(
        train_df=train_df,
        test_df=test_df,
        synthetic_df=synthetic_df,
        seed=ctx.config.seed.evaluation,
        target_column=ctx.config.data.target_col,
        sensitive_columns=ctx.config.data.sensitive_cols,
        schema=ctx.dataset.schema,
        local_train_seconds=local_train_seconds / local_train_rounds,
    )

    metrics: Extras = {}
    for key, value in ctx.eval_suite.local_evaluate(eval_ctx).items():
        try:
            metrics[key] = json.dumps(value, cls=FedbenchEncoder)
        except Exception as e:
            raise ValueError(
                f"Could not encode metric {key} with value {value}."
            ) from e

    update = Update(extras={"metrics": metrics})
    rdict = ctx.serde.to_flwr(update)
    return Message(content=rdict, reply_to=message)
