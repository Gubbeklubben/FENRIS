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
from fedbench.flwr.client.cache_manager import CacheManager, Namespace
from fedbench.flwr.client.context import build_client_context
from fedbench.runtime.component_factory import create_synthesizer

app = ClientApp()


def get_partition_id(flwr_context: Context) -> int:
    # noinspection PyUnnecessaryCast
    return cast(int, flwr_context.node_config["partition-id"])


@app.query("config")
def recv_config(message: Message, flwr_context: Context) -> Message:
    cache_mgr = CacheManager(flwr_context.state)
    cache_mgr.set_cache(Namespace.FRAMEWORK, message.content)

    return Message(content=RecordDict(), reply_to=message)


@app.query("artifacts")
def recv_artifacts(message: Message, flwr_context: Context) -> Message:
    cache_mgr = CacheManager(flwr_context.state)
    cache_mgr.set_cache(Namespace.GLOBAL_INIT_ARTIFACTS, message.content)
    return Message(content=RecordDict(), reply_to=message)


@app.query("fed_init")
def fed_init(message: Message, flwr_context: Context) -> Message:
    cache_mgr = CacheManager(flwr_context.state)
    ctx = build_client_context(cache_mgr)
    partition_id = get_partition_id(flwr_context)
    train_df = ctx.dataset.load_train_partition(partition_id)

    request = ctx.from_flwr(
        message.content,
        ctx.synthesizer_spec.arrays_to_ml_framework_map,
    )
    with ctx.use_synthesizer_cache(cache_mgr) as cache:
        synthesizer = create_synthesizer(
            spec=ctx.synthesizer_spec,
            artifacts=ctx.synthesizer_artifacts,
            client_cache=cache,
        )
        reply = synthesizer.fed_init(
            request, ctx.config.seed, ctx.dataset.schema, train_df
        )
    rdict = ctx.to_flwr(reply)
    return Message(content=rdict, reply_to=message)


@app.train()
def train(message: Message, flwr_context: Context) -> Message:
    cache_mgr = CacheManager(flwr_context.state)
    ctx = build_client_context(cache_mgr)
    partition_id = get_partition_id(flwr_context)
    train_df = ctx.dataset.load_train_partition(partition_id)

    request = ctx.from_flwr(
        message.content,
        ctx.synthesizer_spec.arrays_to_ml_framework_map,
    )
    with ctx.use_synthesizer_cache(cache_mgr) as cache:
        synthesizer = create_synthesizer(
            spec=ctx.synthesizer_spec,
            artifacts=ctx.synthesizer_artifacts,
            client_cache=cache,
        )
        start_time = time.perf_counter_ns()
        reply = synthesizer.train(request, train_df)
        local_train_seconds = (time.perf_counter_ns() - start_time) / 1e9

    rdict = ctx.to_flwr(reply)
    return Message(content=rdict, reply_to=message)


@app.evaluate()
def evaluate(message: Message, flwr_context: Context) -> Message:
    cache_mgr = CacheManager(flwr_context.state)
    ctx = build_client_context(cache_mgr)
    partition_id = get_partition_id(flwr_context)
    train_df = ctx.dataset.load_train_partition(partition_id)
    test_df = ctx.dataset.load_test_partition(partition_id)

    request = ctx.from_flwr(
        message.content,
        ctx.synthesizer_spec.arrays_to_ml_framework_map,
    )
    with ctx.use_synthesizer_cache(cache_mgr) as cache:
        synthesizer = create_synthesizer(
            ctx.synthesizer_spec,
            ctx.synthesizer_artifacts,
            client_cache=cache,
        )
        synthetic_df = synthesizer.sample(
            request,
            len(train_df),
            ctx.config.seed,
        )

    if synthetic_df.empty:
        log_warning(__name__, f"Recv empty sample from {synthesizer}.")
        return Message(
            content=RecordDict({"metrics": MetricRecord()}),
            reply_to=message,
        )

    eval_ctx = LocalEvalContext(
        train_df=train_df,
        test_df=test_df,
        synthetic_df=synthetic_df,
        seed=ctx.config.seed,
        target_column=ctx.config.data.target_col,
        sensitive_columns=ctx.config.data.sensitive_cols,
        schema=ctx.dataset.schema,
        local_train_seconds=math.nan,  # TODO: get from train()
    )

    metrics: Extras = {}
    for key, value in ctx.eval_suite.local_evaluate(eval_ctx).items():
        metrics[key] = json.dumps(value, cls=FedbenchEncoder)

    update = Update(extras={"metrics": metrics})
    rdict = ctx.to_flwr(update)
    return Message(content=rdict, reply_to=message)
