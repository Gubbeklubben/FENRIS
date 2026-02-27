from collections.abc import Iterable
from types import MappingProxyType

import numpy as np
import torch
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

from fedbench.core.algorithm import Algorithm, Synthesizer, Aggregator
from fedbench.core.data import TableSchema
from fedbench.core.update import Update, Extras
from .config import config
from .mlpsynth import MLPSynthesizer


def _split_cat_num(schema: TableSchema) -> tuple[list[str], list[str]]:
    cat_attrs = [
        c.name for c in schema.columns
        if c.kind in ("categorical", "binary")
    ]
    num_attrs = [
        c.name for c in schema.columns
        if c.kind in ("continuous", "integer")
    ]
    return cat_attrs, num_attrs


def _prefix_columns(df: DataFrame, cat_attrs: Iterable[str]) -> None:
    for cat_attr in cat_attrs:
        df[cat_attr] = cat_attr + "_" + df[cat_attr].astype("str")


class FedTabDiff(Algorithm):
    def create_aggregator(self) -> Aggregator:
        return FedTabDiffAggregator(config)

    def create_synthesizer(self) -> Synthesizer:
        return FedTabDiffSynthesizer(config)


class FedTabDiffAggregator(Aggregator):
    def __init__(self, cfg):
        self._cfg = cfg
        self._cat_attrs = None
        self._num_attrs = None
        self._client_preproc_objects = None
        self._client_preproc_extras = None
        self._init_params = None

    def configure_init(
            self,
            seed: int,
            schema: TableSchema,
            client_ids: Iterable[int]) -> Iterable[tuple[int, Update]]:

        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self._cat_attrs, self._num_attrs = _split_cat_num(schema)

        initialize_qt: bool = True
        # TODO! Framework must somehow guarantee reproducible mapping to
        #  partition_id if this is to be reliable.
        for cid in client_ids:
            update = Update()
            cfg = {"initialize-qt": initialize_qt}
            update.extras["config"] = cfg
            initialize_qt = False  # Only one client
            yield cid, update


    def aggregate_init(self, replies: Iterable[Update]) -> Update:
        vocab_classes = set()
        num_scaler = None

        for reply in replies:
            preproc_obj = reply.objects["preproc-objects"]
            if "num-scaler" in preproc_obj:
                num_scaler = preproc_obj["num-scaler"]

            extras = reply.extras["preproc-extras"]
            vocab_classes = vocab_classes | set(extras["vocab-classes"])

        vocab_classes = sorted(vocab_classes)
        label_encoder = LabelEncoder().fit(vocab_classes)

        vocab_per_attr = {}
        for col in self._cat_attrs:
            prefix = f"{col}_"
            tokens = tuple(t for t in vocab_classes if t.startswith(prefix))
            if tokens:
                ids = label_encoder.transform(tokens)
                vocab_per_attr[col] = set(ids)
            else:
                vocab_per_attr = set()

        objects = {
            "num-scaler": num_scaler,
            "label-encoder": label_encoder
        }
        self._client_preproc_objects = MappingProxyType(objects)

        extras = {
            "n-cat-tokens": len(vocab_classes),
            "cat-dim": self._cfg["n_cat_emb"] * len(self._cat_attrs),
            "encoded-dim": self._cfg["n_cat_emb"] * len(self._cat_attrs) +
                           len(self._num_attrs),
            "vocab-per-attr": list(vocab_per_attr)
        }
        self._client_preproc_extras = MappingProxyType(extras)

        synthesizer = MLPSynthesizer(
            d_in=extras["encoded-dim"],
            hidden_layers=self._cfg["mlp_layers"],
            activation=self._cfg["activation"],
            n_cat_tokens=extras["n-cat-tokens"],
            n_cat_emb=self._cfg["n_cat_emb"],
            n_classes=None,
            embedding_learned=False
        )
        self._init_params = synthesizer.state_dict()
        return self._create_update()

    def aggregate_train(self, replies: Iterable[Update]) -> Update:
        return self._create_update()

    def _create_update(self) -> Update:
        return Update(
            arrays={"arrays": self._init_params},
            objects={"preproc-objects": self._client_preproc_objects},
            extras={"preproc-extras": self._client_preproc_extras}
        )


class FedTabDiffSynthesizer(Synthesizer):
    @property
    def arrays_to_ml_framework_map(self) -> dict[str, str] | None:
        return {"arrays": "torch"}

    def __init__(self, cfg):
        self._cfg = cfg
        self._device = cfg["device"]
        self._client_rounds = cfg["client_rounds"]

    def init(
            self,
            request: Update,
            seed: int,
            schema: TableSchema,
            data: DataFrame) -> Update:

        cat_attrs, num_attrs = _split_cat_num(schema)
        _prefix_columns(data, cat_attrs)

        # Aggregator has no data access, pick the simplest possible route
        # and let one client perform the task performed centrally in
        # https://github.com/sattarov/FedTabDiff/blob/main/main.py
        cfg: Extras = request.extras["config"]
        initialize_qt: bool = cfg["initialize-qt"]

        if initialize_qt:
            num_scaler = QuantileTransformer(
                output_distribution="normal",
                random_state=seed
            )
            print(35 * " ", f"num_attrs: {num_attrs}")
            num_scaler.fit(data[num_attrs])
        else:
            num_scaler = None

        vocab_classes = list(np.unique(data[cat_attrs]))

        update = Update()
        update.extras["preproc-extras"] = {
            "vocab-classes": vocab_classes
        }
        if num_scaler is not None:
            # Pickle it for now
            update.objects["preproc-objects"] = {
                "num-scaler": num_scaler
            }
        return update

    def train(self, request: Update, data: DataFrame) -> Update:
        arrays = request.arrays["arrays"]
        preproc = request.extras["preprocessing-results"]
        n_samples = len(data)

        mlp_synth, diffuser = init_model(self._cfg | preproc)
        set_parameters(mlp_synth, arrays)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, mlp_synth.parameters()),
            lr=cfg["learning_rate"]
        )

        # init loss function
        loss_fnc = nn.MSELoss()
        total_losses = []
        rnd = 0

        # iterate over distinct mini-batches
        for _, (batch_cat, batch_num, batch_y) in enumerate(train_loader):

            # set network in training mode
            mlp_synth.train()
            mlp_synth.to(self._device)

            # move batch to device
            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            batch_y = batch_y.to(device)

            # sample timestamps t
            timesteps = diffuser.sample_timesteps(n=batch_cat.shape[0])

            # get cat embeddings
            batch_cat_emb = mlp_synth.embed_categorical(x_cat=batch_cat)

            # concat cat & num
            batch_cat_num = torch.cat((batch_cat_emb, batch_num), dim=1)

            # add noise
            batch_noise_t, noise_t = diffuser.add_gauss_noise(
                x_num=batch_cat_num,
                t=timesteps)

            # conduct forward encoder/decoder pass
            predicted_noise = mlp_synth(
                x=batch_noise_t,
                timesteps=timesteps,
                label=batch_y
            )
            # compute train loss
            train_losses = loss_fnc(
                input=noise_t,
                target=predicted_noise,
            )

            # reset encoder and decoder gradients
            optimizer.zero_grad()

            # run error back-propagation
            train_losses.backward()

            # optimize encoder and decoder parameters
            optimizer.step()

            # collect rec error losses
            total_losses.append(train_losses.detach().cpu().numpy())

            rnd += 1
            if rnd >= self._client_rounds:
                break

        # average of rec errors
        loss = np.mean(np.array(total_losses)).item()
        reply = Update()
        reply["arrays"] = get_parameters(mlp_synth)
        reply["loss"] = loss
        reply["num_samples"] = n_samples
        return reply

    def sample(self, request: Update, num_rows: int, seed: int) -> DataFrame:
        return DataFrame()