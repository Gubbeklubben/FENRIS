from flwr.common import RecordDict, ConfigRecord

_CONFIG = "config"
_ARTIFACTS = "artifacts"
_SYNTHESIZER = "synthesizer"


def _make_prefix(category: str) -> str:
    return f"{category}."


class ClientCacheWrapper:
    def __init__(self, flwr_cache: RecordDict) -> None:
        self._cache = flwr_cache

    def set_config(self, config: ConfigRecord) -> None:
        self._cache.config_records[_CONFIG] = config

    def get_config(self) -> ConfigRecord | None:
        try:
            return self._cache.config_records[_CONFIG]
        except KeyError:
            return None

    def get_artifacts(self) -> RecordDict | None:
        prefix = _make_prefix(_ARTIFACTS)
        out = RecordDict()

        for key in self._cache.array_records.keys():
            if key.startswith(prefix):
                out_key = key.removeprefix(prefix)
                out.array_records[out_key] = self._cache.array_records[key]

        for key in self._cache.config_records.keys():
            if key.startswith(prefix):
                out_key = key.removeprefix(prefix)
                out.config_records[out_key] = self._cache.config_records[key]

        for key in self._cache.metric_records.keys():
            if key.startswith(prefix):
                out_key = key.removeprefix(prefix)
                out.metric_records[out_key] = self._cache.metric_records[key]

        if out:
            return out
        return None

    def set_artifacts(self, artifacts: RecordDict) -> None:
        prefix = _make_prefix(_ARTIFACTS)

        for key, arr in artifacts.array_records.items():
            self._cache.array_records[f"{prefix}{key}"] = arr

        for key, cfg in artifacts.config_records.items():
            self._cache.config_records[f"{prefix}{key}"] = cfg

        for key, metric in artifacts.metric_records.items():
            self._cache.metric_records[f"{prefix}{key}"] = metric

    # noinspection PyMethodMayBeStatic
    def get_synthesizer_cache(self) -> RecordDict | None:
        return None

    def set_synthesizer_cache(self) -> None:
        pass