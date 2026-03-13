from enum import StrEnum

from flwr.common import RecordDict


class Namespace(StrEnum):
    FRAMEWORK = "framework"
    GLOBAL_INIT_ARTIFACTS = "artifacts"
    SYNTHESIZER = "synthesizer"

    def make_prefix(self) -> str:
        return f"{self.value}::"


class CacheManager:
    def __init__(self, flwr_cache: RecordDict) -> None:
        self._cache = flwr_cache

    def get_cache(self, namespace: Namespace) -> RecordDict:
        prefix = namespace.make_prefix()
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

        return out

    def set_cache(self, namespace: Namespace, cache: RecordDict) -> None:
        prefix = namespace.make_prefix()

        for key, arr in cache.array_records.items():
            self._cache.array_records[f"{prefix}{key}"] = arr

        for key, cfg in cache.config_records.items():
            self._cache.config_records[f"{prefix}{key}"] = cfg

        for key, metric in cache.metric_records.items():
            self._cache.metric_records[f"{prefix}{key}"] = metric

        prune_keys = []
        for key in self._cache.keys():
            if key.startswith(prefix) and key.removeprefix(prefix) not in cache:
                prune_keys.append(key)

        for key in prune_keys:
            self._cache.pop(key, None)
