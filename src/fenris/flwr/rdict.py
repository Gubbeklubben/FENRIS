from collections.abc import Iterator, MutableMapping
from typing import cast

from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, RecordDict

type RecordType = ArrayRecord | MetricRecord | ConfigRecord


class _NamespaceView[T](MutableMapping[str, T]):
    def __init__(self, namespace: str, src: MutableMapping[str, T], sep: str) -> None:
        self._namespace = namespace
        self._src = src
        self._sep = sep

    def __iter__(self) -> Iterator[str]:
        prefix = self._prefix()
        for key in self._src:
            if key.startswith(prefix):
                yield key.removeprefix(prefix)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __getitem__(self, key: str) -> T:
        try:
            return self._src[self._full_key(key)]
        except KeyError:
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: T) -> None:
        self._src[self._full_key(key)] = value

    def __delitem__(self, key: str) -> None:
        try:
            del self._src[self._full_key(key)]
        except KeyError:
            raise KeyError(key) from None

    def _full_key(self, key: str) -> str:
        return f"{self._namespace}{self._sep}{key}"

    def _prefix(self) -> str:
        return f"{self._namespace}{self._sep}"


class RDictNamespaceView(_NamespaceView[RecordType]):
    def __init__(self, namespace: str, rdict: RecordDict, sep: str = ".") -> None:
        if not isinstance(rdict, RecordDict):
            raise TypeError(f"Expected RecordDict, got type {type(rdict)}.")
        super().__init__(namespace, rdict, sep)

    @property
    def array_records(self) -> _NamespaceView[ArrayRecord]:
        src = cast(RecordDict, self._src)
        return _NamespaceView[ArrayRecord](
            self._namespace, src.array_records, self._sep
        )

    @property
    def config_records(self) -> _NamespaceView[ConfigRecord]:
        src = cast(RecordDict, self._src)
        return _NamespaceView[ConfigRecord](
            self._namespace, src.config_records, self._sep
        )

    @property
    def metric_records(self) -> _NamespaceView[MetricRecord]:
        src = cast(RecordDict, self._src)
        return _NamespaceView[MetricRecord](
            self._namespace, src.metric_records, self._sep
        )
