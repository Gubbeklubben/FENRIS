from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Literal

Kind = Literal["continuous", "categorical", "binary", "integer"]

# --------------------------------------------------------------------------- #
# Column → keeps the column name (string) and a broad Semantic *kind*
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ColumnSchema:
    name: str
    kind: Kind

# --------------------------------------------------------------------------- #
# Table keeps an ordered tuple of ColumnSchema objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TableSchema:
    columns: Tuple[ColumnSchema, ...] = field(default_factory=tuple)

    # convenience: look‑up helper
    def lookup(self, name: str) -> ColumnSchema:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"Column '{name}' not found in schema.")

    # handy short‑cut for checking column kinds
    def kind_of(self, name: str) -> str:
        return self.lookup(name).kind
