"""Tests for FedbenchEncoder / FedbenchEncoder.decode.

Covers:
- Round-trip for flat and nested dataclasses
- Round-trip for dicts and lists containing dataclasses
- Correct __dataclass__ / __module__ tags in encoded form
- Locally-defined dataclasses (not reachable as module
  attributes) decode to a plain dict
- Unknown module tags pass through decode unchanged
- Primitive values encode and decode normally
"""

import json
from dataclasses import dataclass, field

import pytest

from fedbench.core.encoder import FedbenchEncoder


def dumps(obj) -> str:
    return json.dumps(obj, cls=FedbenchEncoder)


def loads(raw: str):
    return json.loads(raw, object_hook=FedbenchEncoder.decode)


def roundtrip(obj):
    return loads(dumps(obj))


# ---------------------------------------------------------------------------
# Module-level dataclasses used by TestNestedRoundTrip
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Inner:
    a: int = 0
    b: int = 0


@dataclass(frozen=True)
class _Outer:
    x: _Inner = field(default_factory=_Inner)
    y: str = ""


@dataclass
class _OuterWithDictField:
    children: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Encoding format
# ---------------------------------------------------------------------------


class TestEncodingFormat:
    def test_encoded_contains_dataclass_tag(self):
        raw = json.loads(dumps(_Inner(a=1, b=3)))
        assert raw["__dataclass__"] == "_Inner"

    def test_encoded_contains_module_tag(self):
        raw = json.loads(dumps(_Inner(a=1, b=3)))
        assert raw["__module__"] == "tests.test_encoder"

    def test_encoded_contains_all_fields(self):
        raw = json.loads(dumps(_Inner(a=1, b=3)))
        assert raw["a"] == 1
        assert raw["b"] == 3


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_flat_dataclass(self):
        assert roundtrip(_Inner(a=10, b=85)) == _Inner(a=10, b=85)

    def test_flat_dataclass_default_values(self):
        assert roundtrip(_Inner()) == _Inner()

    def test_dataclass_field(self):
        """A dataclass whose field is itself a dataclass."""
        assert roundtrip(_Outer(_Inner(1, 2), "test")) == _Outer(_Inner(1, 2), "test")

    def test_dataclass_with_dict_field_of_dataclasses(self):
        """A dataclass whose field is a dict mapping to dataclasses."""
        payload = _OuterWithDictField(children={"a": _Inner(a=1, b=2)})
        assert roundtrip(payload) == payload

    def test_dict_of_dataclasses(self):
        payload = {"group_a": _Inner(a=10, b=80), "group_b": _Inner(a=20, b=70)}
        assert roundtrip(payload) == payload

    def test_nested_dict_of_dataclasses(self):
        """dict[str, dict[str, dataclass]]"""
        payload = {
            "sensitive_col": {
                "group_0": _Inner(a=5, b=40),
                "group_1": _Inner(a=8, b=35),
            }
        }
        assert roundtrip(payload) == payload

    def test_list_of_dataclasses(self):
        payload = [_Inner(a=1, b=9), _Inner(a=2, b=7)]
        assert roundtrip(payload) == payload


# ---------------------------------------------------------------------------
# Pass-through behaviour
# ---------------------------------------------------------------------------


class TestPassThrough:
    def test_plain_dict(self):
        payload = {"key": "value", "count": 42}
        assert roundtrip(payload) == payload

    def test_primitives(self):
        assert roundtrip(42) == 42
        assert roundtrip(3.14) == pytest.approx(3.14)
        assert roundtrip("hello") == "hello"
        assert roundtrip([1, 2, 3]) == [1, 2, 3]

    def test_locally_defined_dataclass_decodes_to_dict(self):
        """A locally-defined dataclass is not reachable as a module attribute
        on the decoding side, so decode falls back to a plain dict."""

        @dataclass
        class _Local:
            x: int = 0

        result = loads(dumps(_Local(x=7)))
        assert isinstance(result, dict)
        assert result["x"] == 7
