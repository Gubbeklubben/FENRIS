"""Tests for FedbenchEncoder / FedbenchEncoder.decode.

Covers:
- Round-trip for a flat registered dataclass
- Round-trip for a nested structure (dict[str, dict[str, PerGroupConfusion]])
- Unknown dicts pass through decode unchanged
- Unregistered dataclasses are not decoded (raw dict returned)
- Non-dataclass values encode normally
"""

import json
from dataclasses import dataclass

import pytest

from fedbench.core.encoder import FedbenchEncoder
from fedbench.evaluators.fairness import PerGroupConfusion


def dumps(obj) -> str:
    return json.dumps(obj, cls=FedbenchEncoder)


def loads(raw: str):
    return json.loads(raw, object_hook=FedbenchEncoder.decode)


def roundtrip(obj):
    return loads(dumps(obj))


# ---------------------------------------------------------------------------
# Flat dataclass
# ---------------------------------------------------------------------------

class TestFlatRoundTrip:

    def test_per_group_confusion_identity(self):
        """PerGroupConfusion survives a full encode/decode cycle."""
        cm = PerGroupConfusion(tp=10, fp=3, tn=85, fn=2)
        assert roundtrip(cm) == cm

    def test_default_values_preserved(self):
        """Zero-initialised PerGroupConfusion round-trips correctly."""
        cm = PerGroupConfusion()
        assert roundtrip(cm) == cm

    def test_encoded_contains_tag(self):
        """Encoded form must contain the __dataclass__ discriminator tag."""
        cm = PerGroupConfusion(tp=1, fp=2, tn=3, fn=4)
        raw = json.loads(dumps(cm))
        assert raw["__dataclass__"] == "PerGroupConfusion"

    def test_encoded_contains_all_fields(self):
        """All four fields must be present in the encoded dict."""
        cm = PerGroupConfusion(tp=1, fp=2, tn=3, fn=4)
        raw = json.loads(dumps(cm))
        assert raw["tp"] == 1
        assert raw["fp"] == 2
        assert raw["tn"] == 3
        assert raw["fn"] == 4


# ---------------------------------------------------------------------------
# Nested structures
# ---------------------------------------------------------------------------

class TestNestedRoundTrip:

    def test_dict_of_per_group_confusion(self):
        """dict[str, PerGroupConfusion] round-trips correctly."""
        payload = {
            "group_a": PerGroupConfusion(tp=10, fp=1, tn=80, fn=5),
            "group_b": PerGroupConfusion(tp=20, fp=2, tn=70, fn=3),
        }
        result = roundtrip(payload)
        assert result == payload

    def test_nested_dict_of_per_group_confusion(self):
        """dict[str, dict[str, PerGroupConfusion]] round-trips correctly.

        This is the actual payload shape emitted by FairnessEvaluator.local_evaluate.
        object_hook is called bottom-up so inner PerGroupConfusion instances are
        reconstructed before the outer dict is processed.
        """
        payload = {
            "sensitive_col": {
                "group_0": PerGroupConfusion(tp=5, fp=1, tn=40, fn=2),
                "group_1": PerGroupConfusion(tp=8, fp=3, tn=35, fn=4),
            }
        }
        result = roundtrip(payload)
        assert result == payload

    def test_list_of_per_group_confusion(self):
        """list[PerGroupConfusion] round-trips correctly."""
        payload = [
            PerGroupConfusion(tp=1, fp=0, tn=9, fn=0),
            PerGroupConfusion(tp=2, fp=1, tn=7, fn=0),
        ]
        assert roundtrip(payload) == payload


# ---------------------------------------------------------------------------
# Pass-through behaviour
# ---------------------------------------------------------------------------

class TestPassThrough:

    def test_plain_dict_unchanged(self):
        """Dicts without __dataclass__ tag are returned unchanged."""
        payload = {"key": "value", "count": 42}
        assert roundtrip(payload) == payload

    def test_unregistered_dataclass_returned_as_dict(self):
        """Dataclasses not in the registry decode to a plain dict, not the class."""
        @dataclass
        class _Unregistered:
            x: int = 0

        raw = dumps(_Unregistered(x=7))
        result = loads(raw)
        # Decoded as a plain dict since _Unregistered is not registered
        assert isinstance(result, dict)
        assert result["x"] == 7

    def test_primitive_values_unchanged(self):
        """Ints, floats, strings, and lists pass through without modification."""
        assert roundtrip(42) == 42
        assert roundtrip(3.14) == pytest.approx(3.14)
        assert roundtrip("hello") == "hello"
        assert roundtrip([1, 2, 3]) == [1, 2, 3]