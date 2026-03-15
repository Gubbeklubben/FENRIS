import dataclasses
import importlib
import json
from typing import Any


class FedbenchEncoder(json.JSONEncoder):
    """JSON encoder/decoder for dataclasses exchanged between client and server.

    Dataclasses are encoded as a plain dict with ``__dataclass__`` and
    ``__module__`` tags, allowing the decoder to reconstruct the correct type
    by dynamic import without any prior registration.  All other values fall
    through to the default JSON encoder.

    ``dataclasses.asdict`` is recursive, so nested dataclasses are encoded
    correctly provided every type in the tree is a dataclass.
    """

    def default(self, obj: Any) -> Any:
        try:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {
                    "__dataclass__": type(obj).__qualname__,
                    "__module__": type(obj).__module__,
                    **dataclasses.asdict(obj),
                }
            return super().default(obj)
        except Exception as e:
            raise ValueError(
                f"Could not encode object {obj} with type {type(obj)}"
            ) from e

    @staticmethod
    def decode(obj: dict[str, Any]) -> Any:
        """``object_hook`` for ``json.loads`` — reconstructs dataclasses by
        dynamic import.

        Called bottom-up on every dict in the JSON tree, so nested dataclasses
        are reconstructed before the dicts that contain them.  Dicts without
        both tags, or whose class cannot be found, are returned unchanged.
        """
        if "__dataclass__" in obj and "__module__" in obj:
            module = importlib.import_module(obj["__module__"])
            cls = getattr(module, obj["__dataclass__"], None)
            if cls is not None:
                return cls(
                    **{
                        k: v
                        for k, v in obj.items()
                        if k not in ("__dataclass__", "__module__")
                    }
                )
        return obj
