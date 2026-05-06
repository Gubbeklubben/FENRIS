import dataclasses
import importlib
import json
from typing import Any


class FenrisEncoder(json.JSONEncoder):
    """JSON encoder/decoder for dataclasses exchanged between client and server.

    Dataclasses are encoded as a plain dict with ``__dataclass__`` and
    ``__module__`` tags, allowing the decoder to reconstruct the correct type
    by dynamic import without any prior registration.  All other values fall
    through to the default JSON encoder.

    ``dataclasses.asdict`` is recursive, so nested dataclasses are encoded
    correctly provided every type in the tree is a dataclass.
    """

    def default(self, obj: Any) -> Any:
        """Serialize *obj* when the standard encoder cannot handle it.

        Parameters
        ----------
        obj : Any
            The object to serialize.

        Returns
        -------
        dict or Any
            A plain dict with ``__dataclass__`` and ``__module__`` tags if
            *obj* is a dataclass instance; otherwise delegates to the standard
            JSON encoder.
        """
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {
                "__dataclass__": type(obj).__qualname__,
                "__module__": type(obj).__module__,
                **{f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)},
            }
        return super().default(obj)

    @staticmethod
    def decode(obj: dict[str, Any]) -> Any:
        """Reconstruct a dataclass from a tagged dict; for use as ``object_hook``.

        Called bottom-up on every dict in the JSON tree, so nested dataclasses
        are reconstructed before the dicts that contain them.

        Parameters
        ----------
        obj : dict[str, Any]
            A dict parsed from JSON. Must contain both ``__dataclass__`` and
            ``__module__`` keys to trigger reconstruction.

        Returns
        -------
        Any
            The reconstructed dataclass instance if both tags are present and
            the class can be imported; otherwise *obj* is returned unchanged.
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
