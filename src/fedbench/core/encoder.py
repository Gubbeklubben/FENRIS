import dataclasses
import json
from typing import Any

from fedbench.evaluators.fairness import PerGroupConfusion

# Registry of every dataclass that crosses the FL client/server boundary,
# keyed by qualified class name. Both encoder and decoder use this registry,
# so adding a new payload type requires only one entry here.
_DATACLASS_REGISTRY: dict[str, type] = {
    "PerGroupConfusion": PerGroupConfusion,
}


class FedbenchEncoder(json.JSONEncoder):
    """JSON encoder that transparently handles registered dataclasses.

    Registered dataclasses are encoded as a plain dict with a ``__dataclass__``
    tag so the decoder can reconstruct the correct type. All other values fall
    through to the default JSON encoder.

    ``dataclasses.asdict`` is recursive, so nested dataclasses are encoded
    correctly provided every type in the tree is registered.
    """

    def default(self, obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {"__dataclass__": type(obj).__qualname__, **dataclasses.asdict(obj)}
        return super().default(obj)

    @staticmethod
    def decode(obj: dict[str, Any]) -> Any:
        """``object_hook`` for ``json.loads`` — reconstructs registered dataclasses.

        Called bottom-up on every dict in the JSON tree, so nested dataclasses
        are reconstructed before the dicts that contain them.  Unrecognised
        dicts (no ``__dataclass__`` key, or key not in registry) are returned
        unchanged.
        """
        if "__dataclass__" in obj:
            cls = _DATACLASS_REGISTRY.get(obj["__dataclass__"])
            if cls:
                return cls(**{k: v for k, v in obj.items() if k != "__dataclass__"})
        return obj
