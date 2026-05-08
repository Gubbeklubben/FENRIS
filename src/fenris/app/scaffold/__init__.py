from fenris.app.scaffold.collector import Collector
from fenris.app.scaffold.resolver import Resolver
from fenris.app.scaffold.transformer import ComponentTransformer
from fenris.core.component import Component

__all__ = [
    "create_component_scaffold",
]


_resolvers: dict[type[Component], Resolver] = {}


def create_component_scaffold(
    target_cls: type[Component], name: str, cls_name: str
) -> str:

    try:
        rslv = _resolvers[target_cls]
    except KeyError:
        rslv = Resolver(Collector(target_cls))
        _resolvers[target_cls] = rslv

    return rslv.module.visit(ComponentTransformer(name, cls_name)).code
