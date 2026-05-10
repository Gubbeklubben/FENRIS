import libcst as cst

import fenris.app.scaffold.resolver as resolver
from fenris.app.scaffold.collector import Collector
from fenris.app.scaffold.transformer import ComponentTransformer
from fenris.core.component import Component

__all__ = [
    "create_component_scaffold",
]


_modules: dict[type[Component], cst.Module] = {}


def create_component_scaffold(
    target_cls: type[Component], name: str, cls_name: str
) -> str:

    if not issubclass(target_cls, Component):
        raise TypeError(f"{target_cls} is not a subclass of {Component}.")
    try:
        module = _modules[target_cls]
    except KeyError:
        clt = Collector(target_cls)
        clt.collect()
        module = resolver.resolve(clt)
        _modules[target_cls] = module

    return module.visit(ComponentTransformer(name, cls_name)).code
