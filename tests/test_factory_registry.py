from abc import ABC, abstractmethod

import pytest

from fedbench.runtime.registry import FactoryRegistry


class Product:
    @classmethod
    def create(cls):
        return cls()


class AbstractProduct(ABC):
    @abstractmethod
    def do_stuff(self):
        pass


def object_factory():
    return object()


@pytest.fixture
def object_registry() -> FactoryRegistry[object]:
    return FactoryRegistry(group=f"{__name__}.object", product_cls=object)


@pytest.fixture
def product_registry() -> FactoryRegistry[Product]:
    return FactoryRegistry(group=f"{__name__}.product", product_cls=Product)


@pytest.fixture
def abstract_product_registry() -> FactoryRegistry[AbstractProduct]:
    return FactoryRegistry(
        group=f"{__name__}.abstract_product", product_cls=AbstractProduct
    )


def test_constructor_rejects_bad_product_cls():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        FactoryRegistry(group="object", product_cls=object())


def test_call_unknown_factory(object_registry) -> None:
    with pytest.raises(ValueError):
        object_registry.call("unknown-product")
