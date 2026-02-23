from abc import ABC, abstractmethod

import pytest

from fedbench.core.registry import FactoryRegistry


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
        group=f"{__name__}.abstract_product",
        product_cls=AbstractProduct
    )


def test_constructor_rejects_bad_product_cls():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        FactoryRegistry(group="object", product_cls=object())


def test_rejects_module_only_locator(object_registry) -> None:
    with pytest.raises(ValueError):
        object_registry.add_builtin("bad", "modulename")


def test_rejects_duplicate_builtin(object_registry) -> None:
    object_registry.add_builtin("name", "first_module:qualifier")
    with pytest.raises(ValueError):
        object_registry.add_builtin("name", "other_module:qualifier")


def test_raises_on_unexpected_product_type() -> None:
    registry = FactoryRegistry(group="product", product_cls=Product)
    registry.add_builtin("great-product", f"{__name__}:object_factory")
    with pytest.raises(TypeError):
        registry.call("great-product")


def test_contains_added_builtin(object_registry) -> None:
    assert "great-product" not in object_registry
    object_registry.add_builtin("great-product", f"{__name__}:object_factory")
    assert "great-product" in object_registry


def test_can_call_valid_factory_simple_qualifier(product_registry) -> None:
    product_registry.add_builtin("great-product", f"{__name__}:Product")
    product = product_registry.call("great-product")
    assert isinstance(product, Product)


def test_can_call_valid_factory_dotted_qualifier(product_registry) -> None:
    product_registry.add_builtin("great-product", f"{__name__}:Product.create")
    product = product_registry.call("great-product")
    assert isinstance(product, Product)


def test_call_unknown_factory(object_registry) -> None:
    with pytest.raises(ValueError):
        object_registry.call("unknown-product")


def test_can_not_call_abstract_class(abstract_product_registry) -> None:
    abstract_product_registry.add_builtin("product", f"{__name__}:AbstractProduct")
    with pytest.raises(TypeError, match=f"{AbstractProduct} is an abstract class"):
        abstract_product_registry.call("product")