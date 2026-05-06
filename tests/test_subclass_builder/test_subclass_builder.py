import ast
from typing import Literal

import pytest

from fenris.app.scaffold import AbstractMethodCollector, Builder
from tests.test_subclass_builder.components.base import Base
from tests.test_subclass_builder.components.override_abstract_with_abstract import (
    OverrideAbstractWithAbstract,
)
from tests.test_subclass_builder.components.override_abstract_with_concrete import (
    OverrideAbstractWithConcrete,
)
from tests.test_subclass_builder.components.with_mixin import WithMixin

GENERATED_CLASS_NAME = "Test"


def generate_code(cls: type) -> str:
    collector = AbstractMethodCollector(cls)
    builder = Builder(collector)
    return builder.with_name(GENERATED_CLASS_NAME).build().code


@pytest.fixture(scope="session")
def generated_code_for_base() -> str:
    return generate_code(Base)


@pytest.fixture(scope="function")
def generated_code_for_override_abstract_with_concrete() -> str:
    return generate_code(OverrideAbstractWithConcrete)


@pytest.fixture(scope="function")
def generated_code_for_override_abstract_with_abstract() -> str:
    return generate_code(OverrideAbstractWithAbstract)


@pytest.fixture(scope="function")
def generated_code_for_with_mixin() -> str:
    return generate_code(WithMixin)


def assert_in(
    where: Literal["imports", "class_body"], code: str, substring: str, target_cls: type
) -> None:

    index = code.find(f"class {GENERATED_CLASS_NAME}({target_cls.__name__})")
    if index == -1:
        raise ValueError("Could not find expected class def.")

    match where:
        case "imports":
            assert substring in code[:index]
        case "class_body":
            assert substring in code[index:]


def assert_in_imports(code: str, substring: str, target_cls: type) -> None:
    assert_in("imports", code, substring, target_cls)


def assert_in_class_body(code: str, substring: str, target_cls: type) -> None:
    assert_in("class_body", code, substring, target_cls)


def test_syntax_valid(generated_code_for_base):
    # Provoke SyntaxError if generated code is bad for some reason
    ast.parse(generated_code_for_base)


def test_keep_decorator(generated_code_for_base):
    fn_def = """
    @property
    def keep_decorator(self) -> str:
        raise NotImplementedError()
    """
    assert_in_class_body(generated_code_for_base, fn_def, Base)


def test_horizontal_args(generated_code_for_base):
    fn_def = """
    def horizontal_args(self, x: int, y: bool, z: float) -> None:
    """
    assert_in_class_body(generated_code_for_base, fn_def, Base)


def test_vertical_args(generated_code_for_base):
    fn_def = """
    def vertical_args(
        self,
        x: int,
        y: bool,
        z: float,
    ) -> None:
    """
    assert_in_class_body(generated_code_for_base, fn_def, Base)


def test_reference_import_in_params(generated_code_for_base):
    imp = "from pandas import DataFrame"
    assert_in_imports(generated_code_for_base, imp, Base)


def test_reference_import_in_return(generated_code_for_base):
    imp = "import threading"
    assert_in_imports(generated_code_for_base, imp, Base)


def test_reference_import_asname(generated_code_for_base):
    imp = "import numpy as np"
    assert_in_imports(generated_code_for_base, imp, Base)


def test_reference_local(generated_code_for_base):
    imp = (
        f"from {Base.__module__} import SOME_CONSTANT, SomeClass, SomeOtherClass, "
        f"some_decorator"
    )
    assert_in_imports(generated_code_for_base, imp, Base)


def test_removes_irrelevant_imports(generated_code_for_base):
    imps = (
        "from abc import ABC, abstractmethod",
        "from collections.abc import Callable",
    )
    for imp in imps:
        assert imp not in generated_code_for_base


def test_override_abstract_with_concrete(
    generated_code_for_override_abstract_with_concrete,
):
    fn_def = """
    @property
    def keep_decorator(self) -> str:
    """
    assert fn_def not in generated_code_for_override_abstract_with_concrete


def test_override_abstract_with_abstract(
    generated_code_for_override_abstract_with_abstract,
):
    fn_def = """
    @property
    def keep_decorator(self) -> str:
    """
    assert_in_class_body(
        generated_code_for_override_abstract_with_abstract,
        fn_def,
        OverrideAbstractWithAbstract,
    )


def test_with_mixin(generated_code_for_with_mixin):
    fn_def = """
    def mix_it_up(self) -> None:
    """
    assert_in_class_body(generated_code_for_with_mixin, fn_def, WithMixin)
