import ast
from typing import Literal

import pytest

from fenris.app.scaffold import create_component_scaffold
from fenris.core.component import Component
from tests.test_scaffold.components.base import Base
from tests.test_scaffold.components.override_abstract_with_abstract import (
    OverrideAbstractWithAbstract,
)
from tests.test_scaffold.components.override_abstract_with_concrete import (
    OverrideAbstractWithConcrete,
)
from tests.test_scaffold.components.override_cls_var import (
    OverrideRequiredWithNotRequired,
    OverrideRequiredWithRequired,
)
from tests.test_scaffold.components.with_mixin import WithMixin

NAME = "fed_up"
CLASS_NAME = "".join(w.capitalize() for w in NAME.split("_"))


def generate_code(cls: type[Component]) -> str:
    code = create_component_scaffold(cls, CLASS_NAME)
    return code


@pytest.fixture(scope="session")
def generated_code_for_base() -> str:
    return generate_code(Base)


def assert_in(
    where: Literal["imports", "class_body"], code: str, substring: str, target_cls: type
) -> None:

    index = code.find(f"class {CLASS_NAME}({target_cls.__name__})")
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


def test_can_scaffold_relevant():
    # If scaffolding were to crash for some component ABC,
    # this test should reveal it.
    from fenris.app.plugins import plugins

    for group in plugins.groups.values():
        for base in group.bases:
            create_component_scaffold(base, "Test")


def test_cls_vars(generated_code_for_base):
    assert_in_class_body(generated_code_for_base, "REQUIRED: int =", Base)
    assert "NOT_REQUIRED: int =" not in generated_code_for_base
    assert "NOT_ANNOTATED =" not in generated_code_for_base


def test_override_cls_vars():
    code = generate_code(OverrideRequiredWithRequired)
    assert_in_class_body(code, "REQUIRED: int =", OverrideRequiredWithRequired)
    code = generate_code(OverrideRequiredWithNotRequired)
    assert "REQUIRED: int =" not in code


def test_imports_cls_var_accesses(generated_code_for_base):
    assert_in_class_body(generated_code_for_base, "REMEMBER_ACCESSES: ClassVar[", Base)
    assert_in_imports(generated_code_for_base, "ClassVar", Base)


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
        f"from {Base.__module__} import Base, SOME_CONSTANT, SomeClass, "
        f"SomeOtherClass, "
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


def test_override_abstract_with_concrete():
    fn_def = """
    @property
    def keep_decorator(self) -> str:
    """
    code = generate_code(OverrideAbstractWithConcrete)
    assert fn_def not in code


def test_override_abstract_with_abstract():
    fn_def = """
    @property
    def keep_decorator(self) -> str:
    """
    assert_in_class_body(
        generate_code(OverrideAbstractWithAbstract),
        fn_def,
        OverrideAbstractWithAbstract,
    )


def test_with_mixin():
    fn_def = """
    def mix_it_up(self) -> None:
    """
    assert_in_class_body(generate_code(WithMixin), fn_def, WithMixin)
