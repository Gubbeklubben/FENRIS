from abc import ABC

from tests.test_scaffold.components.base import Base


class OverrideRequiredWithNotRequired(Base, ABC):
    REQUIRED: int = 2


class OverrideRequiredWithRequired(Base, ABC):
    # [scaffold] required_cls_var
    REQUIRED: int = 2
