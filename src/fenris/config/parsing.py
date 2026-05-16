import inspect
from collections.abc import Callable, Mapping
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin


def parse_kwargs(value: str) -> dict[str, str]:
    if value is None:
        return {}

    result = {}

    for item in split_outside_brackets(value):
        key, val = item.split("=")
        result[key] = val
    return result


def parse_args(value: str) -> list[str]:
    if value is None:
        return []
    return value.split(",")


def split_outside_brackets(s: str) -> list[str]:
    if not s:
        return []

    result: list[str] = []
    current: list[str] = []
    depth_paren = 0
    depth_brack = 0

    for ch in s:
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack -= 1

        if ch == "," and depth_paren == 0 and depth_brack == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(ch)

    result.append("".join(current).strip())
    return result


def coerce(value: str, annotation: Any) -> Any:

    # Handle Union / Optional (PEP 604 and typing.Union)
    base_type = get_origin(annotation)
    if base_type in (UnionType, Union):
        for arg in get_args(annotation):
            if arg is type(None):
                if value in {"", "none", "null", "None"}:
                    return None
                continue

            try:
                return coerce(value, arg)
            except Exception:
                pass

        raise TypeError(f"Value {value!r} does not match any type in {annotation}")

    # Handle containers (list, tuple, etc.)
    if not base_type:
        base_type = annotation
    if not base_type:
        return value

    if base_type in (list, tuple):
        if base_type is tuple and (not value[0] == "(" or not value[-1] == ")"):
            raise TypeError(f"Expected tuple, got {value}")
        if base_type is list and (not value[0] == "[" or not value[-1] == "]"):
            raise TypeError(f"Expected list, got {value}")

        type_args = get_args(annotation)
        list_type = type_args[0] if len(type_args) > 0 else None

        vals = split_outside_brackets(value[1:-1])
        val_list = [coerce(val, list_type) for val in vals]
        if base_type is tuple:
            return tuple(val_list)
        elif base_type is list:
            return val_list

    # Handle bool specially
    if annotation is bool:
        return value.lower() in {"true", "1", "yes", "on"}

    # Handle Literal[...] types
    if get_origin(annotation) is Literal:
        literals = get_args(annotation)

        for lit in literals:
            # Coerce to the literal's type (str, int, bool, etc.)
            try:
                coerced = type(lit)(value)
            except ValueError:
                continue

            if coerced == lit:
                return lit

        raise TypeError(f"Expected one of {literals}, got {value!r}")

    # Fallback: call the type directly
    return annotation(value)


def is_optional(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is Union:
        return type(None) in get_args(annotation)
    return False


def parse_kwargs_for_function(
    func: Callable[..., Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:

    # Reject unknown parameters
    params = inspect.signature(func).parameters
    unknown = set(kwargs) - set(params)
    if unknown:
        raise ValueError(
            f"Unknown parameters for {func.__name__}: {', '.join(sorted(unknown))}\n"
            f"Valid parameters: {', '.join(sorted(params))}"
        )

    # Parse and validate required params
    parsed: dict[str, Any] = {}
    missing = []
    for name, param in params.items():
        # Is component parameter specified in kwargs?
        if name in kwargs:
            # If param lacks type hint, pass value unchanged, otherwise coerce
            if param.annotation is inspect.Parameter.empty:
                parsed[name] = kwargs[name]
            else:
                parsed[name] = coerce(kwargs[name], param.annotation)

        # If parameter has a default value, use it
        elif param.default is not inspect.Parameter.empty:
            parsed[name] = param.default

        # If parameter is allowed to be None, set it to None
        elif is_optional(param.annotation):
            parsed[name] = None

        # Parameter is required, but missing and has no default value
        else:
            missing.append(f"\n{'':<2}{param}")

    if missing:
        raise TypeError(
            f"Missing required parameter(s) for {func.__name__}:{''.join(missing)}"
        )

    return parsed
