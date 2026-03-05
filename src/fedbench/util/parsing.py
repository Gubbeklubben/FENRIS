import inspect
import re
from types import UnionType
from typing import get_origin, Any, get_args, Callable, Union, Literal


def split_outside_brackets(s: str) -> list[str]:
    if not s:
        return []

    result: list[str] = []
    current: list[str] = []
    depth_paren = 0
    depth_brack = 0

    for ch in s:
        if ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren -= 1
        elif ch == '[':
            depth_brack += 1
        elif ch == ']':
            depth_brack -= 1

        if ch == ',' and depth_paren == 0 and depth_brack == 0:
            result.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)

    result.append(''.join(current).strip())
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

        raise TypeError(
            f"Expected one of {literals}, got {value!r}"
        )

    # Fallback: call the type directly
    return annotation(value)


def is_optional(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is Union:
        return type(None) in get_args(annotation)
    return False


def parse_for_function(func: Callable[..., Any], raw: dict[str, str],) -> dict[str, Any]:
    sig = inspect.signature(func)
    params = sig.parameters

    # Reject unknown parameters
    unknown = set(raw) - set(params)
    if unknown:
        raise TypeError(f"Unknown parameters for {func.__name__}: {', '.join(sorted(unknown))}")

    # Parse and validate required params
    parsed = {}
    for name, param in params.items():
        has_default = param.default is not inspect.Parameter.empty
        optional = has_default or is_optional(param.annotation)

        if name in raw:
            parsed[name] = coerce(raw[name], param.annotation)
        else:
            if not optional:
                raise TypeError(f"Missing required parameter for {func.__name__}: {name}")

    return parsed


def to_snake_case(text: str) -> str:
    return re.sub(r'[^a-z_]+', '_', text.lower())