import keyword

import typer


def validate_identifier(arg: str) -> str:
    if keyword.iskeyword(arg):
        raise typer.BadParameter(f"'{arg}' is a reserved keyword.")

    if not arg.isidentifier():
        raise typer.BadParameter(f"'{arg}' is not a valid python identifier.")

    return arg
