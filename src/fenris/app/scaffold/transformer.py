"""Add the final transformations to a resolved scaffold (cst.Module).

Classes
-------
ComponentTransformer
    A libcst.CSTTransformer that modifies class name, implements Component.name,
    and replaces the bodies of other methods with a statement raising
    NotImplementedError.
"""

import libcst as cst


class ComponentTransformer(cst.CSTTransformer):
    def __init__(self, name: str, cls_name: str) -> None:
        super().__init__()
        self._name = name
        self._cls_name = cls_name

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        return updated_node.with_changes(name=cst.Name(value=self._cls_name))

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        # Don't visit children
        return False

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:

        body = (
            cst.SimpleStatementLine(
                body=[
                    # Deliberate violation of quoting convention
                    # to get doubles in output.
                    cst.Return(value=cst.SimpleString(f'"{self._name}"'))
                ]
            )
            if original_node.name.value == "name"
            else cst.SimpleStatementLine(
                body=[cst.Raise(exc=cst.Call(func=cst.Name("NotImplementedError")))]
            )
        )
        return updated_node.with_changes(body=cst.IndentedBlock(body=[body]))
