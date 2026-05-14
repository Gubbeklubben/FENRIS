"""Add the final transformations to a resolved scaffold (cst.Module).

Classes
-------
ComponentTransformer
    A libcst.CSTTransformer that modifies class name, and replaces
    method bodies with a statement raising NotImplementedError.
"""

import libcst as cst


class ComponentTransformer(cst.CSTTransformer):
    def __init__(self, cls_name: str) -> None:
        super().__init__()
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

        return updated_node.with_changes(
            body=cst.IndentedBlock(
                body=[
                    cst.SimpleStatementLine(
                        body=[
                            cst.Raise(
                                exc=cst.Call(func=cst.Name("NotImplementedError"))
                            )
                        ]
                    )
                ]
            )
        )
