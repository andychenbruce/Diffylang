from typing import Literal


class FunctionDefinition:
    name: str
    from_type: PrimitiveType
    to_type: PrimitiveType
    arguments: FuncArgument
    body: Expression

class PrimitiveType:
    type: Literal["int", "float", "str"] | SumType[PrimitiveTypes, PrimitiveTypes] | ProdType[PrimitiveTypes, PrimitiveTypes]

class SumType[A, B]:
    val: A | B

class ProdType[A, B]:
    left: A
    right: B

class FuncArgument:
    identifier: str
    type: PrimitiveType

class Expression:
    Literal[int, str, float] | EqualsTo

class EqualsTo:
    
