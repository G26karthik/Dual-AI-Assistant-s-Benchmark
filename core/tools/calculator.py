from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_FUNCS: dict[str, Callable[..., Any]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "abs": abs,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        binary_op = _BINARY_OPS[type(node.op)]
        return float(binary_op(left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        operand = _eval_node(node.operand)
        unary_op = _UNARY_OPS[type(node.op)]
        return float(unary_op(operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name in _SAFE_FUNCS:
            args = [_eval_node(arg) for arg in node.args]
            func = _SAFE_FUNCS[func_name]
            return float(func(*args))
    raise ValueError("Unsupported expression")


def calculate(expression: str) -> str:
    """Safely evaluate simple arithmetic and math function expressions."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_node(parsed.body)
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"
