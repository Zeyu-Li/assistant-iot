"""Arithmetic: the LLM writes the expression, this evaluates it (the LLM never does the maths)."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable, Mapping

from assistant.flows.speech import number_phrase
from assistant.ports.flow import FlowContext

_BINARY: Mapping[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.BitXor: operator.pow,  # "2^10" spoken aloud means a power, never XOR
}
_UNARY: Mapping[type, Callable[[Any], Any]] = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCTIONS: Mapping[str, Callable[..., Any]] = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "log": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "factorial": math.factorial,
}
_CONSTANTS: Mapping[str, float] = {"pi": math.pi, "e": math.e}
_MAX_EXPONENT = 1000
_MAX_FACTORIAL = 170


class CalculationError(ValueError):
    pass


def evaluate(expression: str) -> float:
    """Evaluate arithmetic safely: numbers, + - * / // % ** ^, parentheses and a few math functions."""
    try:
        tree = ast.parse(expression.replace("×", "*").replace("÷", "/"), mode="eval")
    except SyntaxError as error:
        raise CalculationError(f"not an arithmetic expression: {expression!r}") from error
    return float(_eval(tree.body))


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if type(node.op) in (ast.Pow, ast.BitXor) and abs(right) > _MAX_EXPONENT:
            raise CalculationError("exponent too large")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
        args = [_eval(a) for a in node.args]
        if node.func.id == "factorial" and (not args or args[0] > _MAX_FACTORIAL):
            raise CalculationError("factorial too large")
        return _FUNCTIONS[node.func.id](*args)
    raise CalculationError(f"unsupported: {ast.dump(node)[:60]}")


class CalculatorFlow:
    async def run(self, context: FlowContext) -> str:
        expression = str(context.arguments.get("expression") or "")
        try:
            result = evaluate(expression)
        except ZeroDivisionError:
            return "That divides by zero, so there's no answer."
        except (CalculationError, ValueError, TypeError, OverflowError):
            return "Sorry, I couldn't work that out."
        return f"It's {number_phrase(result)}."
