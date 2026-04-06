"""
app/tools/calculator.py
────────────────────────
Calculator tool — xử lý phép tính an toàn.
KHÔNG dùng eval() trực tiếp. Dùng ast.parse → chỉ cho phép biểu thức số học.
"""

import ast
import operator
from app.tools.base import BaseTool

# Chỉ cho phép các phép tính an toàn
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Đệ quy đánh giá AST node — chỉ chấp nhận số & phép tính cơ bản."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv)) and right == 0:
            raise ValueError("Không thể chia cho 0")
        return _SAFE_OPS[type(node.op)](left, right)
    raise ValueError(f"Biểu thức không được phép: {ast.dump(node)}")


class CalculatorTool(BaseTool):

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Thực hiện phép tính số học. "
            "Input là biểu thức toán học, ví dụ: '3 * 3500000' hoặc '(12000000 * 0.85) + 3500000'. "
            "Chỉ hỗ trợ +, -, *, /, //, %, **."
        )

    def execute(self, input_text: str) -> str:
        expr = input_text.strip()
        # Cho phép dấu phẩy ngăn cách hàng nghìn
        expr = expr.replace(",", "")
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree)
        # Format kết quả đẹp
        if result == int(result):
            formatted = f"{int(result):,}"
        else:
            formatted = f"{result:,.2f}"
        return f"{formatted}"
