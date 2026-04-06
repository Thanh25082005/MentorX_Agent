"""
tests/test_tools.py
────────────────────
Unit tests cho các tools: Calculator, CourseSearch, WebSearch.
Chạy: pytest tests/test_tools.py -v
"""

import pytest
from app.tools.calculator import CalculatorTool
from app.tools.course_search import CourseSearchTool
from app.tools.web_search import WebSearchTool


# ╔══════════════════════════════════════════════════════════╗
# ║  Calculator Tool                                         ║
# ╚══════════════════════════════════════════════════════════╝

class TestCalculatorTool:

    def setup_method(self):
        self.calc = CalculatorTool()

    def test_basic_addition(self):
        result = self.calc.execute("2 + 3")
        assert "5" in result

    def test_multiplication(self):
        result = self.calc.execute("3 * 3500000")
        assert "10,500,000" in result

    def test_complex_expression(self):
        result = self.calc.execute("(12000000 * 0.85) + 3500000")
        assert result  # should not raise

    def test_division_by_zero(self):
        with pytest.raises(ValueError, match="chia cho 0"):
            self.calc.execute("10 / 0")

    def test_safe_execute_catches_error(self):
        result, success = self.calc.safe_execute("import os")
        assert not success
        assert "Lỗi" in result

    def test_comma_in_number(self):
        result = self.calc.execute("1,000 + 2,000")
        assert "3,000" in result

    def test_name(self):
        assert self.calc.name == "calculator"

    def test_description(self):
        assert "phép tính" in self.calc.description.lower() or "số học" in self.calc.description.lower()


# ╔══════════════════════════════════════════════════════════╗
# ║  Course Search Tool                                      ║
# ╚══════════════════════════════════════════════════════════╝

class TestCourseSearchTool:

    def setup_method(self):
        self.search = CourseSearchTool()

    def test_search_python(self):
        result = self.search.execute("Python")
        assert "Python" in result
        assert "Tìm thấy" in result

    def test_search_beginner(self):
        result = self.search.execute("Beginner")
        assert "Beginner" in result

    def test_search_no_results(self):
        result = self.search.execute("xyznonexistent12345")
        assert "Không tìm thấy" in result

    def test_search_ml(self):
        result = self.search.execute("Machine Learning")
        assert "Machine Learning" in result

    def test_name(self):
        assert self.search.name == "course_search"


# ╔══════════════════════════════════════════════════════════╗
# ║  Web Search Tool (Mock)                                  ║
# ╚══════════════════════════════════════════════════════════╝

class TestWebSearchTool:

    def setup_method(self):
        self.web = WebSearchTool()

    def test_mock_returns_result(self):
        result = self.web.execute("AI trends 2025")
        assert "Mock" in result or "mock" in result

    def test_name(self):
        assert self.web.name == "web_search"

    def test_safe_execute(self):
        result, success = self.web.safe_execute("test query")
        assert success
        assert result
