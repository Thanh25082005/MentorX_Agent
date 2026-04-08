import sys
import os

# Ensure app directory is accessible
sys.path.append(os.getcwd())

from app.tools.web_search import WebSearchTool

tool = WebSearchTool()
print("=== BẮT ĐẦU TEST WEB SEARCH TOOL ===")
print("Tên Tool:", tool.name)
print("Mô tả:", tool.description)
print("\n--- Thực thi tìm kiếm: 'Giá trị Bitcoin hôm nay' ---")

# Dùng try/except xem tool có lỗi gì không
try:
    result = tool.execute("Giá trị Bitcoin hôm nay")
    print(result)
except Exception as e:
    print(f"Lỗi: {e}")
