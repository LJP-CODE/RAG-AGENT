"""
脚本1：生成测试文件
- 在 day_1data/day1_data/input/ 下生成 100 个 txt 文件
- 文件名 file_001.txt ~ file_100.txt
- 每个文件内容为中文测试语句
"""

from pathlib import Path

# ---------- 路径配置 ----------
# 定位到当前脚本所在项目根目录
base_dir = Path(__file__).resolve().parent.parent  # scripts/ -> root

# 输入目录（如果目录尚不存在则自动创建）
input_dir = base_dir / "data" / "day1_data" / "day1_data" / "input"
input_dir.mkdir(parents=True, exist_ok=True)

# ---------- 生成 100 个文件 ----------
total_files = 100

for i in range(1, total_files + 1):
    # 构造文件名，例如 file_001.txt
    file_name = f"file_{i:03d}.txt"
    file_path = input_dir / file_name

    # 文件内容：中文测试语句
    content = f"这是第{i}个测试文件，用于批量读取实验。\n"

    # 写入文件（UTF-8 编码）
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print(f"✅ 成功生成 {total_files} 个测试文件到: {input_dir}")
print(f"   文件范围: file_001.txt ~ file_{total_files:03d}.txt")
