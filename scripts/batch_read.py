"""
脚本2：批量读取 + 计时 + 统计报告
- 读取 day_1data/day1_data/input/ 下所有 txt 文件
- 统计总文件数、成功数、失败数、成功率、总耗时(秒)、平均耗时(毫秒)
"""

import time
from pathlib import Path


def main():
    # ---------- 路径配置 ----------
    base_dir = Path(__file__).resolve().parent.parent  # scripts/ -> root
    input_dir = base_dir / "data" / "day1_data" / "day1_data" / "input"

    # 检查输入目录是否存在
    if not input_dir.exists():
        print(f"❌ 目录不存在: {input_dir}")
        return

    # 获取所有 txt 文件并排序
    files = sorted(input_dir.glob("*.txt"))
    total = len(files)

    if total == 0:
        print("⚠️  目录中没有找到 .txt 文件")
        return

    print(f"📂 共发现 {total} 个 txt 文件，开始读取...\n")

    # ---------- 批量读取 + 计时 ----------
    success_count = 0
    failure_count = 0
    start_total = time.time()          # 总计时起点

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # 成功：打印文件名和内容长度
            print(f"  ✅ {file_path.name}  ({len(content)} 字符)")
            success_count += 1
        except Exception as e:
            print(f"  ❌ {file_path.name}  读取失败: {e}")
            failure_count += 1

    total_elapsed = time.time() - start_total   # 总耗时（秒）

    # ---------- 计算统计指标 ----------
    success_rate = (success_count / total) * 100
    avg_time_ms = (total_elapsed / total) * 1000  # 平均每文件耗时（毫秒）

    # ---------- 打印统计报告 ----------
    print("\n" + "=" * 45)
    print("            📊 批量读取统计报告")
    print("=" * 45)
    print(f"  总文件数:      {total}")
    print(f"  成功数:        {success_count}")
    print(f"  失败数:        {failure_count}")
    print(f"  成功率:        {success_rate:.2f}%")
    print(f"  总耗时:        {total_elapsed:.4f} 秒")
    print(f"  平均耗时:      {avg_time_ms:.2f} 毫秒/文件")
    print("=" * 45)


if __name__ == "__main__":
    main()
