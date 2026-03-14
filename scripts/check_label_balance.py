# 统计 data.csv 与 OCEMOTION.csv 的类别分布，用于检查是否平衡
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(BASE_DIR)

try:
    import pandas as pd
except ImportError:
    print("请先安装 pandas: pip install pandas")
    sys.exit(1)

EMOTION_NAMES = {0: "悲伤", 1: "开心", 2: "生气", 3: "惊讶", 4: "恐惧", 5: "厌恶"}


def load_and_count(csv_path):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns and "emotion_name" in df.columns:
        name2id = {v: k for k, v in EMOTION_NAMES.items()}
        df["label"] = df["emotion_name"].map(name2id)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).astype({"label": int})
    df = df[df["label"].between(0, 5)]
    return df["label"].value_counts().sort_index(), len(df)


def main():
    print("=" * 60)
    print("数据集类别分布（是否平衡）")
    print("=" * 60)

    for name, path in [
        ("OCEMOTION.csv", os.path.join(BASE_DIR, "data", "originalData", "OCEMOTION.csv")),
        ("data.csv", os.path.join(BASE_DIR, "data", "data.csv")),
    ]:
        if not os.path.isfile(path):
            print(f"\n[{name}] 文件不存在: {path}\n")
            continue
        counts, total = load_and_count(path)
        print(f"\n【{name}】 总样本数: {total}")
        print("-" * 50)
        max_c, min_c = counts.max(), counts.min()
        for i in range(6):
            n = int(counts.get(i, 0))
            pct = 100 * n / total
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"  {EMOTION_NAMES[i]:4s}({i}): {n:6d}  ({pct:5.2f}%)  {bar}")
        imbalance = (max_c - min_c) / min_c if min_c > 0 else float("inf")
        print(f"  类别比例 最多/最少 ≈ {max_c}/{min_c} ≈ {imbalance:.2f}x")
        if imbalance > 2.0:
            print(f"  → 不平衡（建议关注少数类或使用类别权重）")
        else:
            print(f"  → 相对平衡")
    print()


if __name__ == "__main__":
    main()
