# 数据集智能平衡脚本：基于中位数目标对各类别执行欠采样/过采样，解决类别不平衡问题并打乱数据顺序

import pandas as pd
import os
from sklearn.utils import resample


def smart_balance_dataset(input_file, output_file):
    # 配置列名 (根据你的数据结构)
    COL_COARSE = 'coarse_label'
    COL_EMOTION_NAME = 'emotion_name'
    COL_LABEL = 'label'
    COL_TEXT = 'text'

    print(f"🚀 开始处理: {input_file}")

    # 1. 读取数据 (自动处理编码)
    try:
        if input_file.endswith('.csv'):
            try:
                df = pd.read_csv(input_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(input_file, encoding='gbk')
        elif input_file.endswith('.xlsx'):
            df = pd.read_excel(input_file)
        else:
            raise ValueError("不支持的文件格式，请使用 .csv 或 .xlsx")
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 清洗空值
    df = df.dropna(subset=[COL_TEXT, COL_LABEL])
    original_total = len(df)
    print(f"✅ 成功加载 {original_total:,} 条有效数据")

    # 2. 统计原始分布
    print("\n📊 [原始分布]")
    dist_orig = df[COL_EMOTION_NAME].value_counts()
    min_count = dist_orig.min()
    max_count = dist_orig.max()
    median_count = int(dist_orig.median())

    print(f"   最少类别 ({dist_orig.idxmin()}): {min_count:,}")
    print(f"   最多类别 ({dist_orig.idxmax()}): {max_count:,}")
    print(f"   🎯 设定目标数量 (中位数): {median_count:,}")

    # 3. 执行智能平衡 (混合策略)
    print(f"\n⚖️  正在执行平衡 (目标: 每类 {median_count:,} 条)...")

    df_balanced = pd.DataFrame()
    emotions = df[COL_EMOTION_NAME].unique()

    for emotion in emotions:
        df_class = df[df[COL_EMOTION_NAME] == emotion]
        current_count = len(df_class)

        if current_count > median_count:
            # --- 欠采样 (Discard) ---
            # 随机选取 median_count 条
            df_resampled = resample(df_class,
                                    replace=False,
                                    n_samples=median_count,
                                    random_state=42)
            # print(f"   - {emotion}: 丢弃 {current_count - median_count:,} 条")

        elif current_count < median_count:
            # --- 过采样 (Replicate) ---
            # 有放回抽样，补足到 median_count 条
            df_resampled = resample(df_class,
                                    replace=True,
                                    n_samples=median_count,
                                    random_state=42)
            # print(f"   - {emotion}: 复制生成 {median_count - current_count:,} 条")

        else:
            # 正好相等
            df_resampled = df_class

        df_balanced = pd.concat([df_balanced, df_resampled])

    # 4. 打乱数据顺序 (Shuffle)
    # 非常重要：防止模型学习到数据的排列顺序（如前1000条全是开心）
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    balanced_total = len(df_balanced)

    # 5. 输出最终统计
    print("\n✅ [平衡完成] 最终分布")
    print("-" * 40)
    final_dist = df_balanced[COL_EMOTION_NAME].value_counts().sort_index()

    print(f"{'情感':<10} | {'数量':>10} | {'状态'}")
    print("-" * 40)
    for emotion, count in final_dist.items():
        status = "平衡"
        print(f"{emotion:<10} | {count:>10,} | {status}")

    print("-" * 40)
    print(f"原始总数: {original_total:,}")
    print(f"平衡总数: {balanced_total:,}")
    if balanced_total > original_total:
        print(f"变化: ➕ 增加了 {balanced_total - original_total:,} 条 (通过复制少数类)")
    else:
        print(f"变化: ➖ 减少了 {original_total - balanced_total:,} 条 (通过丢弃多数类)")

    # 6. 保存文件
    try:
        if output_file.endswith('.csv'):
            # utf-8-sig 确保 Excel 打开不乱码
            df_balanced.to_csv(output_file, index=False, encoding='utf-8-sig')
        elif output_file.endswith('.xlsx'):
            df_balanced.to_excel(output_file, index=False)

        print(f"\n💾 成功保存至: {output_file}")
        print("✨ 所有工作已完成！可以直接使用该文件进行训练。")

    except Exception as e:
        print(f"❌ 保存失败: {e}")


if __name__ == "__main__":
    # ================= ⚙️ 配置区域 (只需修改这里) =================
    INPUT_FILE = 'data_expand.csv'  # 【修改】你的原始文件名
    OUTPUT_FILE = 'data_balanced.csv'  # 【修改】你想保存的新文件名
    # =============================================================

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到文件 '{INPUT_FILE}'")
        print("请确保脚本和数据文件在同一个文件夹内。")
    else:
        smart_balance_dataset(INPUT_FILE, OUTPUT_FILE)