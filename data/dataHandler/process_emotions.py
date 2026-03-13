import pandas as pd
import os
import sys

# ================= 配置区域 (硬编码文件名) =================
INPUT_FILE = "merged_output.csv"
OUTPUT_FILE = "new_train.csv"


# =========================================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到输入文件 '{INPUT_FILE}'")
        sys.exit(1)

    print(f"🚀 开始处理数据...")
    print(f"📂 输入文件：{INPUT_FILE}")
    print(f"💾 输出文件：{OUTPUT_FILE}")

    df = None
    read_success = False

    # 定义可能的编码和分隔符组合
    encodings = ['gbk', 'utf-8-sig', 'utf-8']
    separators = [',', '\t']

    # 尝试不同的组合来读取文件
    for enc in encodings:
        if read_success: break
        for sep in separators:
            try:
                # 尝试方案 1: 有表头 (text, labels)
                df = pd.read_csv(INPUT_FILE, sep=sep, usecols=['text', 'labels'], dtype={'text': str, 'labels': str},
                                 on_bad_lines='skip', encoding=enc)

                # 【修复点】Python 3.8 兼容写法：先计算变量
                sep_name = 'Tab' if sep == '\t' else 'Comma'
                print(f"✅ 读取成功：编码={enc}, 分隔符={sep_name}, 模式=有表头")

                read_success = True
                break
            except Exception:
                pass

            try:
                # 尝试方案 2: 无表头 (列 0, 1)
                df = pd.read_csv(INPUT_FILE, sep=sep, header=None, usecols=[0, 1], names=['text', 'labels'],
                                 dtype={'text': str, 'labels': str}, on_bad_lines='skip', encoding=enc)

                # 【修复点】Python 3.8 兼容写法
                sep_name = 'Tab' if sep == '\t' else 'Comma'
                print(f"✅ 读取成功：编码={enc}, 分隔符={sep_name}, 模式=无表头")

                read_success = True
                break
            except Exception:
                continue

    if not read_success or df is None:
        print(f"❌ 读取文件失败：无法识别编码或格式。已尝试编码：{encodings}")
        sys.exit(1)

    initial_count = len(df)
    print(f"📊 原始数据行数：{initial_count}")

    # ================= 核心处理逻辑 (向量化加速) =================

    # 确保 labels 是字符串类型并去除空格
    df['labels'] = df['labels'].astype(str).str.strip()

    # 过滤多标签 (包含逗号) 和 Neutral (27)
    is_multi_label = df['labels'].str.contains(',', na=False)
    is_neutral = df['labels'] == '27'

    mask_valid = (~is_multi_label) & (~is_neutral)

    df_clean = df[mask_valid].copy()

    skipped_count = initial_count - len(df_clean)
    print(f"⏭️  已跳过 {skipped_count} 行 (多标签或 Neutral)")

    if df_clean.empty:
        print("⚠️  警告：过滤后没有剩余数据！")
        return

    # 转换为整数
    df_clean['label_id'] = df_clean['labels'].astype(int)

    # 3. 构建映射字典
    mapping_dict = {}

    def add_mapping(ids, coarse_val, cn_name, basic_id_val):
        for i in ids:
            mapping_dict[i] = (coarse_val, cn_name, basic_id_val)

    # Anger (2): 2, 3, 10
    add_mapping([2, 3, 10], 0, "生气", 2)
    # Disgust (5): 11
    add_mapping([11], 0, "厌恶", 5)
    # Fear (4): 14, 19
    add_mapping([14, 19], 0, "恐惧", 4)
    # Sadness (0): 9, 12, 16, 24, 25
    add_mapping([25, 9, 12, 16, 24], 0, "悲伤", 0)
    # Joy (1): 0, 1, 4, 5, 8, 13, 15, 17, 18, 20, 21, 23
    add_mapping([17, 1, 4, 13, 15, 18, 20, 23, 21, 0, 8, 5], 1, "开心", 1)
    # Surprise (3): 6, 7, 22, 26 (ambiguous -> coarse=2)
    add_mapping([26, 22, 6, 7], 2, "惊讶", 3)

    # 4. 执行映射
    mapped_series = df_clean['label_id'].map(mapping_dict)
    df_final = df_clean[mapped_series.notna()].copy()

    if df_final.empty:
        print("❌ 错误：所有数据映射失败。")
        return

    # 拆分元组
    coarse_vals, name_vals, label_vals = zip(*df_final['label_id'].map(mapping_dict))

    df_final['coarse_label'] = coarse_vals
    df_final['emotion_name'] = name_vals
    df_final['label'] = label_vals

    final_df = df_final[['coarse_label', 'emotion_name', 'label', 'text']]

    # 5. 保存
    print("💾 正在保存结果...")
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"✅ 处理完成！")
    print(f"📈 最终有效数据量：{len(final_df)}")

    print("\n--- 情感分布统计 ---")
    print(final_df['emotion_name'].value_counts().to_string())
    print("\n--- 正负中性分布 ---")
    print(f"0 (负面): {(final_df['coarse_label'] == 0).sum()}")
    print(f"1 (正面): {(final_df['coarse_label'] == 1).sum()}")
    print(f"2 (模糊): {(final_df['coarse_label'] == 2).sum()}")


if __name__ == "__main__":
    main()