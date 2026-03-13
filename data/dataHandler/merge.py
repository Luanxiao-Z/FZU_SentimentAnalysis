import pandas as pd
import os
import glob
import csv


def is_valid_csv(file_path):
    """
    简单判断文件是否为合法的 CSV 格式
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            sample = f.read(1024)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            return True
    except Exception:
        return False


def merge_all_csvs_in_directory(output_file):
    # 获取当前目录下所有 .csv 文件
    csv_files = glob.glob("*.csv")

    # 排除输出文件本身（避免循环合并）
    csv_files = [f for f in csv_files if f != output_file and f != os.path.basename(output_file)]

    # 过滤掉非 CSV 文件（双重保险）
    valid_csv_files = []
    for f in csv_files:
        if is_valid_csv(f):
            valid_csv_files.append(f)
        else:
            print(f"⚠️  跳过无效 CSV 文件: {f}")

    if not valid_csv_files:
        print("❌ 未找到有效的 CSV 文件")
        return

    print(f"🔍 找到 {len(valid_csv_files)} 个有效的 CSV 文件:")
    for f in valid_csv_files:
        print(f"  - {f}")

    dfs = []
    for f in valid_csv_files:
        try:
            df = pd.read_csv(f)
            print(f"✅ 成功读取: {f}, 形状: {df.shape}, 列名: {list(df.columns)}")
            dfs.append(df)
        except Exception as e:
            print(f"❌ 读取文件 {f} 失败: {e}")

    if not dfs:
        print("❌ 没有成功读取任何 CSV 文件")
        return

    # 检查所有文件的列名是否一致
    first_columns = set(dfs[0].columns)
    for i, df in enumerate(dfs[1:], 1):
        if set(df.columns) != first_columns:
            print(f"❌ 第 {i + 1} 个文件的列名与第一个文件不一致！")
            print(f"   文件: {valid_csv_files[i]}")
            print(f"   当前列: {set(df.columns)}")
            print(f"   期望列: {first_columns}")
            return

    # 打乱每个 DataFrame 的顺序
    for i in range(len(dfs)):
        dfs[i] = dfs[i].sample(frac=1, random_state=None).reset_index(drop=True)
        print(f"🔀 已打乱第 {i + 1} 个文件的顺序")

    # 合并所有 DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)

    # 再次打乱合并后的数据（确保整体随机）
    merged_df = merged_df.sample(frac=1, random_state=None).reset_index(drop=True)
    print(f"🔀 已打乱合并后的数据")

    # 保存结果
    try:
        merged_df.to_csv(output_file, index=False)
        print(f"✅ 合并完成，结果已保存到: {output_file}")
        print(f"📊 最终数据形状: {merged_df.shape}")
        print(f"📋 最终列名: {list(merged_df.columns)}")
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")


# 执行合并
if __name__ == "__main__":
    output_file = "output/data.csv"
    merge_all_csvs_in_directory(output_file)