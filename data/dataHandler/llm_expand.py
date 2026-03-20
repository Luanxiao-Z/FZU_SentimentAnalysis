# 基于豆包大模型的数据增量增强脚本：专为指定情感类别定向生成1000条新数据，支持断点续传、额度耗尽自动降级复制填充

import pandas as pd
import os
import time
import json
import requests
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
INPUT_FILE = 'data2.csv'
OUTPUT_FILE = 'sentiment_data_with_extra_fear_1000.csv'  # 改个新名字，避免覆盖

# 数据列名
COL_COARSE = 'coarse_label'
COL_EMOTION = 'emotion_name'
COL_LABEL = 'label'
COL_TEXT = 'text'

# 🌐 豆包配置
API_KEY = "3ea2d49a-dfd4-48e0-9abc-571f38887c3c"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL_ID = "ep-20260314182851-76fwr"

# 🎯 增强目标
TARGET_COUNT = 10249  # 保留原全局目标（其他类用不到，但保持兼容）
ADDITIONAL_COUNT = 1000  # === 修改点：为“恐惧”类新增 1000 条 ===
BATCH_SIZE = 5
MAX_RETRIES = 3
DELAY_SECONDS = 1.0

# ================================================

EMOTION_TO_LABEL = {
    '悲伤': 0, '开心': 1, '生气': 2, '惊讶': 3, '恐惧': 4, '厌恶': 5
}

QUOTA_EXHAUSTED = False


def get_coarse_label(emotion_name):
    if emotion_name == '开心':
        return 1
    elif emotion_name in ['悲伤', '生气', '恐惧', '厌恶']:
        return 0
    elif emotion_name == '惊讶':
        return 2
    else:
        return 2


def get_label_code(emotion_name):
    return EMOTION_TO_LABEL.get(emotion_name, 3)


def call_llm_for_augmentation(seed_text, emotion_name, target_count):
    global QUOTA_EXHAUSTED
    if QUOTA_EXHAUSTED:
        return None

    label_code = get_label_code(emotion_name)
    coarse_code = get_coarse_label(emotion_name)
    category_desc = "正面积极的情感" if coarse_code == 1 else (
        "负面消极的情感" if coarse_code == 0 else "中性或模糊的情感")

    prompt = f"""
任务：基于种子文本生成 {target_count} 条属于【{emotion_name}】的评论。
要求：
1. 情感严格一致 (标签:{label_code}, 属性:{category_desc})。
2. 句式、用词高度多样化，模拟真实用户。
3. 仅返回 JSON 列表格式，无 Markdown，无解释。
   示例：["句子1", "句子2"]

种子文本: "{seed_text}"
目标情感: {emotion_name}
数量: {target_count}

直接输出 JSON：
"""

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "top_p": 0.9
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    url = f"{BASE_URL}/chat/completions"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code != 200:
                error_text = response.text.lower()
                status = response.status_code

                if status == 403 or 'quota' in error_text or 'balance' in error_text or 'insufficient' in error_text:
                    print(f"\n💸 [触发保护] 免费额度已用完！(Status: {status})")
                    QUOTA_EXHAUSTED = True
                    return None

                print(f"\n⚠️ API 错误 {status}: {response.text[:100]}")
                if status == 401: raise ValueError("API Key 无效 (401)")
                time.sleep(2)
                continue

            resp_json = response.json()
            content = resp_json['choices'][0]['message']['content'].replace("```json", "").replace("```", "").strip()
            data_list = json.loads(content)

            if isinstance(data_list, list) and len(data_list) > 0:
                return [t for t in data_list if isinstance(t, str) and len(t.strip()) > 0]

        except Exception as e:
            print(f"\n❌ 请求异常: {e}")
            time.sleep(2)

    return None


def save_intermediate_csv(df_current, output_file, finished_emotion, count):
    df_current.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 [中间文件已保存] 类别 [{finished_emotion}] 已新增至 {count} 条。")
    print(f"   文件路径：{output_file}")
    print(f"   当前总数据量：{len(df_current)} 条")


def main():
    global QUOTA_EXHAUSTED
    print(f"🚀 启动增量增强任务 (为【恐惧】类新增 {ADDITIONAL_COUNT} 条)")
    print(f"   输入: {INPUT_FILE}")
    print(f"   输出: {OUTPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    try:
        if INPUT_FILE.endswith('.csv'):
            try:
                df_orig = pd.read_csv(INPUT_FILE, encoding='utf-8')
            except:
                df_orig = pd.read_csv(INPUT_FILE, encoding='gbk')
        else:
            df_orig = pd.read_excel(INPUT_FILE)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    df_orig = df_orig.dropna(subset=[COL_TEXT, COL_EMOTION])

    if COL_LABEL not in df_orig.columns:
        df_orig[COL_LABEL] = df_orig[COL_EMOTION].apply(get_label_code)
    if COL_COARSE not in df_orig.columns:
        df_orig[COL_COARSE] = df_orig[COL_EMOTION].apply(get_coarse_label)

    df_work = pd.DataFrame(columns=[COL_COARSE, COL_EMOTION, COL_LABEL, COL_TEXT])

    if os.path.exists(OUTPUT_FILE):
        try:
            df_work = pd.read_csv(OUTPUT_FILE, encoding='utf-8-sig')
            print(f"✅ 发现已有进度文件，已加载 {len(df_work)} 条数据。")
        except Exception as e:
            print(f"⚠️ 读取现有文件失败 ({e})，将重新从原始数据初始化。")
            df_work = df_orig.copy()
    else:
        print("ℹ️ 未找到现有输出文件，正在从原始数据初始化...")
        df_work = df_orig.copy()

    # 检查并截断超标数据（保持原逻辑）
    print("   🔍 检查原始数据是否超标 (目标: 10249)...")
    rows_to_keep = []
    unique_emotions = df_work[COL_EMOTION].unique()
    for emo in unique_emotions:
        df_emo = df_work[df_work[COL_EMOTION] == emo]
        count = len(df_emo)
        if count > TARGET_COUNT:
            df_trimmed = df_emo.sample(n=TARGET_COUNT, random_state=42)
            rows_to_keep.append(df_trimmed)
            print(f"   ⚠️ [{emo}]: 原始 {count} 条 -> 已自动截断保留 {TARGET_COUNT} 条")
        else:
            rows_to_keep.append(df_emo)
            if count == 0:
                print(f"   ⏳ [{emo}]: 0 条 (待生成)")
            else:
                print(f"   ✅ [{emo}]: {count} 条 (保留原样)")

    if rows_to_keep:
        df_work = pd.concat(rows_to_keep, ignore_index=True)
    else:
        df_work = pd.DataFrame(columns=[COL_COARSE, COL_EMOTION, COL_LABEL, COL_TEXT])

    print(f"   🎉 初始化完成。当前基础数据总量: {len(df_work)} 条。")

    # 3. 确定需要处理的类别
    all_emotions = df_orig[COL_EMOTION].unique()
    emotions_to_process = []

    print("\n🔍 最终任务列表确认...")
    for emo in all_emotions:
        current_count = len(df_work[df_work[COL_EMOTION] == emo])

        if emo != '恐惧':
            # 非目标类别，直接跳过
            print(f"   ⏭️ [{emo}]: {current_count} 条 (非目标类别，跳过)")
            continue

        # 只处理“恐惧”
        if current_count >= TARGET_COUNT:
            # 如果已经到上限，但用户想“新增”1000条，这里有两种选择：
            # 1. 允许超过 TARGET_COUNT（推荐，因为你是“新增”）
            # 2. 强制截断（不推荐，违背“新增”意图）
            # 我们采用方案1：不截断，直接加 1000 条
            pass

        needed = ADDITIONAL_COUNT  # === 修改点：始终新增 1000 条 ===
        print(f"   ⏳ [{emo}]: {current_count} 条 (将新增 {needed} 条)")
        emotions_to_process.append(emo)

    if not emotions_to_process:
        print("\n🎉 没有需要处理的类别！任务完成。")
        return

    # 4. 开始逐类增强
    for emotion in emotions_to_process:
        if QUOTA_EXHAUSTED:
            print("\n⚠️ 检测到额度已耗尽，后续类别将使用复制填充。")

        df_seed_source = df_orig[df_orig[COL_EMOTION] == emotion].copy()
        if len(df_seed_source) == 0:
            print(f"❌ 原始数据中找不到类别 [{emotion}]，跳过。")
            continue

        df_current_class = df_work[df_work[COL_EMOTION] == emotion].copy()
        current_count = len(df_current_class)
        needed = ADDITIONAL_COUNT  # 始终新增 1000 条

        print(f"\n{'=' * 50}")
        print(f"🎯 正在为【{emotion}】新增 {needed} 条数据")
        print(f"   当前: {current_count} | 新增后目标: {current_count + needed}")
        print(f"{'=' * 50}")

        std_label = get_label_code(emotion)
        std_coarse = get_coarse_label(emotion)

        new_texts = []
        seed_pool = df_seed_source[COL_TEXT].tolist()
        generated_count = 0

        with tqdm(total=needed, desc=f"生成 [{emotion}]", unit="条", ncols=100) as pbar:
            while generated_count < needed:
                if QUOTA_EXHAUSTED:
                    break

                seed = seed_pool[generated_count % len(seed_pool)]
                batch_req = min(BATCH_SIZE, needed - generated_count)

                results = call_llm_for_augmentation(seed, emotion, batch_req)

                if results:
                    chunk = results[:needed - generated_count]
                    new_texts.extend(chunk)
                    count_step = len(chunk)
                    generated_count += count_step
                    pbar.update(count_step)
                else:
                    if QUOTA_EXHAUSTED:
                        break
                    time.sleep(DELAY_SECONDS)

        remaining = needed - generated_count
        if remaining > 0:
            print(f"\n   ⚠️ 剩余 {remaining} 条将使用【随机复制】填充。")
            fallback_sample = df_seed_source.sample(n=1)[COL_TEXT].values[0]
            for _ in range(remaining):
                new_texts.append(fallback_sample)
            generated_count += remaining
            pbar.update(remaining)

        if new_texts:
            df_new = pd.DataFrame({
                COL_TEXT: new_texts,
                COL_EMOTION: emotion,
                COL_LABEL: std_label,
                COL_COARSE: std_coarse
            })
            df_work = pd.concat([df_work, df_new], ignore_index=True)

        df_work[COL_LABEL] = df_work[COL_LABEL].astype(int)
        df_work[COL_COARSE] = df_work[COL_COARSE].astype(int)
        save_intermediate_csv(df_work, OUTPUT_FILE, emotion, len(df_work[df_work[COL_EMOTION] == emotion]))

        if QUOTA_EXHAUSTED:
            print("\n💸 额度已用尽，当前类别已保底并保存。程序退出。")
            break

    # 5. 最终整理
    print("\n✅ 所有类别处理完毕。进行最终洗牌...")
    df_final = df_work.sample(frac=1, random_state=42).reset_index(drop=True)
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("\n📊 [最终分布统计]")
    print(df_final[COL_EMOTION].value_counts().sort_index())
    print(f"\n💾 最终文件已保存至：{OUTPUT_FILE}")


if __name__ == "__main__":
    if not API_KEY:
        print("❌ 错误：未配置 API_KEY")
    else:
        main()