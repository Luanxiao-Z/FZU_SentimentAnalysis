# 百度大模型(AIT)多线程翻译脚本：调用百度AI文本翻译接口，通过自定义Prompt优化情感与人名翻译，支持并发处理、智能限流重试及进度监控，将英文数据翻译为中文

import pandas as pd
import requests
import random
import time
import os
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= ⚙️ 优化后的配置区域 =================

# --- 1. 认证信息 (保持不变) ---
BAIDU_TRANSLATION_API_KEY = 'T8d0_d6mog3o852cal84921qg'
APP_ID = '20260308002568413'

# --- 2. 功能配置 (重点优化这里) ---
MODEL_TYPE = 'llm'       # 必须使用 llm

# 🔥 优化点 1: 加入具体的翻译指令 (Prompt)
# 针对您之前的句子 (含情感和人名)，推荐使用以下指令：
TRANSLATION_INSTRUCTION = (
    "请扮演一位专业的文学翻译家。"
    "1. 翻译要自然流畅，符合中文表达习惯，严禁出现生硬的翻译腔。"
    "2. 准确保留原文的情感色彩（如歉意、喜爱、幽默等）。"
    "3. 遇到人名、地名或专有名词（如 Sapphira, Cirilla, Scarlett），请音译为常用的中文译名，不要意译。"
    "4. 只输出翻译后的中文，不要包含任何解释。"
)

# 🔥 优化点 2: 如果有特定术语，请在控制台上传后开启此项
USE_TERMINOLOGY = False  # 如果没上传术语库，请保持 False，否则可能报错

# --- 3. 文件与任务配置 ---
INPUT_FILE = 'train.csv'
OUTPUT_FILE = 'translated_result_ait_optimized.csv' # 换个文件名方便对比

TEST_MODE = False
TEST_ROWS = 50

# --- 4. 性能配置 (优化稳定性) ---
SOURCE_LANG = 'en'  # 强制指定源语言，不要用 auto
TARGET_LANG = 'zh'
MAX_WORKERS = 5     # 🔥 优化点 3: 稍微降低并发，保证每个请求都有充足的 Token 和时间
TIMEOUT = 45        # 🔥 优化点 4: 增加超时时间，防止大模型思考慢被切断

# ===========================================

# 全局变量
qps_error_count = 0
lock = threading.Lock()


def translate_ait(text):
    """调用百度大模型文本翻译 API (AIT)"""
    global qps_error_count

    if not isinstance(text, str) or not text.strip():
        return text

    url = 'https://fanyi-api.baidu.com/ait/api/aiTextTranslate'

    # 构造请求体 (JSON)
    payload = {
        'appid': APP_ID,
        'q': text,
        'from': SOURCE_LANG,
        'to': TARGET_LANG,
        'model_type': MODEL_TYPE
    }

    # 可选参数
    if TRANSLATION_INSTRUCTION:
        payload['reference'] = TRANSLATION_INSTRUCTION

    if USE_TERMINOLOGY:
        payload['needIntervene'] = 1

    # 请求头 (使用 Bearer Token 鉴权)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {BAIDU_TRANSLATION_API_KEY}'
    }

    # 最多重试 3 次
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=TIMEOUT)

            # 处理 HTTP 状态码
            if response.status_code != 200:
                if response.status_code == 429:  # QPS 限流
                    with lock:
                        qps_error_count += 1
                    time.sleep(random.uniform(1.5, 3.0))
                    continue
                else:
                    print(f"\n⚠️ HTTP 错误: {response.status_code} - {response.text}")
                    return text

            result = response.json()

            # 处理业务错误
            if 'error_code' in result:
                code = result['error_code']
                msg = result.get('error_msg', '')

                if code == '54001':  # 签名/鉴权错误
                    print(f"\n❌ 鉴权失败: {msg} (请检查 API Key 和 AppID)")
                    return text
                elif code == '52001':  # QPS 超限 (业务层)
                    with lock:
                        qps_error_count += 1
                    time.sleep(random.uniform(1.5, 3.0))
                    continue
                else:
                    # 其他错误打印一次即可，避免刷屏
                    if attempt == 0:
                        print(f"\n⚠️ API 错误 {code}: {msg}")
                    time.sleep(0.5)
                    continue

            # 解析成功结果
            # 返回格式与标准 API 类似: { "trans_result": [ { "src": "...", "dst": "..." } ] }
            if 'trans_result' in result and len(result['trans_result']) > 0:
                return result['trans_result'][0]['dst']
            else:
                return text

        except Exception as e:
            time.sleep(0.5)
            continue

    return text


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到文件 '{INPUT_FILE}'")
        return

    # 1. 读取数据
    print("📂 正在读取数据...")
    df = None
    for enc in ['utf-8-sig', 'gbk', 'utf-8']:
        try:
            df = pd.read_csv(INPUT_FILE, encoding=enc)
            break
        except:
            continue

    if df is None:
        print("❌ 无法读取文件，请检查编码。")
        return

    total_rows = len(df)

    # 2. 确定翻译范围
    if TEST_MODE:
        limit = min(TEST_ROWS, total_rows)
        print(f"🧪 [测试模式] 仅翻译前 {limit} 行数据...")
        target_df = df.iloc[:limit].copy()
        output_name = OUTPUT_FILE.replace('.csv', '_test_sample.csv')
    else:
        limit = total_rows
        print(f"🚀 [全量模式] 将翻译全部 {total_rows} 行数据...")
        target_df = df.copy()
        output_name = OUTPUT_FILE

    texts_to_translate = target_df.iloc[:, 0].tolist()
    results = [None] * len(texts_to_translate)

    model_desc = "大模型" if MODEL_TYPE == 'llm' else "传统机器"
    print(f"⚡ 启动 {model_desc} 翻译 (并发数: {MAX_WORKERS})...")
    if TRANSLATION_INSTRUCTION:
        print(f"   📝 翻译指令: {TRANSLATION_INSTRUCTION}")

    start_time = time.time()

    # 3. 执行翻译
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(translate_ait, text): i
            for i, text in enumerate(texts_to_translate)
        }

        completed = 0
        total = len(texts_to_translate)

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                results[index] = texts_to_translate[index]

            completed += 1
            # 进度显示
            if completed % 10 == 0 or completed == total:
                print(f"   ⏳ 进度: {completed}/{total}", end='\r')

    end_time = time.time()

    # 4. 写回数据并保存
    target_df.iloc[:, 0] = results
    target_df.to_csv(output_name, index=False, encoding='utf-8-sig')

    duration = end_time - start_time
    speed = len(texts_to_translate) / duration if duration > 0 else 0

    print("\n" + "=" * 40)
    print(f"✅ 任务完成！")
    print(f"📄 输出文件: {output_name}")
    print(f"⏱️  耗时: {duration:.2f} 秒 | 速度: {speed:.2f} 行/秒")
    if qps_error_count > 0:
        print(f"⚠️  触发限流重试: {qps_error_count} 次")

    # 5. 预览结果
    print("\n👀 结果预览 (前 3 行):")
    print(target_df.head(3).to_string())
    print("=" * 40)

    if TEST_MODE:
        print("\n💡 提示: 当前是测试模式。如果效果满意，请修改脚本中:")
        print(f"   TEST_MODE = False")
        print("   然后重新运行以翻译全部数据。")


if __name__ == "__main__":
    main()