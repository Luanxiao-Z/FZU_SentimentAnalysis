"""
音频->ASR->情感推理 统一链路自测脚本

将原先 `test_asr_processor.py` 的离线能力自测（ffmpeg 分段、16k/单声道标准化）
合并到本文件中，达到“ASR 测试文件合一，只保留一个脚本”的目的。

目的：
1) 模拟 `pages/1_单条文本分析.py` 点击“开始分析”时的后端逻辑：
   audio_to_text(audio) -> handler.predict(transcript)
2) 做“真实执行 + 结构校验”，不写死结果文本，不伪造预测值。
3) 在不提供 `--audio` 时，完成“离线音频处理管线自测”（不需要 ASR key）。

检查项（全部来自程序真实返回）：
- transcript 必须非空，且尽量包含中文字符
- handler.predict 返回字段齐全且类型正确
- top3 元素结构为 (emotion_name: str, probability: float)
- all_probabilities 包含 6 类情感概率

用法：
1) 只做离线 ASR 音频处理自测（不需要 ASR key）
python scripts/test_asr_processor.py

2) 真实跑 ASR + 情感推理（需要 `config/asr_secrets.toml`）
python scripts/test_asr_processor.py --audio "你的.wav或.mp3"

可选：
python scripts/test_asr_processor.py --audio "你的.wav" --config "config/asr_secrets.toml"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH, EMOTION_NAMES, COARSE_MAP
from src.model_handler import EmotionModelHandler
from src.utils.asr_processor import audio_to_text, AsrError, segment_wav, normalize_audio_to_wav_16k_mono


_CHN_RE = re.compile(r"[\u4e00-\u9fff]")


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def _write_sine_wav(
    path: Path,
    *,
    duration_s: float,
    sample_rate: int = 16000,
    freq_hz: float = 440.0,
) -> None:
    """写入简单正弦波 wav，用于验证 ffmpeg/切片管线（离线自测）。"""
    import wave
    import math

    total_samples = int(duration_s * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        for i in range(total_samples):
            v = 0.25 * math.sin(2 * math.pi * freq_hz * (i / sample_rate))
            sample = int(max(-1.0, min(1.0, v)) * 32767)
            wf.writeframesraw(sample.to_bytes(2, byteorder="little", signed=True))


def _read_wav_header(path: Path) -> tuple[int, int, float]:
    """读取 wav header：channels / sample_rate / duration_s。"""
    import wave

    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        framerate = wf.getframerate()
        frames = wf.getnframes()
        duration_s = frames / float(framerate) if framerate else 0.0
        sampwidth = wf.getsampwidth()
        _ = sampwidth  # 当前断言不需要
        return channels, framerate, duration_s


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified ASR audio self-test + optional ASR->emotion pipeline test")
    parser.add_argument(
        "--audio",
        required=False,
        default=None,
        help="真实音频文件路径（wav/mp3）。不提供则只做离线切片/标准化自测。",
    )
    parser.add_argument("--config", default=None, help="ASR secrets TOML 路径（默认 config/asr_secrets.toml）")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else (PROJECT_ROOT / "config" / "asr_secrets.toml")
    secrets_ok = config_path.exists()

    # ---------- Stage A: 离线 wav 分段测试（不依赖 key） ----------
    print("[Stage A] segment_wav on synthetic wav ...")
    import tempfile

    with tempfile.TemporaryDirectory(prefix="asr_test_") as td:
        td_path = Path(td)
        wav_in = td_path / "synthetic_16k_mono.wav"
        _write_sine_wav(wav_in, duration_s=6.2, sample_rate=16000, freq_hz=440.0)

        chunks = segment_wav(wav_in, chunk_seconds=2)
        try:
            assert chunks, "segment_wav 返回空 chunk 列表"
            assert len(chunks) >= 2, f"切片段数过少：{len(chunks)}"
            print(f"[OK] segment_wav chunks={len(chunks)}")
        finally:
            for p in chunks:
                try:
                    if p and Path(p).exists():
                        Path(p).unlink()
                except Exception:
                    pass

    # ---------- Stage B: 标准化为 16k/单声道（需要 secrets 存在） ----------
    if secrets_ok:
        print("[Stage B] normalize_audio_to_wav_16k_mono header check ...")
        with tempfile.TemporaryDirectory(prefix="asr_test_norm_") as td2:
            td2_path = Path(td2)
            wav_in = td2_path / "synthetic_in.wav"
            _write_sine_wav(wav_in, duration_s=1.8, sample_rate=22050, freq_hz=330.0)

            normalized_path = normalize_audio_to_wav_16k_mono(
                wav_in,
                input_suffix_hint=".wav",
                config_path=config_path,
            )
            normalized = Path(normalized_path)
            try:
                channels, framerate, duration_s = _read_wav_header(normalized)
                assert channels == 1, f"期望 channels=1，但实际={channels}"
                assert framerate == 16000, f"期望 sample_rate=16000，但实际={framerate}"
                assert duration_s > 0.1, f"标准化后时长异常：{duration_s}"
                print(f"[OK] normalized channels={channels} rate={framerate} duration={duration_s:.2f}s")
            finally:
                try:
                    if normalized.exists():
                        normalized.unlink()
                except Exception:
                    pass
    else:
        print(f"[Stage B] SKIP：未找到 secrets：{config_path}（无法进行标准化 header 检查）")

    # 离线自测结束：如果没有传 --audio，则不进行真实 ASR + 情感推理
    if not args.audio:
        print("[OK] 离线 ASR 能力自测完成（未传 --audio，跳过真实 ASR 与情感推理）。")
        return 0

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[ERR] 音频不存在：{audio_path}")
        return 2

    if not secrets_ok:
        print(f"[ERR] 未找到 ASR 私密配置：{config_path}")
        print("请先准备 config/asr_secrets.toml，然后再运行该测试。")
        return 2

    # ---------- Stage C1: ASR 转写（真实调用） ----------
    try:
        transcript = audio_to_text(audio_path, config_path=config_path)
    except AsrError as e:
        print(f"[ASR ERROR] {e}")
        if e.details is not None:
            print(f"[DETAILS] {e.details}")
        return 1
    except Exception as e:
        print(f"[ERR] ASR 调用异常：{e}")
        return 1

    transcript = (transcript or "").strip()
    _assert(transcript, "ASR transcript 为空（识别失败或返回为空字符串）。")

    # 不要求“必须有中文”，但至少尽量应包含中文
    if not _CHN_RE.search(transcript):
        # 这里不直接 fail（极端情况下可能识别为数字/英文），但要提示
        print(f"[WARN] transcript 未检测到明显中文字符，当前 transcript 前缀：{transcript[:80]!r}")

    print("[OK] ASR transcript 获取成功，前缀如下：")
    print(transcript[:200])

    # ---------- Stage C2: 情感推理（真实模型推理） ----------
    model_dir = Path(str(MODEL_PATH))
    # Transformers 在目录内通常会寻找这些默认权重文件名；
    # 没有权重时继续跑只会报 OSError，所以先做明确检查。
    has_weights = any(
        [
            (model_dir / "pytorch_model.bin").exists(),
            (model_dir / "model.safetensors").exists(),
            # 兜底：部分训练/导出可能叫其他 bin 名，优先让你知道当前目录确实没有常规权重文件
            any(model_dir.glob("*.bin")),
            any(model_dir.glob("*.safetensors")),
        ]
    )
    if not has_weights:
        print(f"[ERR] 未找到情感模型权重文件：{model_dir}")
        print("请按 README 将训练好的模型权重放到该目录，并确保至少包含：")
        print("- `pytorch_model.bin`（推荐）或 `model.safetensors`")
        return 2

    handler = EmotionModelHandler(model_path=str(MODEL_PATH))
    handler.load_model()

    try:
        result = handler.predict(transcript)
    except Exception as e:
        print(f"[ERR] 情感推理失败：{e}")
        return 1

    # ---------- Stage C3: 返回结构校验（用于证明 UI 不会崩） ----------
    required_keys = ["fine_id", "fine_name", "coarse", "confidence", "all_probabilities", "top3"]
    for k in required_keys:
        _assert(k in result, f"predict 返回缺少字段：{k}")

    fine_id = result["fine_id"]
    try:
        fine_id_int = int(fine_id)
    except Exception as e:
        raise AssertionError(f"fine_id 无法转换为 int：{fine_id!r}") from e

    _assert(fine_id_int in range(6), f"fine_id 不在 0~5：{fine_id_int}")
    _assert(result["fine_name"] in EMOTION_NAMES.values(), f"fine_name 不在已知情感集合：{result['fine_name']!r}")
    _assert(result["coarse"] in COARSE_MAP.values(), f"coarse 不在已知集合：{result['coarse']!r}")

    conf = result["confidence"]
    try:
        conf_val = float(conf)
    except Exception as e:
        raise AssertionError(f"confidence 无法转 float：{conf!r}") from e
    _assert(0.0 <= conf_val <= 1.0, f"confidence 不在 0~1：{conf_val}")

    all_probs = result["all_probabilities"]
    _assert(isinstance(all_probs, dict), "all_probabilities 必须是 dict")
    _assert(len(all_probs) == 6, f"all_probabilities 应为 6 类，但当前={len(all_probs)}")
    for k, v in all_probs.items():
        _assert(k in EMOTION_NAMES.values(), f"all_probabilities key 非已知情感：{k!r}")
        _assert(isinstance(v, (float, int)), f"all_probabilities value 非数字：{v!r}")

    top3 = result["top3"]
    _assert(isinstance(top3, list), "top3 必须是 list")
    _assert(len(top3) == 3, f"top3 应为 3 项，但当前={len(top3)}")
    for item in top3:
        _assert(isinstance(item, tuple) and len(item) == 2, f"top3 每项应为 (str, float)，当前={item!r}")
        name, prob = item
        _assert(isinstance(name, str) and name in EMOTION_NAMES.values(), f"top3 emotion_name 非已知情感：{name!r}")
        _assert(isinstance(prob, (float, int)), f"top3 prob 非数字：{prob!r}")

    print("[OK] audio->ASR->predict 全链路完成，返回结构可直接用于 UI 展示。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

