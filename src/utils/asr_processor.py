# 【成员】： 102300403朱茜茜（ASR接口负责人）
# 任务：
# 1.申请百度/讯飞语音识别（ASR）API Key。
# 2.实现音频文件处理逻辑，支持MP3/WAV格式。
# 3.针对长音频，实现分段切片上传与结果拼接逻辑，确保长文本完整性。
# 产出：
# utils/asr_processor.py，支持音频输入，返回文本字符串。
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import requests


class AsrError(RuntimeError):
    """ASR 处理失败。"""

    def __init__(self, message: str, *, provider: str | None = None, details: Any | None = None):
        super().__init__(message)
        self.provider = provider
        self.details = details


@dataclass(frozen=True)
class AsrSettings:
    provider: str = "baidu"
    timeout: float = 18.0
    dev_pid: int = 1537  # 普通话默认
    chunk_seconds: int = 45  # 按长度分段，避免单次请求太长
    # 百度 VOP server_api 常用音频约束：wav/pcm + 16k/单声道
    target_sample_rate: int = 16000
    target_channels: int = 1

    baidu_api_key: str | None = None
    baidu_secret_key: str | None = None


_TOKEN_CACHE: dict[str, Any] = {"token": None, "expires_at": 0.0}


def _default_config_path() -> Path:
    # 约定：repo_root/config/asr_secrets.toml
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "config" / "asr_secrets.toml"


def _load_settings(*, config_path: str | Path | None) -> AsrSettings:
    path = Path(config_path) if config_path is not None else _default_config_path()
    if not path.exists():
        raise AsrError(
            "未找到 ASR 私密配置文件。请将 `config/asr_secrets.example.toml` 复制为 "
            "`config/asr_secrets.toml` 并填写密钥。",
            provider=None,
            details={"expected_path": str(path)},
        )

    try:
        import tomllib  # py3.11+

        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        try:
            import tomli  # type: ignore
        except Exception as e:
            raise AsrError(
                "当前 Python 不支持 tomllib，且未安装 tomli。请执行：pip install tomli",
                details=str(e),
            ) from e

        data = tomli.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise AsrError("读取 ASR 配置失败（TOML 格式可能有误）。", details=str(e)) from e

    asr_cfg = (data.get("asr") or {}) if isinstance(data, dict) else {}
    baidu_cfg = (data.get("baidu_asr") or {}) if isinstance(data, dict) else {}

    def _get_int(d: Any, k: str, default: int) -> int:
        try:
            return int(d.get(k, default))
        except Exception:
            return default

    def _get_float(d: Any, k: str, default: float) -> float:
        try:
            return float(d.get(k, default))
        except Exception:
            return default

    provider = str(asr_cfg.get("provider", "baidu")).strip().lower()
    return AsrSettings(
        provider=provider,
        timeout=_get_float(asr_cfg, "timeout", 18.0),
        dev_pid=_get_int(asr_cfg, "dev_pid", 1537),
        chunk_seconds=_get_int(asr_cfg, "chunk_seconds", 45),
        target_sample_rate=_get_int(asr_cfg, "target_sample_rate", 16000),
        target_channels=_get_int(asr_cfg, "target_channels", 1),
        baidu_api_key=str(baidu_cfg.get("api_key")).strip() if baidu_cfg.get("api_key") else None,
        baidu_secret_key=str(baidu_cfg.get("secret_key")).strip() if baidu_cfg.get("secret_key") else None,
    )


def _read_input_bytes(audio: str | Path | bytes | BinaryIO) -> bytes:
    if isinstance(audio, (str, Path)):
        return Path(audio).read_bytes()
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    if hasattr(audio, "read"):
        try:
            audio.seek(0)  # type: ignore[attr-defined]
        except Exception:
            pass
        data = audio.read()  # type: ignore[call-arg]
        if isinstance(data, str):
            data = data.encode("utf-8")
        return bytes(data)
    raise AsrError("不支持的音频输入类型。")


def _ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise AsrError(
            "系统中未找到 `ffmpeg`。请先安装并配置到环境变量（PATH）。",
            provider=None,
        )


def _write_bytes_to_temp_file(tmp_dir: str | Path, data: bytes, suffix: str) -> Path:
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    name = f"input_{uuid.uuid4().hex}{suffix}"
    p = tmp_dir / name
    p.write_bytes(data)
    return p


def normalize_audio_to_wav_16k_mono(
    audio: str | Path | bytes | BinaryIO,
    *,
    input_suffix_hint: str = ".wav",
    config_path: str | Path | None = None,
) -> str:
    """
    把 MP3/WAV 等音频统一转成满足百度 VOP 的 wav：16kHz / 单声道 / PCM 16bit。
    返回：标准 wav 的文件路径（字符串）。
    """
    _ensure_ffmpeg_available()
    settings = _load_settings(config_path=config_path)

    raw = _read_input_bytes(audio)
    tmp_dir = Path(os.environ.get("TMPDIR") or Path.cwd()) / "asr_tmp"
    # 不使用固定目录：避免并发冲突；仍保留 fallback。
    # 这里尽量使用当前进程临时目录。
    import tempfile

    with tempfile.TemporaryDirectory(prefix="asr_", suffix="_tmp") as td:
        td_path = Path(td)
        input_path = _write_bytes_to_temp_file(td_path, raw, suffix=input_suffix_hint)
        out_wav = td_path / f"normalized_{input_path.stem}.wav"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            str(settings.target_channels),
            "-ar",
            str(settings.target_sample_rate),
            "-acodec",
            "pcm_s16le",
            str(out_wav),
        ]
        # 调试：打印将要执行的命令和当前工作目录
        print(f"[DEBUG] 执行命令: {' '.join(cmd)}")
        print(f"[DEBUG] 当前工作目录: {os.getcwd()}")
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 调试：打印子进程的结果
        print(f"[DEBUG] ffmpeg 返回码: {r.returncode}")
        print(f"[DEBUG] 输出文件是否存在: {out_wav.exists()}")
        print(f"[DEBUG] ffmpeg stderr: {r.stderr.decode('utf-8', errors='ignore')}")
        if r.returncode != 0 or not out_wav.exists():
            raise AsrError(
                "音频标准化失败（ffmpeg 转换出错）。",
                provider=settings.provider,
                details={"stderr": (r.stderr.decode("utf-8", errors="ignore")[:800])},
            )

        # 临时目录会在 with 块结束后自动清理；
        # 因此复制到一个“可持续存在”的临时文件路径外。
        import tempfile

        final_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        final_path = Path(final_tmp.name)
        final_tmp.close()
        final_path.write_bytes(out_wav.read_bytes())
        return str(final_path)


def segment_wav(
    wav_path: str | Path,
    *,
    chunk_seconds: int,
) -> list[str]:
    """把标准 wav 切分成多个 wav chunk，返回 chunk 路径列表（按文件名排序）。"""
    _ensure_ffmpeg_available()
    wav_path = str(wav_path)

    import tempfile

    with tempfile.TemporaryDirectory(prefix="asr_seg_") as out_dir:
        out_dir_path = Path(out_dir)
        # Windows 下路径含反斜杠，直接用 List 传参即可。
        # -reset_timestamps 1：确保每段时间从 0 开始
        pattern = str(out_dir_path / "chunk_%03d.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            wav_path,
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            pattern,
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            raise AsrError(
                "音频分段失败（ffmpeg segment 出错）。",
                provider=None,
                details={"stderr": (r.stderr.decode("utf-8", errors="ignore")[:800])},
            )

        chunk_paths = sorted(str(p) for p in out_dir_path.glob("chunk_*.wav"))
        if not chunk_paths:
            raise AsrError("音频分段后未生成任何 chunk。", provider=None)

        # 将 chunk 拷贝到外部临时文件，避免 TemporaryDirectory 清理后路径失效。
        # 复制后返回外部稳定路径。
        final_dir = Path(tempfile.mkdtemp(prefix="asr_seg_final_"))
        final_paths: list[str] = []
        try:
            for p in chunk_paths:
                src = Path(p)
                dst = final_dir / src.name
                dst.write_bytes(src.read_bytes())
                final_paths.append(str(dst))
            return final_paths
        except Exception:
            # 若复制失败，清理 final_dir
            try:
                shutil.rmtree(final_dir, ignore_errors=True)
            finally:
                raise


def _baidu_get_access_token(*, api_key: str, secret_key: str, timeout: float) -> str:
    now = time.time()
    cached = _TOKEN_CACHE.get("token")
    expires_at = float(_TOKEN_CACHE.get("expires_at") or 0.0)
    if isinstance(cached, str) and cached and now < (expires_at - 30):
        return cached

    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": api_key, "client_secret": secret_key}
    try:
        r = requests.get(url, params=params, timeout=timeout)
    except Exception as e:
        raise AsrError("请求百度 access_token 失败（网络异常）。", provider="baidu", details=str(e)) from e

    if r.status_code != 200:
        raise AsrError(
            f"请求百度 access_token 失败（HTTP {r.status_code}）。",
            provider="baidu",
            details=r.text[:500],
        )
    try:
        j = r.json()
    except Exception as e:
        raise AsrError("百度 access_token 响应不是合法 JSON。", provider="baidu", details=str(e)) from e

    token = j.get("access_token")
    if not token:
        raise AsrError("百度 access_token 获取失败（缺少 access_token）。", provider="baidu", details=j)

    expires_in = float(j.get("expires_in") or 0.0)
    _TOKEN_CACHE["token"] = str(token)
    _TOKEN_CACHE["expires_at"] = now + max(0.0, expires_in)
    return str(token)


def _baidu_asr_wav_bytes_to_text(
    *,
    wav_bytes: bytes,
    wav_len: int,
    api_key: str,
    secret_key: str,
    dev_pid: int,
    timeout: float,
) -> str:
    if not api_key or not secret_key:
        raise AsrError("百度 ASR 未配置 api_key/secret_key。", provider="baidu")

    token = _baidu_get_access_token(api_key=api_key, secret_key=secret_key, timeout=timeout)

    # 音频字段要求 base64
    speech_b64 = base64.b64encode(wav_bytes).decode("ascii")
    payload = {
        "format": "wav",
        "rate": 16000,
        "channel": 1,
        "cuid": uuid.uuid4().hex,
        "token": token,
        "speech": speech_b64,
        "len": wav_len,
        "dev_pid": int(dev_pid),
    }

    url = "https://vop.baidu.com/server_api"
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=timeout)
    except Exception as e:
        raise AsrError("请求百度语音识别失败（网络异常）。", provider="baidu", details=str(e)) from e

    if r.status_code != 200:
        raise AsrError(
            f"请求百度语音识别失败（HTTP {r.status_code}）。",
            provider="baidu",
            details=r.text[:800],
        )

    try:
        j = r.json()
    except Exception as e:
        raise AsrError("百度语音识别响应不是合法 JSON。", provider="baidu", details=str(e)) from e

    # 典型字段：err_no / err_msg / result(数组)
    err_no = j.get("err_no", None)
    if err_no not in (0, "0", None):
        raise AsrError(
            f"百度语音识别业务错误：{j.get('err_no')} {j.get('err_msg','')}".strip(),
            provider="baidu",
            details=j,
        )

    result = j.get("result") or []
    texts: list[str] = []
    if isinstance(result, list):
        for it in result:
            if isinstance(it, str) and it.strip():
                texts.append(it.strip())
    if not texts:
        # 某些场景可能 result 结构不同，兜底返回空
        # 不在此打印 token 等敏感内容
        return ""
    # 拼接 chunk 内结果
    return "".join(texts).strip()


def audio_to_text(
    audio: str | Path | bytes | BinaryIO,
    *,
    input_suffix_hint: str | None = None,
    config_path: str | Path | None = None,
    keep_temp: bool = False,
) -> str:
    """
    将音频（MP3/WAV 等）转为中文文本：
    1) 标准化到 16k/单声道 wav
    2) 按 chunk_seconds 分段
    3) 逐段调用百度 VOP ASR
    4) 拼接文本并返回
    """
    try:
        settings = _load_settings(config_path=config_path)
    except FileNotFoundError as e:
        raise AsrError("请求百度 ASR 失败（配置文件缺失）", provider="baidu") from e
    # if settings.provider != "baidu":
    #     raise AsrError(f"当前仅实现 baidu ASR，实际 provider={settings.provider}")

    # 输入后缀提示：仅用于 ffmpeg 临时文件保存，主要影响文件识别
    if input_suffix_hint is None:
        # 若 audio 是路径，自动推断
        if isinstance(audio, (str, Path)):
            suf = Path(str(audio)).suffix.lower()
            input_suffix_hint = suf if suf in (".mp3", ".wav", ".m4a", ".amr", ".aac") else ".wav"
        else:
            input_suffix_hint = ".wav"

    normalized_wav_path = normalize_audio_to_wav_16k_mono(
        audio,
        input_suffix_hint=input_suffix_hint,
        config_path=config_path,
    )

    chunk_paths: list[str] = []
    chunk_parent_dirs: set[Path] = set()
    try:
        chunk_paths = segment_wav(normalized_wav_path, chunk_seconds=settings.chunk_seconds)
        chunk_parent_dirs = {Path(p).parent for p in chunk_paths if p}

        seg_texts: list[str] = []
        for cp in chunk_paths:
            # 正常模式：读取文件并进行 ASR
            wav_bytes = Path(cp).read_bytes()
            seg_text = _baidu_asr_wav_bytes_to_text(
                wav_bytes=wav_bytes,
                wav_len=len(wav_bytes),
                api_key=str(settings.baidu_api_key or ""),
                secret_key=str(settings.baidu_secret_key or ""),
                dev_pid=settings.dev_pid,
                timeout=settings.timeout,
            )
            # 允许某些段返回空字符串
            if seg_text:
                seg_texts.append(seg_text)

        # 拼接所有 chunk 的文本
        # 这里用空字符串拼接，最大化避免“句子间断裂”。
        transcript = "".join(seg_texts).strip()
        return transcript
    finally:
        # 清理临时文件（chunk 可能在 segment_wav 内部创建到 final_dir）
        if not keep_temp:
            try:
                if os.path.exists(normalized_wav_path):
                    os.remove(normalized_wav_path)
            except Exception:
                pass
            for cp in chunk_paths:
                try:
                    if os.path.exists(cp):
                        os.remove(cp)
                except Exception:
                    pass
            # 删除 chunk 所属的空目录，避免临时目录残留
            for d in chunk_parent_dirs:
                try:
                    if d.exists() and d.is_dir() and not any(d.iterdir()):
                        shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass


__all__ = ["audio_to_text", "normalize_audio_to_wav_16k_mono", "segment_wav", "AsrError"]