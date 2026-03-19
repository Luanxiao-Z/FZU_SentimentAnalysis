from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import requests

try:
    from PIL import Image, ImageOps
except Exception as e:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = e
else:
    _PIL_IMPORT_ERROR = None


class OcrError(RuntimeError):
    """OCR 处理失败。"""

    def __init__(self, message: str, *, provider: str | None = None, details: Any | None = None):
        super().__init__(message)
        self.provider = provider
        self.details = details


@dataclass(frozen=True)
class OcrSettings:
    provider: str = "baidu"
    timeout: float = 12.0
    max_side: int = 1600
    jpeg_quality: int = 85

    baidu_api_key: str | None = None
    baidu_secret_key: str | None = None


_TOKEN_CACHE: dict[str, Any] = {"token": None, "expires_at": 0.0}


def extract_text_from_image(
    image: str | Path | bytes | BinaryIO,
    *,
    provider: str | None = None,
    preprocess: bool = True,
    lang: str = "CHN_ENG",
    timeout: float | None = None,
    config_path: str | Path | None = None,
) -> str:
    """
    通用 OCR：支持 JPG/PNG 输入，返回拼接后的文本字符串。

    Args:
        image: 图片路径 / bytes / 文件对象
        provider: 'baidu' 或 'aliyun'（默认从配置读取）
        preprocess: 是否进行图片预处理（压缩、转换、纠偏）
        lang: 语言类型（百度参数名 language_type），默认中英混合 CHN_ENG
        timeout: 网络超时（秒），默认从配置读取
        config_path: 私密配置文件路径（TOML）；默认使用 repo 根目录 `config/ocr_secrets.toml`
    """
    settings = _load_settings(config_path=config_path)
    use_provider = (provider or settings.provider or "baidu").strip().lower()
    req_timeout = float(timeout if timeout is not None else settings.timeout)

    raw_bytes = _read_image_bytes(image)
    if preprocess:
        img_bytes, mime = _preprocess_image_bytes(
            raw_bytes,
            max_side=settings.max_side,
            jpeg_quality=settings.jpeg_quality,
        )
    else:
        img_bytes, mime = raw_bytes, "image/*"

    if use_provider == "baidu":
        resp_json = _baidu_general_ocr(
            img_bytes,
            api_key=settings.baidu_api_key,
            secret_key=settings.baidu_secret_key,
            language_type=lang,
            timeout=req_timeout,
        )
        return _normalize_text_from_baidu(resp_json)

    if use_provider == "aliyun":
        raise OcrError("阿里云 OCR provider 尚未接入（请先使用 provider='baidu'）", provider="aliyun")

    raise OcrError(f"不支持的 OCR provider: {use_provider}", provider=use_provider)


def _default_config_path() -> Path:
    # 约定：repo_root/config/ocr_secrets.toml
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "config" / "ocr_secrets.toml"


def _load_settings(*, config_path: str | Path | None) -> OcrSettings:
    path = Path(config_path) if config_path is not None else _default_config_path()
    if not path.exists():
        raise OcrError(
            "未找到 OCR 私密配置文件。请将 `config/ocr_secrets.example.toml` 复制为 "
            "`config/ocr_secrets.toml` 并填写密钥。",
            provider=None,
            details={"expected_path": str(path)},
        )

    try:
        import tomllib  # py3.11+

        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        # py3.10 兼容：使用第三方 tomli
        try:
            import tomli  # type: ignore
        except Exception as e:
            raise OcrError(
                "当前 Python 不支持 tomllib，且未安装 tomli。请执行：pip install tomli",
                details=str(e),
            ) from e
        try:
            data = tomli.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise OcrError("读取 OCR 配置失败（TOML 格式可能有误）。", details=str(e)) from e
    except Exception as e:
        raise OcrError("读取 OCR 配置失败（TOML 格式可能有误）。", details=str(e)) from e

    ocr_cfg = (data.get("ocr") or {}) if isinstance(data, dict) else {}
    baidu_cfg = (data.get("baidu_ocr") or {}) if isinstance(data, dict) else {}

    def _get_int(d: Mapping[str, Any], k: str, default: int) -> int:
        v = d.get(k, default)
        try:
            return int(v)
        except Exception:
            return default

    def _get_float(d: Mapping[str, Any], k: str, default: float) -> float:
        v = d.get(k, default)
        try:
            return float(v)
        except Exception:
            return default

    provider = str(ocr_cfg.get("provider", "baidu")).strip().lower()
    return OcrSettings(
        provider=provider,
        timeout=_get_float(ocr_cfg, "timeout", 12.0),
        max_side=_get_int(ocr_cfg, "max_side", 1600),
        jpeg_quality=_get_int(ocr_cfg, "jpeg_quality", 85),
        baidu_api_key=str(baidu_cfg.get("api_key")).strip() if baidu_cfg.get("api_key") else None,
        baidu_secret_key=str(baidu_cfg.get("secret_key")).strip() if baidu_cfg.get("secret_key") else None,
    )


def _read_image_bytes(image: str | Path | bytes | BinaryIO) -> bytes:
    if isinstance(image, (str, Path)):
        p = Path(image)
        return p.read_bytes()
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    # file-like
    if hasattr(image, "read"):
        try:
            image.seek(0)  # type: ignore[attr-defined]
        except Exception:
            pass
        data = image.read()  # type: ignore[call-arg]
        if isinstance(data, str):
            data = data.encode("utf-8")
        return bytes(data)
    raise OcrError("不支持的图片输入类型。")


def _preprocess_image_bytes(
    raw_bytes: bytes,
    *,
    max_side: int,
    jpeg_quality: int,
) -> tuple[bytes, str]:
    if Image is None or ImageOps is None:
        raise OcrError(
            "缺少图片预处理依赖 Pillow。请先安装：pip install Pillow",
            details=str(_PIL_IMPORT_ERROR),
        )

    from io import BytesIO

    try:
        im = Image.open(BytesIO(raw_bytes))
    except Exception as e:
        raise OcrError("无法读取图片（可能不是合法的 JPG/PNG）。", details=str(e)) from e

    # 处理 EXIF 旋转（手机拍照常见）
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass

    # 转换到 RGB，并处理透明背景（PNG->JPEG 时白底更稳）
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im.convert("RGBA"))
        im = bg.convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")

    # 缩放
    w, h = im.size
    longest = max(w, h)
    if max_side > 0 and longest > max_side:
        scale = max_side / float(longest)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        im = im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    # 统一输出 JPEG（体积更可控、OCR 通常更稳）
    out = BytesIO()
    q = int(max(30, min(95, jpeg_quality)))
    im.save(out, format="JPEG", quality=q, optimize=True)
    return out.getvalue(), "image/jpeg"


def _baidu_get_access_token(*, api_key: str, secret_key: str, timeout: float) -> str:
    now = time.time()
    cached = _TOKEN_CACHE.get("token")
    expires_at = float(_TOKEN_CACHE.get("expires_at") or 0.0)
    if isinstance(cached, str) and cached and now < (expires_at - 30):
        return cached

    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key,
    }
    try:
        r = requests.get(url, params=params, timeout=timeout)
    except Exception as e:
        raise OcrError("请求百度 access_token 失败（网络异常）。", provider="baidu", details=str(e)) from e

    if r.status_code != 200:
        raise OcrError(
            f"请求百度 access_token 失败（HTTP {r.status_code}）。",
            provider="baidu",
            details=r.text[:500],
        )
    try:
        j = r.json()
    except Exception as e:
        raise OcrError("百度 access_token 响应不是合法 JSON。", provider="baidu", details=str(e)) from e

    token = j.get("access_token")
    if not token:
        raise OcrError("百度 access_token 获取失败（缺少 access_token）。", provider="baidu", details=j)
    expires_in = float(j.get("expires_in") or 0.0)
    _TOKEN_CACHE["token"] = token
    _TOKEN_CACHE["expires_at"] = now + max(0.0, expires_in)
    return str(token)


def _baidu_general_ocr(
    image_bytes: bytes,
    *,
    api_key: str | None,
    secret_key: str | None,
    language_type: str,
    timeout: float,
) -> dict[str, Any]:
    if not api_key or not secret_key:
        raise OcrError(
            "百度 OCR 未配置 api_key/secret_key。请在 `config/ocr_secrets.toml` 的 [baidu_ocr] 中填写。",
            provider="baidu",
        )

    token = _baidu_get_access_token(api_key=api_key, secret_key=secret_key, timeout=timeout)
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
    params = {"access_token": token}

    # 百度要求 image 字段为 base64
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    data = {
        "image": image_b64,
        "language_type": language_type,
        "detect_direction": "true",
        "paragraph": "false",
    }

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        r = requests.post(url, params=params, data=data, headers=headers, timeout=timeout)
    except Exception as e:
        raise OcrError("请求百度 OCR 失败（网络异常）。", provider="baidu", details=str(e)) from e

    if r.status_code != 200:
        raise OcrError(
            f"百度 OCR 请求失败（HTTP {r.status_code}）。",
            provider="baidu",
            details=r.text[:800],
        )
    try:
        j = r.json()
    except Exception as e:
        raise OcrError("百度 OCR 响应不是合法 JSON。", provider="baidu", details=str(e)) from e

    if "error_code" in j:
        raise OcrError(
            f"百度 OCR 业务错误：{j.get('error_code')} {j.get('error_msg','')}".strip(),
            provider="baidu",
            details=j,
        )
    return j


_INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def _normalize_text_from_baidu(resp_json: Mapping[str, Any]) -> str:
    items = resp_json.get("words_result") or []
    lines: list[str] = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                w = it.get("words")
                if isinstance(w, str) and w.strip():
                    lines.append(w.strip())

    text = "\n".join(lines).strip()
    if not text:
        return ""

    text = _INVISIBLE_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()
