import streamlit as st
import os
import shutil
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import EMOTION_STYLES
from src.utils import (
    split_chinese_sentences,
    format_batch_results,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_md,
    extract_text_from_txt,
    extract_text_from_image,
    audio_to_text,
)


MAX_FILE_MB = 20
MAX_SENTENCES_DEFAULT = 150


def _file_is_image(name: str) -> bool:
    name = name.lower()
    return any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"])


def _file_is_audio(name: str) -> bool:
    name = name.lower()
    return any(name.endswith(ext) for ext in [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"])


def _file_is_video(name: str) -> bool:
    name = name.lower()
    return any(name.endswith(ext) for ext in [".mp4", ".webm", ".mov", ".mkv", ".avi"])


def _file_is_text_doc(name: str) -> bool:
    name = name.lower()
    return any(name.endswith(ext) for ext in [".pdf", ".txt", ".docx", ".md"])


def _get_size_bytes(st_uploaded_file) -> int:
    # Streamlit UploadedFile: .size 可能存在也可能不存在（不同版本）
    size = getattr(st_uploaded_file, "size", None)
    if size is not None:
        return int(size)
    # 没有 size 就保守处理：尽量不要崩
    return 0


def _preview_uploaded_file(f) -> None:
    """只负责预览（截图要求：图片直渲染；音频/视频用 Streamlit 播放器）。"""
    name = getattr(f, "name", "uploaded")
    if _file_is_image(name):
        st.image(f, use_container_width=True, caption=name)
        return
    if _file_is_audio(name):
        try:
            if hasattr(f, "seek"):
                f.seek(0)
            data = f.read()
        except Exception:
            data = f
        st.audio(data, format=_guess_audio_format(name), start_time=0)
        st.caption(f"{name}")
        return
    if _file_is_video(name):
        try:
            if hasattr(f, "seek"):
                f.seek(0)
            data = f.read()
        except Exception:
            data = f

        # 显式指定 MIME 类型，提高兼容性（部分浏览器/编解码下不显式会失败）
        st.video(data, format=_guess_video_format(name))
        st.caption(f"{name}")
        return
    if _file_is_text_doc(name):
        st.info(f"已上传文档：{name}（将提取文本用于情感分析）")
        return

    st.warning(f"暂不支持预览该文件类型：{name}")


def _guess_audio_format(name: str) -> str:
    # Streamlit audio 的 format 仅做提示；如果猜不到也不影响渲染
    name = name.lower()
    for ext in [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"]:
        if name.endswith(ext):
            return ext.replace(".", "")
    return "audio"


def _guess_video_format(name: str) -> str:
    name = (name or "").lower()
    for ext, mime in [
        (".mp4", "video/mp4"),
        (".webm", "video/webm"),
        (".mov", "video/quicktime"),
        (".mkv", "video/x-matroska"),
        (".avi", "video/x-msvideo"),
    ]:
        if name.endswith(ext):
            return mime
    return "video/mp4"


def _extract_text_from_uploaded_docs(files: list) -> str:
    """把可解析的文档抽成文本（供后续情感分析）。"""
    extracted = []
    for f in files:
        name = getattr(f, "name", "").lower()
        # 注意：UploadedFile 在不同解析函数里会被读取指针位置影响，因此每个函数内部
        # 都尽量使用 seek(0) 或写入临时文件的方式处理（你们已有的工具函数已覆盖大多数情况）。
        if name.endswith(".pdf"):
            try:
                # extract_text_from_pdf 支持文件对象/路径；这里传对象让它走 seek 路径
                text = extract_text_from_pdf(f)
                if text:
                    extracted.append(text)
            except Exception as e:
                st.warning(f"PDF 文本提取失败：{name}，错误：{e}")
        elif name.endswith(".txt"):
            try:
                text = extract_text_from_txt(f)
                if text:
                    extracted.append(text)
            except Exception as e:
                st.warning(f"TXT 文本提取失败：{name}，错误：{e}")
        elif name.endswith(".docx"):
            try:
                text = extract_text_from_docx(f)
                if text:
                    extracted.append(text)
            except Exception as e:
                st.warning(f"DOCX 文本提取失败：{name}，错误：{e}")
        elif name.endswith(".md"):
            try:
                text = extract_text_from_md(f)
                if text:
                    extracted.append(text)
            except Exception as e:
                st.warning(f"Markdown 文本提取失败：{name}，错误：{e}")
        elif name.endswith(".pdf") is False and name.endswith(".txt") is False:
            # 兜底：直接不处理
            pass
        else:
            # 其它情况略过
            pass

    return "\n\n".join([t for t in extracted if t.strip()])


def _extract_text_from_uploaded_media(files: list) -> str:
    """
    把上传的“图片/音频/视频”自动抽成文本（供后续情感分析）。

    - 图片：OCR
    - 音频：ASR（长音频会在后端内部切片）
    - 视频：提取音轨 -> ASR
    """
    extracted: list[str] = []
    for f in files:
        name = getattr(f, "name", "").lower()
        if not name:
            continue

        try:
            if _file_is_image(name):
                text = extract_text_from_image(f, preprocess=True, lang="CHN_ENG")
                text = (text or "").strip()
                if text:
                    extracted.append(text)
                continue

            if _file_is_audio(name):
                ext = os.path.splitext(name)[1].lower()
                text = audio_to_text(f, input_suffix_hint=ext if ext else None)
                text = (text or "").strip()
                if text:
                    extracted.append(text)
                continue

            if _file_is_video(name):
                ext = os.path.splitext(name)[1].lower() or ".mp4"

                # Streamlit UploadedFile 是内存/临时流：视频转写需要落盘路径
                if hasattr(f, "seek"):
                    f.seek(0)
                data = f.read()
                if isinstance(data, str):
                    data = data.encode("utf-8")

                tmp_dir = tempfile.mkdtemp(prefix="multi_video_upload_")
                video_path = os.path.join(tmp_dir, f"input{ext}")
                try:
                    with open(video_path, "wb") as fp:
                        fp.write(data)

                    # 这里按需导入，避免 moviepy/ffmpeg 未安装导致页面整体崩溃
                    try:
                        from src.utils.video_processor import video_to_transcript
                    except Exception as e:
                        raise RuntimeError(
                            "视频转写依赖缺失：请安装 `moviepy` 并确保系统可用 `ffmpeg`。"
                        ) from e

                    text = video_to_transcript(video_path, cleanup_audio=True)
                    text = (text or "").strip()
                    if text:
                        extracted.append(text)
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                continue

        except Exception as e:
            st.warning(f"自动文本提取失败：{getattr(f, 'name', 'unknown')}，错误：{e}")

    return "\n\n".join([t for t in extracted if t.strip()])


def _predict_and_visualize(handler, text: str, max_sentences: int):
    """(旧逻辑) 已保留但不再用于主流程。"""
    sentences = split_chinese_sentences(text)
    sentences = [s for s in sentences if s and s.strip()]
    if not sentences:
        st.warning("未检测到可分析的中文句子。请检查输入文本。")
        return

    sentences = sentences[:max_sentences]

    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()

    results = []
    total = len(sentences)
    for idx, s in enumerate(sentences):
        r = handler.predict(s)
        results.append(
            {
                "sentence": s,
                "fine_emotion": r["fine_name"],
                "coarse_emotion": r["coarse"],
                "confidence": r["confidence"],
                **{f"prob_{k}": v for k, v in r["all_probabilities"].items()},
                "all_probabilities": r["all_probabilities"],
                "top3": r["top3"],
            }
        )
        progress_bar.progress((idx + 1) / total)
        status_placeholder.write(f"正在分析 {idx + 1}/{total}：{r['fine_name']}")

    # 表格用格式化函数（复用你们 batch 页的字段组织）
    batch_results_for_df = []
    for item in results:
        batch_results_for_df.append(
            {
                "text": item["sentence"],
                "fine_name": item["fine_emotion"],
                "coarse": item["coarse_emotion"],
                "confidence": item["confidence"],
                "all_probabilities": item["all_probabilities"],
            }
        )
    result_df = format_batch_results(batch_results_for_df)

    st.subheader("多模态情感分析结果可视化")

    # 1) 分布（细粒度饼图 + 粗粒度饼图）
    col_fine, col_coarse = st.columns([1.2, 1.0], gap="large")
    with col_fine:
        st.markdown("<div class='muted' style='margin-bottom:6px;'>细粒度情感分布（按句子统计）</div>", unsafe_allow_html=True)
        fine_counts = result_df["fine_emotion"].value_counts()
        fig_pie = px.pie(
            values=fine_counts.values,
            names=fine_counts.index,
            title="",
            color=fine_counts.index,
            color_discrete_map={EMOTION_STYLES[i]["name"]: EMOTION_STYLES[i]["color"] for i in range(6)},
        )
        fig_pie.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_coarse:
        st.markdown("<div class='muted' style='margin-bottom:6px;'>粗粒度情感分布（按句子统计）</div>", unsafe_allow_html=True)
        coarse_counts = result_df["coarse_emotion"].value_counts()
        fig_coarse = px.pie(
            values=coarse_counts.values,
            names=coarse_counts.index,
            title="",
            color=coarse_counts.index,
            color_discrete_map={"正面": "#2ecc71", "负面": "#e74c3c", "中性": "#95a5a6"},
        )
        fig_coarse.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_coarse, use_container_width=True)

    # 2) 雷达图：平均概率
    categories = list(results[0]["all_probabilities"].keys())
    avg_values = []
    for k in categories:
        avg_values.append(sum([it["all_probabilities"].get(k, 0.0) for it in results]) / len(results))

    col_radar, col_bar = st.columns([1, 1], gap="large")
    with col_radar:
        st.markdown("<div class='muted' style='margin-bottom:6px;'>平均情感概率（雷达图）</div>", unsafe_allow_html=True)
        fig_radar = go.Figure()
        # 复用 single 页的雷达图结构：顺序闭合
        # categories 是中文情感名列表，EMOTION_STYLES 的 name 与其一致
        # 找一个颜色：取开心(或其它)颜色作为统一主色
        accent = "#ef4444"
        fig_radar.add_trace(
            go.Scatterpolar(
                r=avg_values + [avg_values[0]],
                theta=categories + [categories[0]],
                mode="lines+markers",
                name="平均情感概率",
                line_color=accent,
                line_width=3,
                marker=dict(size=6, color=accent),
            )
        )
        fig_radar.update_layout(
            template="simple_white",
            polar=dict(
                bgcolor="#ffffff",
                radialaxis=dict(visible=True, showgrid=True, range=[0, 1], dtick=0.2, gridcolor="#d1d5db", gridwidth=1.2),
                angularaxis=dict(showgrid=True, gridcolor="#e5e7eb", gridwidth=1.0),
            ),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(size=13, color="#111827"),
            showlegend=False,
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_bar:
        st.markdown("<div class='muted' style='margin-bottom:6px;'>平均情感概率（柱状图）</div>", unsafe_allow_html=True)
        # 颜色按类别映射
        color_map = {EMOTION_STYLES[i]["name"]: EMOTION_STYLES[i]["color"] for i in range(6)}
        fig_bar = px.bar(
            x=categories,
            y=avg_values,
            color=categories,
            color_discrete_map=color_map,
            labels={"x": "情感类别", "y": "概率"},
        )
        fig_bar.update_layout(template="simple_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", showlegend=False, height=360,
                               margin=dict(l=20, r=20, t=30, b=20))
        fig_bar.update_traces(marker_line_width=0)
        st.plotly_chart(fig_bar, use_container_width=True)

    # 3) 结果表 + 下载
    st.markdown("<div class='card-title' style='margin-top:18px;'>详细结果表（前 N 句）</div>", unsafe_allow_html=True)
    html_table = result_df.to_html(classes="result-table", index=False, border=0, justify="left", escape=False)
    st.markdown(html_table, unsafe_allow_html=True)

    csv = result_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="下载结果",
        data=csv,
        file_name="multimodal_emotion_results.csv",
        mime="text/csv",
        key="download_multimodal_result",
        type="primary",
    )


def render_page(handler):
    st.markdown("<div class='section-title'>🖼️ 多模态分析</div>", unsafe_allow_html=True)

    if "multimodal_result_df" not in st.session_state:
        st.session_state.multimodal_result_df = None

    if st.session_state.multimodal_result_df is None:
        _render_upload_interface(handler)
    else:
        _render_result_interface()


def _run_multimodal_analysis(handler, combined_text: str, max_sentences: int) -> pd.DataFrame:
    """把输入文本拆句 -> 逐句预测 -> 输出结果 DataFrame（不做可视化）。"""
    sentences = split_chinese_sentences(combined_text)
    sentences = [s for s in sentences if s and s.strip()]
    if not sentences:
        st.warning("未检测到可分析的中文句子。请检查输入文本。")
        return pd.DataFrame()

    sentences = sentences[:max_sentences]

    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()

    results = []
    total = len(sentences)
    for idx, s in enumerate(sentences):
        r = handler.predict(s)
        results.append(
            {
                "text": s,
                "fine_name": r["fine_name"],
                "coarse": r["coarse"],
                "confidence": r["confidence"],
                "all_probabilities": r["all_probabilities"],
            }
        )
        progress_bar.progress((idx + 1) / total)
        status_placeholder.write(f"正在分析 {idx + 1}/{total}：{r['fine_name']}")

    return format_batch_results(results)


def _render_upload_interface(handler):
    # 外层整页卡片框线：复用批量页的 CSS 选择器
    with st.container(border=True):
        st.markdown('<div class="batch-uploader-wrap">', unsafe_allow_html=True)

        st.markdown(
            """
            <div class='muted' style='margin-bottom:10px;'>
            支持多格式文件上传并预览：图片直接渲染，音频/视频使用播放器；文档（PDF/TXT/DOCX/MD）可直接提取文本。
            图片/音频/视频也会自动走 OCR/ASR/转写生成文本用于情感分析。
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "上传多媒体文件（多模态支持：图片/音频/视频/文档）",
            type=["png", "jpg", "jpeg", "webp", "mp3", "wav", "m4a", "mp4", "webm", "pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        # 文件大小限制（截图要求：建议 <20MB）
        valid_files = []
        if uploaded_files:
            for f in uploaded_files:
                size_bytes = _get_size_bytes(f)
                if size_bytes and size_bytes > MAX_FILE_MB * 1024 * 1024:
                    st.error(f"文件过大（>{MAX_FILE_MB}MB）：{getattr(f, 'name', 'unknown')}，请换小的文件。")
                    continue
                valid_files.append(f)

        if valid_files:
            st.markdown("<div class='batch-block-title'>文件预览</div>", unsafe_allow_html=True)
            preview_cols = st.columns([1, 1], gap="large")

            # 预览布局：最多展示前 4 个文件，避免页面太长
            max_preview = min(len(valid_files), 4)
            for i in range(max_preview):
                c = preview_cols[i % 2]
                with c:
                    _preview_uploaded_file(valid_files[i])

        st.markdown("<div class='batch-block-title'>用于情感分析的文本</div>", unsafe_allow_html=True)

        auto_text = ""
        if valid_files:
            auto_text = _extract_text_from_uploaded_docs(valid_files)

        override_text = st.text_area(
            "（可选）自动填充文档提取文本；如需修改可在此处覆盖。",
            value=auto_text,
            height=200,
            label_visibility="collapsed",
        )

        max_sentences = st.slider(
            "最多分析句子数（性能保护）",
            min_value=20,
            max_value=400,
            value=MAX_SENTENCES_DEFAULT,
            step=10,
        )

        action_left, action_right = st.columns([1, 2.3], gap="medium")
        with action_left:
            run_btn = st.button("开始多模态分析", type="primary", use_container_width=True)

        with action_right:
            if run_btn:
                combined_text = (override_text or "").strip()

                with st.spinner("正在提取多模态文本并拆句进行情感分析..."):
                    # 额外为图片/音频/视频自动抽取文本（不重复处理文档）
                    media_text = _extract_text_from_uploaded_media(valid_files)
                    if media_text.strip():
                        combined_text = f"{combined_text}\n\n{media_text}".strip() if combined_text else media_text.strip()

                    if not combined_text:
                        st.warning("未提取到可分析文本。请上传文档，或上传可识别的图片/音频/视频。")
                        st.stop()

                    result_df = _run_multimodal_analysis(handler, combined_text, max_sentences=max_sentences)
                    if result_df is None or result_df.empty:
                        st.warning("分析结果为空，请检查输入文本。")
                        st.stop()
                    st.session_state.multimodal_result_df = result_df
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def _render_result_interface():
    # 结果页也保持外框线效果
    result_df = st.session_state.multimodal_result_df
    if result_df is None or result_df.empty:
        st.warning("当前没有可展示的分析结果。")
        return

    with st.container(border=True):
        st.markdown('<div class="batch-uploader-wrap">', unsafe_allow_html=True)

        top_row_left, top_row_right = st.columns([1, 0.15], gap="small")
        with top_row_left:
            st.markdown('<div class="card-title">结果展示</div>', unsafe_allow_html=True)
        with top_row_right:
            if st.button("返回上传界面", type="primary", use_container_width=True, key="back_to_upload_multi"):
                st.session_state.multimodal_result_df = None
                st.rerun()

        # 上半部分：左右两个饼图
        col_fine, col_coarse = st.columns([1.25, 1.0], gap="large")
        with col_fine:
            st.markdown("<div class='muted' style='margin-bottom:6px;'>细粒度情感分布</div>", unsafe_allow_html=True)
            fine_counts = result_df["fine_emotion"].value_counts()
            fig_pie = px.pie(
                values=fine_counts.values,
                names=fine_counts.index,
                title="",
                color=fine_counts.index,
                color_discrete_map={EMOTION_STYLES[i]["name"]: EMOTION_STYLES[i]["color"] for i in range(6)},
            )
            fig_pie.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_coarse:
            st.markdown("<div class='muted' style='margin-bottom:6px;'>粗粒度情感分布</div>", unsafe_allow_html=True)
            coarse_counts = result_df["coarse_emotion"].value_counts()
            fig_coarse = px.pie(
                values=coarse_counts.values,
                names=coarse_counts.index,
                title="",
                color=coarse_counts.index,
                color_discrete_map={"正面": "#2ecc71", "负面": "#e74c3c", "中性": "#95a5a6"},
            )
            fig_coarse.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_coarse, use_container_width=True)

        # 下半部分：平均概率雷达图 + 柱状图
        categories = [EMOTION_STYLES[i]["name"] for i in range(6)]
        avg_values = []
        for c in categories:
            col = f"prob_{c}"
            if col in result_df.columns:
                avg_values.append(float(result_df[col].mean()))
            else:
                avg_values.append(0.0)

        col_radar, col_bar = st.columns([1, 1], gap="large")
        with col_radar:
            st.markdown("<div class='muted' style='margin-bottom:6px;'>平均情感概率（雷达图）</div>", unsafe_allow_html=True)
            fig_radar = go.Figure()
            accent = "#ef4444"
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=avg_values + [avg_values[0]],
                    theta=categories + [categories[0]],
                    mode="lines+markers",
                    name="平均情感概率",
                    line_color=accent,
                    line_width=3,
                    marker=dict(size=6, color=accent),
                )
            )
            fig_radar.update_layout(
                template="simple_white",
                polar=dict(
                    bgcolor="#ffffff",
                    radialaxis=dict(visible=True, showgrid=True, range=[0, 1], dtick=0.2, gridcolor="#d1d5db", gridwidth=1.2),
                    angularaxis=dict(showgrid=True, gridcolor="#e5e7eb", gridwidth=1.0),
                ),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(size=13, color="#111827"),
                showlegend=False,
                height=360,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_bar:
            st.markdown("<div class='muted' style='margin-bottom:6px;'>平均情感概率（柱状图）</div>", unsafe_allow_html=True)
            color_map = {EMOTION_STYLES[i]["name"]: EMOTION_STYLES[i]["color"] for i in range(6)}
            fig_bar = px.bar(
                x=categories,
                y=avg_values,
                color=categories,
                color_discrete_map=color_map,
                labels={"x": "情感类别", "y": "概率"},
            )
            fig_bar.update_layout(
                template="simple_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                showlegend=False,
                height=360,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            fig_bar.update_traces(marker_line_width=0)
            st.plotly_chart(fig_bar, use_container_width=True)

        # 详细结果表 + 下载
        st.markdown("<div class='card-title' style='margin-top:18px;'>详细结果</div>", unsafe_allow_html=True)
        html_table = result_df.to_html(classes="result-table", index=False, border=0, justify="left", escape=False)
        st.markdown(html_table, unsafe_allow_html=True)

        csv = result_df.to_csv(index=False, encoding="utf-8-sig")
        st.markdown('<div class="download-btn-wrap" style="text-align:right;margin-top:12px;">', unsafe_allow_html=True)
        st.download_button(
            label="下载结果",
            data=csv,
            file_name="multimodal_emotion_results.csv",
            mime="text/csv",
            key="download_multimodal_result",
            type="primary",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

