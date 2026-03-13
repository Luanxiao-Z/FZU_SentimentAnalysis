# ==========================================
# 批量文本分析页面
# ==========================================

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
from pathlib import Path
from src.config import EMOTION_STYLES
from src.utils import load_csv_with_encoding, validate_dataframe, format_batch_results


def render_page(handler):
    """渲染批量文本分析页面"""
    st.markdown('<div class="section-title">⛃ 批量文本分析</div>', unsafe_allow_html=True)

    # 判断是否已有批量分析结果
    result_df = st.session_state.get("batch_result_df")

    if result_df is None:
        # 第一个界面：上传 CSV + 文本预览 + 开始批量分析
        render_upload_interface(handler)
    else:
        # 第二个界面：结果展示
        render_result_interface(result_df)


def render_upload_interface(handler):
    """渲染上传界面"""
    st.markdown('<div class="batch-page-wrap">', unsafe_allow_html=True)
    st.markdown("<div class='batch-subtitle'>上传包含 text 列的 CSV 文件，预览确认后点击开始批量分析。</div>", unsafe_allow_html=True)

    # 使用可变 key 来重置上传控件
    if "batch_upload_nonce" not in st.session_state:
        st.session_state.batch_upload_nonce = 0
    upload_key = f"batch_uploader_{st.session_state.batch_upload_nonce}"

    with st.container(border=True):
        st.markdown('<div class="batch-uploader-wrap">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "上传 CSV 文件",
            type=["csv"],
            label_visibility="collapsed",
            key=upload_key,
        )

        # 上传后显示文件信息
        if uploaded_file is not None:
            st.markdown(
                f"""
                <div class="batch-uploader-overlay">
                    <div class="batch-uploader-file">
                        <div class="batch-uploader-icon">📄</div>
                        <div class="batch-uploader-name">{uploaded_file.name}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        df = None
        if uploaded_file:
            df, last_error = load_csv_with_encoding(uploaded_file)

            if df is None and last_error is not None:
                st.markdown(
                    f"""
                    <div style="
                        margin-top: 8px;
                        padding: 10px 14px;
                        border-radius: 8px;
                        background: #E9FBF3;
                        color: #256C47;
                        font-size: 0.9rem;
                        font-weight: 650;
                    ">
                        无法读取 CSV 文件，请确认编码为 UTF-8 / UTF-8-SIG / GBK / GB2312 / BIG5 / CP936 / LATIN1 等常见编码之一。
                        错误信息：{last_error}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("重新选择文件", key="reset_bad_csv"):
                    st.session_state.batch_upload_nonce += 1
                    st.rerun()

            if df is not None:
                valid, error_msg = validate_dataframe(df, ['text'])
                if not valid:
                    st.error(error_msg)
                    df = None

        # 文本预览区域
        st.markdown("<div class='batch-block-title'>文本预览</div>", unsafe_allow_html=True)
        if df is not None:
            st.markdown(f"<div class='batch-block-meta'>共读取 {len(df)} 条数据（展示前 50 条）</div>", unsafe_allow_html=True)
            preview_samples = df["text"].astype(str).head(50).tolist()
            preview_text = "\n\n".join(preview_samples)
        else:
            st.markdown("<div class='batch-block-meta'>共读取 0 条数据</div>", unsafe_allow_html=True)
            preview_text = ""

        st.text_area(
            "当前文本预览",
            value=preview_text,
            height=220,
            label_visibility="collapsed",
            disabled=True,
        )

        # 底部操作行
        action_left, action_right = st.columns([1.1, 3.2], gap="medium")
        with action_left:
            run_batch = st.button(
                "开始批量分析",
                type="primary",
                use_container_width=True,
            )
        with action_right:
            progress_bar = st.progress(0.0)
            status_placeholder = st.empty()

        if run_batch and df is None:
            st.markdown(
                """
                <div style="
                    margin-top: 8px;
                    padding: 10px 14px;
                    border-radius: 8px;
                    background: #E9FBF3;
                    color: #256C47;
                    font-size: 0.9rem;
                    font-weight: 650;
                ">
                    请先上传包含 text 列的 CSV 文件后再开始批量分析。
                </div>
                """,
                unsafe_allow_html=True,
            )

        if run_batch and df is not None:
            results = []
            total = len(df)
            for idx, row in df.iterrows():
                text = str(row["text"])
                r = handler.predict(text)
                results.append({
                    "text": text,
                    "fine_emotion": r["fine_name"],
                    "coarse_emotion": r["coarse"],
                    "confidence": r["confidence"],
                    **{f"prob_{k}": v for k, v in r["all_probabilities"].items()},
                })

                progress_bar.progress((idx + 1) / total)
                status_placeholder.write(f"正在分析 {idx + 1}/{total}：{r['fine_name']}")

            st.session_state.batch_result_df = format_batch_results(results)
            st.session_state.scroll_to_top = True
            st.success("分析完成。")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_result_interface(result_df):
    """渲染结果展示界面"""
    if st.session_state.get("scroll_to_top"):
        components.html(
            "<script>window.scrollTo({top: 0, left: 0, behavior: 'instant'});</script>",
            height=0,
        )
        st.session_state.scroll_to_top = False

    with st.container(border=True):
        top_row_left, top_row_right = st.columns([1, 0.15], gap="small")
        with top_row_left:
            st.markdown('<div class="card-title">结果展示</div>', unsafe_allow_html=True)
        with top_row_right:
            if st.button("返回上传界面", type="primary", use_container_width=True, key="back_to_upload"):
                st.session_state.batch_result_df = None
                st.rerun()

        # 上半部分：左右两个饼图
        col_fine, col_coarse = st.columns([1.25, 1.0], gap="large")
        with col_fine:
            st.markdown('<div class="muted" style="margin-bottom:6px;">细粒度情感分布</div>', unsafe_allow_html=True)
            fine_counts = result_df["fine_emotion"].value_counts()
            fig_pie = px.pie(
                values=fine_counts.values,
                names=fine_counts.index,
                title="",
                color=fine_counts.index,
                color_discrete_map={EMOTION_STYLES[i]["name"]: EMOTION_STYLES[i]["color"] for i in range(6)},
            )
            fig_pie.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#1f2d3d"),
                legend=dict(font=dict(color="#1f2d3d")),
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_coarse:
            st.markdown('<div class="muted" style="margin-bottom:6px;">粗粒度情感分布</div>', unsafe_allow_html=True)
            coarse_counts = result_df["coarse_emotion"].value_counts()
            fig_coarse = px.pie(
                values=coarse_counts.values,
                names=coarse_counts.index,
                title="",
                color=coarse_counts.index,
                color_discrete_map={"正面": "#2ecc71", "负面": "#e74c3c", "中性": "#95a5a6"},
            )
            fig_coarse.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#1f2d3d"),
                legend=dict(font=dict(color="#1f2d3d")),
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_coarse, use_container_width=True)

        # 下半部分：详细结果表 + 下载按钮
        st.markdown('<div class="card-title" style="margin-top:18px;">详细结果</div>', unsafe_allow_html=True)

        html_table = result_df.to_html(
            classes="result-table",
            index=False,
            border=0,
            justify="left",
            escape=False,
        )
        st.markdown(html_table, unsafe_allow_html=True)

        # 下载按钮
        csv = result_df.to_csv(index=False, encoding="utf-8-sig")
        st.markdown('<div class="download-btn-wrap" style="text-align:right;margin-top:12px;">', unsafe_allow_html=True)
        st.download_button(
            label="下载结果",
            data=csv,
            file_name="emotion_analysis_results.csv",
            mime="text/csv",
            key="download_batch_result",
            type="primary",
        )
        st.markdown('</div>', unsafe_allow_html=True)