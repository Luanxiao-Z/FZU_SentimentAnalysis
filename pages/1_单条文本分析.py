# ==========================================
# 单条文本分析页面
# ==========================================

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.config import EMOTION_STYLES, EMOTION_NAMES
from src.utils import get_example_texts, emotion_id_by_name, get_coarse_badge_class


def render_page(handler):
    """渲染单条文本分析页面"""
    st.markdown('<div class="section-title">📝 单条文本分析</div>', unsafe_allow_html=True)

    col_input, col_result = st.columns([2.1, 1.4], gap="large")

    with col_input:
        user_input, analyze_btn = render_text_input_card()

    with col_result:
        with st.container(border=True):
            st.markdown('<div class="result-card-marker" style="display:none;"></div>', unsafe_allow_html=True)

            if analyze_btn:
                if not user_input.strip():
                    st.warning("请输入文本后再分析。")
                else:
                    with st.spinner("正在分析情感..."):
                        result = handler.predict(user_input.strip())
                    st.session_state.last_result = result
                    st.session_state.show_charts = True

            # 标题行 + 右侧粗粒度结果
            if "last_result" in st.session_state:
                _coarse = st.session_state.last_result.get("coarse", "中性")
            else:
                _coarse = "中性"
            _coarse_cls = get_coarse_badge_class(_coarse)
            st.markdown(
                f"""
                <div class="result-title-row">
                    <div class="card-title">🧾 分析结果</div>
                    <details class="coarse-details">
                        <summary class="coarse-pill {_coarse_cls}">{_coarse}</summary>
                        <div class="coarse-menu">
                            <div class="coarse-option positive">正面</div>
                            <div class="coarse-option negative">负面</div>
                            <div class="coarse-option neutral">中性</div>
                        </div>
                    </details>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if "last_result" not in st.session_state:
                # 初始占位
                st.markdown(
                    """
                    <div class="result-head">
                        <div class="emotion-main">🙂 待分析</div>
                        <div class="badge-stack">
                            <div class="mini-badge">置信度 --</div>
                        </div>
                    </div>
                    <div class="muted" style="margin-top: 10px; font-weight: 900;">可能的情感</div>
                    <div class="top3-row">
                        <div class="top3-item">
                            <div class="left">🙂 情感 1</div>
                            <div class="right">--</div>
                        </div>
                        <div class="top3-item">
                            <div class="left">🙂 情感 2</div>
                            <div class="right">--</div>
                        </div>
                        <div class="top3-item">
                            <div class="left">🙂 情感 3</div>
                            <div class="right">--</div>
                        </div>
                    </div>
                    <div class="muted" style="margin-top: 8px;">在左侧输入文本并点击「开始分析」，这里会展示真实的情感结果与 Top-3 概率。</div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                result = st.session_state.last_result
                emotion_id = int(result["fine_id"])
                style = EMOTION_STYLES[emotion_id]

                st.markdown(
                    f"""
                    <div class="result-head">
                        <div class="emotion-main">{style['emoji']} {result['fine_name']}</div>
                        <div class="badge-stack">
                            <div class="mini-badge">置信度 {result['confidence']:.2f}</div>
                        </div>
                    </div>
                    <div class="muted" style="margin-top: 10px; font-weight: 900;">可能的情感</div>
                    <div class="top3-row">
                        {''.join([
                            f'''<div class="top3-item">
                                    <div class="left">{EMOTION_STYLES[emotion_id_by_name(em)]['emoji']} {em}</div>
                                    <div class="right">{prob*100:.1f}%</div>
                                </div>'''
                            for em, prob in result["top3"]
                        ])}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # 图表区
    if st.session_state.get("show_charts", True):
        st.markdown('<div class="section-title">📊 情感概率分布</div>', unsafe_allow_html=True)

        if "last_result" in st.session_state:
            result = st.session_state.last_result
            emotion_id = int(result["fine_id"])
            _style = EMOTION_STYLES[emotion_id]
            categories = list(result["all_probabilities"].keys())
            values = list(result["all_probabilities"].values())
        else:
            _style = {"color": "#ef4444"}
            categories = [EMOTION_STYLES[i]["name"] for i in range(6)]
            values = [0.0] * len(categories)

        with st.container(border=True):
            if "last_result" not in st.session_state:
                st.markdown(
                    "<div class='muted' style='margin-bottom:10px;'>先在上方输入文本并点击「开始分析」，这里将展示各情感概率分布图。</div>",
                    unsafe_allow_html=True,
                )

            chart_left, chart_right = st.columns([1, 1], gap="large")
            with chart_left:
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    mode="lines+markers",
                    name="情感概率",
                    line_color=_style["color"],
                    line_width=3,
                    marker=dict(size=6, color=_style["color"]),
                ))
                fig_radar.update_layout(
                    template="simple_white",
                    polar=dict(
                        bgcolor="#ffffff",
                        radialaxis=dict(
                            visible=True,
                            showgrid=True,
                            range=[0, 1],
                            tick0=0.0,
                            dtick=0.2,
                            gridcolor="#d1d5db",
                            gridwidth=1.2,
                            linecolor="#9ca3af",
                            tickfont=dict(size=12, color="#4b5563"),
                        ),
                        angularaxis=dict(
                            showgrid=True,
                            gridcolor="#e5e7eb",
                            gridwidth=1.0,
                            linecolor="#9ca3af",
                            tickfont=dict(size=12, color="#4b5563"),
                        ),
                    ),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(size=13, color="#111827"),
                    showlegend=False,
                    height=360,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            with chart_right:
                fig_bar = px.bar(
                    x=categories,
                    y=values,
                    color=categories,
                    color_discrete_map={EMOTION_STYLES[i]["name"]: EMOTION_STYLES[i]["color"] for i in range(6)},
                    labels={"x": "情感类别", "y": "概率"},
                )
                fig_bar.update_layout(
                    template="simple_white",
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    xaxis=dict(
                        showgrid=False,
                        zerolinecolor="#9ca3af",
                        tickfont=dict(size=12, color="#374151"),
                        title_font=dict(size=13, color="#111827"),
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="#d1d5db",
                        gridwidth=1.2,
                        zerolinecolor="#9ca3af",
                        tickfont=dict(size=12, color="#374151"),
                        title_font=dict(size=13, color="#111827"),
                        range=[0, 1],
                    ),
                    font=dict(size=13, color="#111827"),
                    showlegend=False,
                    height=360,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig_bar, use_container_width=True)


def render_text_input_card() -> tuple:
    """渲染文本输入卡片"""
    if "pending_example_text" in st.session_state:
        st.session_state.pop("pending_example_text", None)

    with st.container(border=True):
        st.markdown('<div class="input-card-marker"></div>', unsafe_allow_html=True)

        user_input = st.text_area(
            "请输入要分析的文本",
            key="single_text",
            height=170,
            placeholder="请输入要分析的文本…",
            label_visibility="collapsed",
        )

        examples = get_example_texts()
        example_display = ["（不使用示例）"] + [f"{label}：{text}" for label, text in examples]

        def _apply_example_to_text() -> None:
            choice = st.session_state.get("example_choice", "（不使用示例）")
            if choice == "（不使用示例）":
                return
            for label, text in examples:
                if choice.startswith(f"{label}："):
                    st.session_state["single_text"] = text
                    return

        action_left, action_right = st.columns([3.2, 1.2], gap="small")
        with action_left:
            st.selectbox(
                "示例文本列表",
                options=example_display,
                key="example_choice",
                label_visibility="collapsed",
                on_change=_apply_example_to_text,
                help="点击示例后会自动填充到上方输入框",
            )
        with action_right:
            analyze_btn = st.button("开始分析", type="primary", use_container_width=True)

    return user_input, analyze_btn