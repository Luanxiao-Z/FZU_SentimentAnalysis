# ==========================================
# 细粒度情感分析界面（Streamlit）
# ==========================================

import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

# ------------------------------------------
# 页面配置
# ------------------------------------------

st.set_page_config(
    page_title="细粒度情感分析系统",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式（独立文件）
_css_path = Path(__file__).with_name("app_emotion.css")
st.markdown(f"<style>{_css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ------------------------------------------
# 加载配置和模型
# ------------------------------------------

@st.cache_resource
def load_emotion_model():
    """加载细粒度情感模型"""
    model_path = './emotion_model'

    # 加载配置
    with open(f'{model_path}/emotion_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 转换键为整数类型
    config['emotion_names'] = {int(k): v for k, v in config['emotion_names'].items()}
    config['coarse_map'] = {int(k): v for k, v in config['coarse_map'].items()}

    # 加载模型
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    return tokenizer, model, device, config


# 加载
with st.spinner('正在加载情感分析模型...'):
    tokenizer, model, device, config = load_emotion_model()

EMOTION_NAMES = config['emotion_names']
COARSE_MAP = config['coarse_map']
NUM_LABELS = config['num_labels']

# 情感颜色和图标映射
EMOTION_STYLES = {
    0: {"name": "悲伤", "emoji": "😢", "color": "#3498db", "bg": "#ebf5fb"},
    1: {"name": "开心", "emoji": "😊", "color": "#f39c12", "bg": "#fef5e7"},
    2: {"name": "生气", "emoji": "😠", "color": "#e74c3c", "bg": "#fdedec"},
    3: {"name": "惊讶", "emoji": "😲", "color": "#9b59b6", "bg": "#f5eef8"},
    4: {"name": "恐惧", "emoji": "😨", "color": "#2c3e50", "bg": "#eaeded"},
    5: {"name": "厌恶", "emoji": "🤢", "color": "#27ae60", "bg": "#e9f7ef"}
}

# ------------------------------------------
# 侧边栏与导航
# ------------------------------------------

PAGES = ["单条文本分析", "批量文本分析", "关于系统"]
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

with st.sidebar:
    st.markdown("""
    <div class="sidebar-inner">
        <div class="sidebar-brand">
            <div class="logo">
                <svg width="30" height="30" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path d="M3 12h4l2-4 3 8 2-4h5" stroke="#16a34a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M12 21s-7-4.35-9.5-8.5C.5 8.5 3.2 5.5 6.5 5.5c1.9 0 3.3 1 4.3 2 1-1 2.4-2 4.3-2 3.3 0 6 3 4 7-2.5 4.15-9.1 8.5-9.1 8.5z" stroke="#16a34a" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" opacity="0.35"/>
                </svg>
            </div>
            <div class="title">情感分析系统</div>
        </div>
        <div class="sidebar-nav-caption"></div>
    """, unsafe_allow_html=True)

    if st.button("单条文本分析", use_container_width=True, disabled=(st.session_state.page == "单条文本分析")):
        st.session_state.page = "单条文本分析"
        st.rerun()

    if st.button("批量文本分析", use_container_width=True, disabled=(st.session_state.page == "批量文本分析")):
        st.session_state.page = "批量文本分析"
        st.rerun()

    if st.button("关于系统", use_container_width=True, disabled=(st.session_state.page == "关于系统")):
        st.session_state.page = "关于系统"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-footer-wrap"><div class="sidebar-footer">© 2026 智能系统综合设计 | 细粒度情感分析系统</div></div>',
        unsafe_allow_html=True
    )


# ------------------------------------------
# 预测函数
# ------------------------------------------

def predict_emotion(text):
    """预测单条文本的情感"""
    # 编码
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 预测
    with torch.no_grad():
        outputs = model(**inputs)

    # 处理结果
    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    pred_id = torch.argmax(probabilities).item()
    confidence = probabilities[pred_id].item()

    # 所有情感的概率
    all_probs = {EMOTION_NAMES[i]: probabilities[i].item() for i in range(int(NUM_LABELS))}

    # 粗粒度判断
    coarse = COARSE_MAP[pred_id]

    return {
        'fine_id': pred_id,
        'fine_name': EMOTION_NAMES[pred_id],
        'coarse': coarse,
        'confidence': confidence,
        'all_probabilities': all_probs,
        'top3': sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    }


# ------------------------------------------
# 主界面：内容区（按导航切换）
# ------------------------------------------

st.markdown('<div class="main-header">细粒度情感分析系统</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">面向中文文本的 6 类细粒度情感识别与可视化</div>', unsafe_allow_html=True)

def _coarse_badge_class(coarse: str) -> str:
    if coarse == "正面":
        return "positive"
    if coarse == "负面":
        return "negative"
    return "neutral"

def _emotion_id_by_name(emotion_name: str) -> int:
    for k, v in EMOTION_NAMES.items():
        if v == emotion_name:
            return int(k)
    return 0

def render_text_input_card() -> tuple[str, bool]:
    """渲染：输入文本 + 示例按钮 + 开始分析（作为一个组件）"""
    # 清理之前用于填充文本框的临时状态（不再使用）
    if "pending_example_text" in st.session_state:
        st.session_state.pop("pending_example_text", None)

    with st.container(border=True):
        # 标记：用于 CSS 通过 :has() 选中这个容器做卡片化样式
        # 不要用“打开 div 再放 Streamlit 组件”的方式包裹内容（会渲染出空白长条框）
        st.markdown('<div class="input-card-marker"></div>', unsafe_allow_html=True)

        user_input = st.text_area(
            "请输入要分析的文本",
            key="single_text",
            height=170,
            placeholder="请输入要分析的文本…",
            label_visibility="collapsed",
        )

        examples = [
            ("开心示例", "今天终于拿到了心仪的 offer，感觉所有的努力都值得了！"),
            ("悲伤示例", "听到这个坏消息，我心里非常难过。"),
            ("惊讶示例", "居然中了大奖，太意外了！"),
        ]
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

if "single_text" not in st.session_state:
    st.session_state.single_text = ""
if "show_charts" not in st.session_state:
    # 让“情感概率分布”默认就展示（未分析时显示占位/空图）
    st.session_state.show_charts = True

if st.session_state.page == "单条文本分析":
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
                        result = predict_emotion(user_input.strip())
                    st.session_state.last_result = result
                    st.session_state.show_charts = True

            # 标题行 + 右侧粗粒度结果（中性/正面/负面）
            # 放在按钮逻辑之后，确保点击“开始分析”后能显示最新结果
            if "last_result" in st.session_state:
                _coarse = st.session_state.last_result.get("coarse", "中性")
            else:
                _coarse = "中性"
            _coarse_cls = _coarse_badge_class(_coarse)
            _coarse_cls = "positive" if _coarse_cls == "positive" else ("negative" if _coarse_cls == "negative" else "neutral")
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
                # 初始占位：在还未开始分析时也展示与正式结果相同的布局结构
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
                                    <div class="left">{EMOTION_STYLES[_emotion_id_by_name(em)]['emoji']} {em}</div>
                                    <div class="right">{prob*100:.1f}%</div>
                                </div>'''
                            for em, prob in result["top3"]
                        ])}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # 图表区已默认展示，不再需要“查看图表”按钮

    if st.session_state.show_charts:
        st.markdown('<div class="section-title">📊 情感概率分布</div>', unsafe_allow_html=True)

        # 未分析时：用 0 值占位，保持这块始终出现
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

        # 使用 Streamlit 自带的 container(border=True) 把雷达图和柱状图整体包在同一个长条白色框里
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
                    line_color=_style["color"],  # 红色折线
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
                            dtick=0.2,            # 0,0.2,0.4,0.6,0.8,1.0 -> 多个同心圆
                            gridcolor="#d1d5db",  # 更清晰的一圈圈灰色圆圈
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
                        showgrid=False,          # 只显示横向灰线，不要竖线
                        zerolinecolor="#9ca3af",
                        tickfont=dict(size=12, color="#374151"),
                        title_font=dict(size=13, color="#111827"),
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="#d1d5db",     # 更明显的灰色横线
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

elif st.session_state.page == "批量文本分析":
    st.markdown('<div class="section-title">⛃ 批量文本分析</div>', unsafe_allow_html=True)

    # 判断是否已有批量分析结果：无结果时只展示上传 + 预览界面，有结果时展示结果页
    result_df = st.session_state.get("batch_result_df")

    if result_df is None:
        # ---------- 第一个界面：上传 CSV + 文本预览 + 开始批量分析 ----------
        st.markdown('<div class="batch-page-wrap">', unsafe_allow_html=True)
        st.markdown("<div class='batch-subtitle'>上传包含 text 列的 CSV 文件，预览确认后点击开始批量分析。</div>", unsafe_allow_html=True)

        # 用一个可变 key 来“重置”上传控件，解决用户上传错误文件后无法重新选择的问题
        if "batch_upload_nonce" not in st.session_state:
            st.session_state.batch_upload_nonce = 0
        upload_key = f"batch_uploader_{st.session_state.batch_upload_nonce}"

        with st.container(border=True):
            # 顶部上传区域：一个大虚线框，内部通过 CSS ::before 显示云图标 + 文本
            st.markdown('<div class="batch-uploader-wrap">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "上传CSV文件",
                type=["csv"],
                label_visibility="collapsed",
                key=upload_key,
            )
            # 上传后在同一块区域显示“文件图标 + 文件名”，并保持高度不变
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
                # 尝试自动识别多种常见编码，尽量兼容更多 CSV 文件
                tried_encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "big5", "cp936", "latin1"]
                last_error = None
                for enc in tried_encodings:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=enc)
                        break
                    except Exception as e:  # noqa: BLE001
                        last_error = e
                        df = None

                if df is None and last_error is not None:
                    # 读取 CSV 失败的提示：浅绿色提示条，说明支持的编码范围
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

                    # 提供一个“重新选择文件”按钮，点击后重置上传控件 key 并刷新页面
                    if st.button("重新选择文件", key="reset_bad_csv"):
                        st.session_state.batch_upload_nonce += 1
                        st.rerun()

                if df is not None and "text" not in df.columns:
                    st.error("CSV 文件必须包含 `text` 列。")
                    df = None

            # 文本预览区域：无论是否已上传，都显示出来（未上传时为空占位）
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

            # 底部操作行：左按钮，右进度条（匹配原始截图二的布局）
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
                # 批量分析提示条：浅绿色提示
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
                    r = predict_emotion(text)
                    results.append({
                        "text": text,
                        "fine_emotion": r["fine_name"],
                        "coarse_emotion": r["coarse"],
                        "confidence": r["confidence"],
                        **{f"prob_{k}": v for k, v in r["all_probabilities"].items()},
                    })

                    progress_bar.progress((idx + 1) / total)
                    status_placeholder.write(f"正在分析 {idx + 1}/{total}：{r['fine_name']}")

                st.session_state.batch_result_df = pd.DataFrame(results)
                # 批量分析结束后：进入结果页时自动回到顶部
                st.session_state.scroll_to_top = True
                st.success("分析完成。")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # ---------- 第二个界面：结果展示（饼图 + 表格） ----------
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
                # 返回上传界面：清空 batch_result_df 并刷新
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

            # 使用自定义 HTML 表格，配合上面的 .result-table 样式，确保白底浅灰边框效果
            html_table = result_df.to_html(
                classes="result-table",
                index=False,
                border=0,
                justify="left",
                escape=False,
            )
            st.markdown(html_table, unsafe_allow_html=True)

            # 下载按钮：单独包一层容器，使用 CSS 做成圆角深绿色按钮（参照截图）
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

elif st.session_state.page == "关于系统":
    # 顶部标题
    st.markdown(
        "<div class='section-title'>📚 关于系统</div>",
        unsafe_allow_html=True,
    )

    # 直接输出一整块 HTML，并通过 components.html 真正渲染为 DOM（避免被当作代码块）
    about_html = """
<html>
  <head>
    <style>
      body {
        margin: 0;
        padding: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: transparent;
      }
      .about-root { width: 100%; }
      .about-card-box {
        background: #ffffff;
        border-radius: 16px;
        border: 1.5px solid #d0d5dd;
        box-shadow: 0 12px 28px rgba(17, 24, 39, 0.08);
        padding: 22px 26px;
        width: 100%;
        color: #1f2d3d;
        font-size: 1rem;
      }
      .about-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 14px;
      }
      .about-title {
        font-weight: 950;
        font-size: 1.25rem;
        letter-spacing: .2px;
      }
      .about-subtitle {
        color: #7b8794;
        font-weight: 650;
        font-size: .92rem;
        margin-top: 4px;
      }
      .about-section {
        border-top: 1px dashed #e5e7eb;
        padding-top: 16px;
        margin-top: 16px;
      }
      .about-section:first-of-type {
        border-top: none;
        padding-top: 0;
        margin-top: 0;
      }
      .about-section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 950;
        margin-bottom: 10px;
        font-size: 1.05rem;
      }
      .about-two-col {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px 18px;
        align-items: start;
      }
      .about-emotion-item {
        display: grid;
        grid-template-columns: 88px 1fr;
        gap: 10px;
        padding: 10px 12px;
        border-radius: 12px;
        border: 1px solid #eef0f5;
        background: #ffffff;
      }
      .about-emotion-name {
        font-weight: 950;
        white-space: nowrap;
      }
      .about-emotion-desc {
        color: #52606d;
        font-weight: 650;
      }
      .about-badge-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 12px;
        border-radius: 12px;
        border: 1px solid #eef0f5;
        background: #ffffff;
        margin-bottom: 10px;
      }
      .about-badge {
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 950;
        font-size: .85rem;
        border: 1px solid #e6eaf1;
        background: #f7f9fc;
        color: #52606d;
        white-space: nowrap;
      }
      .about-badge-pos { background: #e9fbf3; border-color: #bfe8d6; color: #0f766e; }
      .about-badge-neg { background: #ffefef; border-color: #ffd1d1; color: #b42318; }
      .about-badge-neu { background: #fdf3e6; border-color: #f1d7b0; color: #8a5a00; }
      .about-mapping-text { font-weight: 750; color: #1f2d3d; }
      .about-tech-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 12px;
      }
      .about-tech-card {
        border: 1px solid #eef0f5;
        border-radius: 12px;
        padding: 12px 12px;
        background: #ffffff;
      }
      .about-tech-k { color: #7b8794; font-weight: 750; font-size: .9rem; }
      .about-tech-v { color: #1f2d3d; font-weight: 950; margin-top: 6px; }
      @media (max-width: 900px) {
        .about-card-box {
          padding: 16px 14px;
        }
        .about-two-col { grid-template-columns: 1fr; }
        .about-tech-grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class="about-root">
      <div class="about-card-box">
        <div class="about-header">
          <div>
            <div class="about-title">关于系统</div>
            <div class="about-subtitle">细粒度（6类）情感识别 + 粗粒度（正/负/中性）映射与可视化</div>
          </div>
        </div>

        <div class="about-section">
          <div class="about-section-title"><span>🖋️</span><span>细粒度情感说明</span></div>
          <div class="about-two-col">
            <div class="about-emotion-item"><div class="about-emotion-name">😊 开心</div><div class="about-emotion-desc">积极、愉悦、满足</div></div>
            <div class="about-emotion-item"><div class="about-emotion-name">😲 惊讶</div><div class="about-emotion-desc">意外、震惊、诧异</div></div>
            <div class="about-emotion-item"><div class="about-emotion-name">😢 悲伤</div><div class="about-emotion-desc">失落、痛苦、泪坠</div></div>
            <div class="about-emotion-item"><div class="about-emotion-name">😨 恐惧</div><div class="about-emotion-desc">害怕、担心、焦虑</div></div>
            <div class="about-emotion-item"><div class="about-emotion-name">😠 生气</div><div class="about-emotion-desc">愤怒、不满、恼火</div></div>
            <div class="about-emotion-item"><div class="about-emotion-name">🤢 厌恶</div><div class="about-emotion-desc">反感、鄙视、嫌弃</div></div>
          </div>
        </div>

        <div class="about-section">
          <div class="about-section-title"><span>📊</span><span>粗粒度映射说明</span></div>
          <div class="about-badge-row">
            <span class="about-badge about-badge-pos">正面</span>
            <span class="about-mapping-text">😊 开心</span>
          </div>
          <div class="about-badge-row">
            <span class="about-badge about-badge-neg">负面</span>
            <span class="about-mapping-text">😢 悲伤　😠 生气　😨 恐惧　🤢 厌恶</span>
          </div>
          <div class="about-badge-row" style="margin-bottom:0;">
            <span class="about-badge about-badge-neu">中性</span>
            <span class="about-mapping-text">😲 惊讶</span>
          </div>
        </div>

        <div class="about-section">
          <div class="about-section-title"><span>🛠️</span><span>技术栈说明</span></div>
          <div class="about-tech-grid">
            <div class="about-tech-card">
              <div class="about-tech-k">模型</div>
              <div class="about-tech-v">BERT-base-Chinese</div>
            </div>
            <div class="about-tech-card">
              <div class="about-tech-k">框架</div>
              <div class="about-tech-v">PyTorch + Transformers</div>
            </div>
            <div class="about-tech-card">
              <div class="about-tech-k">界面</div>
              <div class="about-tech-v">Streamlit + Plotly</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </body>
</html>
"""
    # 高度放大一些，让白框在常见窗口高度下尽量填满可视区域
    components.html(about_html, height=900, scrolling=False)

# ------------------------------------------
# 页脚
# ------------------------------------------
pass