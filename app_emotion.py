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

# ------------------------------------------
# 页面配置
# ------------------------------------------

st.set_page_config(
    page_title="细粒度情感分析系统",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* ----- global ----- */
    .stApp { background: #f6f7fb; }
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #eef0f5;
    }
    .block-container {
        /* 让主内容区域尽量铺满宽度，并留出上下内边距 */
        padding-top: 3.25rem;
        padding-bottom: 2rem;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }

    .main-header {
        font-size: 2.0rem;
        font-weight: 900;
        color: #1f2d3d;
        text-align: left;
        margin: 0.25rem 0 1.0rem 0;
        letter-spacing: .2px;
    }
    .app-subtitle {
        color: #7b8794;
        margin-top: -0.75rem;
        margin-bottom: 1.0rem;
        font-weight: 650;
    }

    /* ----- cards ----- */
    .card {
        background: #ffffff;
        border: 1px solid #eef0f5;
        border-radius: 16px;
        padding: 18px 18px;
        box-shadow: 0 10px 30px rgba(17, 24, 39, 0.06);
    }
    .card + .card { margin-top: 16px; }
    .card-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #1f2d3d;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 12px;
        letter-spacing: .2px;
    }
    .muted {
        color: #7b8794;
        font-size: 0.95rem;
        font-weight: 650;
    }

    /* Streamlit bordered containers -> card look (real wrapping)
       Streamlit may render either stContainer or stVerticalBlockBorderWrapper depending on version. */
    section.main div[data-testid="stContainer"],
    section.main div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1.5px solid #d0d5dd !important;
        border-radius: 16px !important;
        box-shadow: 0 12px 28px rgba(17, 24, 39, 0.08);
        overflow: hidden;
    }
    section.main div[data-testid="stContainer"] > div,
    section.main div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 22px 22px !important;
    }

    /* Equal-height for the two top cards (input + result) */
    section.main div[data-testid="stContainer"]:has(.input-card-marker),
    section.main div[data-testid="stVerticalBlockBorderWrapper"]:has(.input-card-marker),
    section.main div[data-testid="stContainer"]:has(.result-card-marker),
    section.main div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-card-marker) {
        height: 360px;
        display: flex;
        flex-direction: column;
    }

    /* ----- sidebar ----- */
    /* Streamlit sidebar content container */
    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
        display: flex;
        flex-direction: column;
        height: 100vh;
        padding-bottom: 72px; /* reserve room for fixed footer */
    }
    .sidebar-inner {
        width: min(260px, 92%);
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 14px 6px 6px 6px;
        margin-bottom: 14px;
        justify-content: center;
        width: 100%;
    }
    .sidebar-brand .logo {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: transparent;
        box-shadow: none;
        display: grid;
        place-items: center;
        color: #16a34a;
    }
    .sidebar-brand .title {
        font-weight: 950;
        color: #1f2d3d;
        font-size: 1.65rem;
        line-height: 1.0;
        letter-spacing: .2px;
    }
    .sidebar-nav-caption {
        color: #9aa5b1;
        font-weight: 800;
        font-size: .85rem;
        padding: 4px 10px 8px 10px;
        margin-top: 6px;
        text-align: center;
    }
    /* Sidebar navigation buttons (match screenshot) */
    section[data-testid="stSidebar"] div.stButton {
        display: flex;
        justify-content: center;
        margin: 18px 0;
        width: 100%;
    }
    section[data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        background: #eaf7f3;
        border: 1px solid #bfe8d6;
        color: #1f2d3d;
        border-radius: 6px;
        padding: 14px 14px;
        min-height: 52px;
        font-size: 1.08rem;
        font-weight: 950;
        box-shadow: 0 6px 14px rgba(31, 45, 61, 0.18);
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 18px rgba(31, 45, 61, 0.22);
        border-color: #9fe0c6;
    }
    section[data-testid="stSidebar"] div.stButton > button:disabled {
        background: #dff3ea;
        border-color: #9fe0c6;
        box-shadow: inset 0 0 0 1px rgba(16, 185, 129, 0.25), 0 6px 14px rgba(31, 45, 61, 0.16);
        opacity: 1.0;
    }

    .sidebar-footer-wrap {
        position: fixed;
        bottom: 12px;
        left: 0;
        width: 21rem;          /* Streamlit default sidebar width */
        padding: 0 0.75rem;
        z-index: 9999;
        pointer-events: none;
    }
    .sidebar-footer {
        width: 100%;
        text-align: center;
        color: #7b8794;
        font-size: 0.8rem;
        font-weight: 650;
        padding: 0;
        white-space: nowrap;
    }

    /* On narrow screens, keep within sidebar */
    @media (max-width: 900px) {
        .sidebar-footer-wrap { width: 100%; }
    }

    /* ----- result panel ----- */
    .result-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 12px;
    }
    .emotion-main {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.35rem;
        font-weight: 900;
        color: #1f2d3d;
    }
    .badge {
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 900;
        font-size: .85rem;
        border: 1px solid #e6eaf1;
        background: #f7f9fc;
        color: #52606d;
        white-space: nowrap;
    }
    .badge.positive { background: #e9fbf3; border-color: #bfe8d6; color: #0f766e; }
    .badge.negative { background: #ffefef; border-color: #ffd1d1; color: #b42318; }
    .badge.neutral  { background: #f3f4f6; border-color: #e5e7eb; color: #374151; }
    .conf { margin-top: 4px; color: #7b8794; font-weight: 800; }
    .top3-row { margin-top: 8px; display: grid; gap: 10px; }
    .top3-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 10px 12px;
        border-radius: 12px;
        border: 1px solid #eef0f5;
        background: #ffffff;
    }
    .top3-item .left { display: flex; align-items: center; gap: 10px; font-weight: 900; color: #1f2d3d; }
    .top3-item .right { font-weight: 900; color: #1f2d3d; }

    /* Streamlit widgets tweak */
    div[data-testid="stTextArea"] textarea {
        border-radius: 18px;
        border: 1px solid #C5DEDB;   /* 与批量界面同色 */
        background: #FFFFFF;
        padding: 14px 14px;
        font-size: 1.0rem;
        color: #3C5C58;
    }
    div[data-testid="stTextArea"] textarea::placeholder {
        color: #9aa5b1;
        opacity: 1;
    }
    div[data-testid="stTextArea"] textarea:focus {
        outline: none !important;
        box-shadow: none !important;
        border: 1px solid #C5DEDB !important;
    }
    div[data-testid="stTextArea"] textarea:focus-visible {
        outline: none !important;
    }

    /* Prevent truncation of explanatory text */
    .muted {
        white-space: normal;
        overflow: visible;
        word-break: break-word;
    }
    /* Main-area buttons (avoid overriding sidebar look) */
    section.main div.stButton > button {
        border-radius: 999px;
        font-weight: 900;
        padding: 0.45rem 0.9rem;
        min-height: 40px;
        background: #ffffff;
        border: 1px solid #d0d5dd;
        color: #111827;
        box-shadow: 0 4px 10px rgba(17, 24, 39, 0.08);
    }
    /* Force non-primary buttons to match (Streamlit varies by version) */
    section.main button[data-testid^="baseButton-"]:not([data-testid$="primary"]) {
        background: #FFFFFF !important;
        border: 1px solid #C5DEDB !important;   /* 与上传框同色系 */
        color: #3C5C58 !important;
        border-radius: 999px !important;
        box-shadow: 0 4px 10px rgba(17, 24, 39, 0.08) !important;
    }

    /* Section titles to match screenshot */
    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.35rem;
        font-weight: 950;
        color: #1f2d3d;
        margin: 6px 0 14px 0;
        letter-spacing: .2px;
    }

    /* Batch upload dropzone (match截图配色) */
    .batch-uploader-wrap { position: relative; }
    /* 兼容不同版本的 data-testid 命名 */
    div[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploader"],
    div[data-testid^="stFileUploader"] {
        position: relative;
        border: 3px dashed #C5DEDB !important;   /* 虚线边框色 */
        border-radius: 26px !important;
        background: #FFFFFF !important;          /* 背景色 */
        padding: 48px 18px !important;
        min-height: 180px;
        overflow: hidden;
    }
    /* 隐藏内部默认内容，只保留一个可点击区域 */
    div[data-testid="stFileUploaderDropzone"] * ,
    div[data-testid="stFileUploader"] * ,
    div[data-testid^="stFileUploader"] * {
        opacity: 0 !important;
    }
    /* 在虚线框内部居中显示云图标 + 文本 */
    div[data-testid="stFileUploaderDropzone"]::before,
    div[data-testid="stFileUploader"]::before,
    div[data-testid^="stFileUploader"]::before {
        content: "☁\\A拖拽或点击上传CSV\\A需包含'text'列";
        white-space: pre-line;
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #3C5C58;
        font-weight: 950;
        font-size: 1.2rem;
        text-align: center;
        pointer-events: none;
        line-height: 1.5;
    }

    /* Input action row */
    .hint {
        color: #9aa5b1;
        font-weight: 700;
        font-size: .95rem;
        margin-top: -6px;
        margin-bottom: 10px;
    }

    /* Make primary button look像PSD里浅绿按钮 */
    section.main button[kind="primary"] {
        background: #2C7268 !important;          /* 按钮底色 */
        border: 1px solid #2C7268 !important;
        color: #ffffff !important;
        border-radius: 999px !important;
        min-height: 44px !important;
        padding: 0.55rem 1.15rem !important;
        box-shadow: 0 12px 24px rgba(43, 79, 75, 0.26);
    }
    /* Primary button selector for Streamlit 1.5x */
    section.main button[data-testid$="primary"] {
        background: #2C7268 !important;
        border: 1px solid #2C7268 !important;
        color: #ffffff !important;
        border-radius: 999px !important;
        min-height: 44px !important;
        padding: 0.55rem 1.15rem !important;
        box-shadow: 0 12px 24px rgba(43, 79, 75, 0.26) !important;
    }

    /* Result right badges */
    .badge-stack {
        display: flex;
        flex-direction: column;
        gap: 10px;
        align-items: flex-end;
    }
    .mini-badge {
        padding: 6px 10px;
        border-radius: 10px;
        font-weight: 900;
        font-size: .82rem;
        border: 1px solid #e6eaf1;
        background: #f7f9fc;
        color: #52606d;
        white-space: nowrap;
    }

    /* Progress bar colors for批量分析 */
    div[role="progressbar"] > div {
        background-color: #B5CFCD !important;   /* 填充色 */
    }
    div[role="progressbar"] {
        background-color: #F5F5F5 !important;   /* 背景色 */
    }

    /* About page card text颜色统一为深黑 */
    .about-card {
        color: #1f2d3d;
        font-size: 0.98rem;
    }
    .about-card strong {
        color: #1f2d3d;
    }
</style>
""", unsafe_allow_html=True)


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
    with st.container(border=True):
        st.markdown('<div class="input-card-marker" style="display:none;"></div>', unsafe_allow_html=True)

        user_input = st.text_area(
            "请输入要分析的文本",
            key="single_text",
            height=170,
            placeholder="请输入要分析的文本…",
            label_visibility="collapsed",
        )

        ex_cols = st.columns([1, 1, 1, 1.2], gap="small")
        examples = [
            ("开心示例", "今天终于拿到了心仪的 offer，感觉所有的努力都值得了！"),
            ("悲伤示例", "听到这个坏消息，我心里非常难过。"),
            ("惊讶示例", "居然中了大奖，太意外了！"),
        ]
        for i, (label, text) in enumerate(examples):
            with ex_cols[i]:
                if st.button(label, key=f"pill_{i}", use_container_width=True):
                    st.session_state.single_text = text
                    st.rerun()

        with ex_cols[3]:
            analyze_btn = st.button("开始分析", type="primary", use_container_width=True)

    return user_input, analyze_btn

if "single_text" not in st.session_state:
    st.session_state.single_text = ""
if "show_charts" not in st.session_state:
    st.session_state.show_charts = False

if st.session_state.page == "单条文本分析":
    st.markdown('<div class="section-title">📝 单条文本分析</div>', unsafe_allow_html=True)

    col_input, col_result = st.columns([2.1, 1.4], gap="large")

    with col_input:
        user_input, analyze_btn = render_text_input_card()

    with col_result:
        with st.container(border=True):
            st.markdown('<div class="result-card-marker" style="display:none;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🧾 分析结果</div>', unsafe_allow_html=True)

            if analyze_btn:
                if not user_input.strip():
                    st.warning("请输入文本后再分析。")
                else:
                    with st.spinner("正在分析情感..."):
                        result = predict_emotion(user_input.strip())
                    st.session_state.last_result = result
                    st.session_state.show_charts = True

            if "last_result" not in st.session_state:
                st.markdown('<div class="muted">在左侧输入文本并点击「开始分析」，这里会展示情感结果与 Top-3。</div>', unsafe_allow_html=True)
            else:
                result = st.session_state.last_result
                emotion_id = int(result["fine_id"])
                style = EMOTION_STYLES[emotion_id]
                badge_cls = _coarse_badge_class(result["coarse"])

                st.markdown(
                    f"""
                    <div class="result-head">
                        <div class="emotion-main">{style['emoji']} {result['fine_name']}</div>
                        <div class="badge-stack">
                            <div class="badge {badge_cls}">{result['coarse']}</div>
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

                btn_spacer, btn_right = st.columns([1, 1], gap="small")
                with btn_spacer:
                    st.write("")
                with btn_right:
                    view = st.button("查看图表", use_container_width=True)
                if view:
                    st.session_state.show_charts = True

    if "last_result" in st.session_state and st.session_state.show_charts:
        st.markdown('<div class="section-title">📊 情感概率分布</div>', unsafe_allow_html=True)
        result = st.session_state.last_result
        emotion_id = int(result["fine_id"])
        style = EMOTION_STYLES[emotion_id]

        categories = list(result["all_probabilities"].keys())
        values = list(result["all_probabilities"].values())

        st.markdown('<div class="card">', unsafe_allow_html=True)
        chart_left, chart_right = st.columns([1, 1], gap="large")
        with chart_left:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="情感概率",
                line_color=style["color"],
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
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
                showlegend=False,
                height=360,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "批量文本分析":
    st.markdown('<div class="section-title">⛃ 批量文本分析</div>', unsafe_allow_html=True)

    # 左：上传 + 预览 + 按钮 + 进度条；右：统计结果
    left_col, right_col = st.columns([1.6, 1.4], gap="large")

    with left_col:
        # 顶部上传区域：一个大虚线框，内部通过 CSS ::before 显示云图标 + 文本
        st.markdown('<div class="batch-uploader-wrap">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("上传CSV文件", type=["csv"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        df = None
        if uploaded_file:
            # 兼容不同编码的 CSV（优先按 UTF-8 读取，失败再尝试常见的 GBK）
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(uploaded_file, encoding="gbk")
                except Exception as e:  # noqa: BLE001
                    st.error(f"无法读取 CSV 文件，请确认编码为 UTF-8 或 GBK。错误信息：{e}")
                    df = None

            if df is not None and "text" not in df.columns:
                st.error("CSV 文件必须包含 `text` 列。")
                df = None

        # 文本预览区域：无论是否已上传，都显示出来（未上传时为空占位）
        st.markdown(
            "<div style='font-weight:900;font-size:1.05rem;color:#1f2d3d;'>文本预览</div>",
            unsafe_allow_html=True,
        )
        if df is not None:
            st.markdown(f"<div class='muted'>共读取 {len(df)} 条数据</div>", unsafe_allow_html=True)
            preview_samples = df["text"].astype(str).head(50).tolist()
            preview_text = "\n\n".join(preview_samples)
        else:
            st.markdown("<div class='muted'>共读取 0 条数据</div>", unsafe_allow_html=True)
            preview_text = ""

        st.text_area(
            "当前文本预览",
            value=preview_text,
            height=220,
            label_visibility="collapsed",
            disabled=True,
        )

        # 底部操作行：左按钮，右进度条（匹配截图布局）
        action_left, action_right = st.columns([1.1, 3.2], gap="medium")
        with action_left:
            run_batch = st.button(
                "开始批量分析",
                type="primary",
                use_container_width=True,
            )
        with action_right:
            # 先渲染一个 0% 的进度条，让布局与截图一致
            progress_bar = st.progress(0.0)
            status_placeholder = st.empty()

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

            result_df = pd.DataFrame(results)
            st.success("分析完成。")

            # 在右侧列展示统计图与结果表
            with right_col:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">细粒度分布</div>', unsafe_allow_html=True)
                fine_counts = result_df["fine_emotion"].value_counts()
                fig_pie = px.pie(
                    values=fine_counts.values,
                    names=fine_counts.index,
                    title="细粒度情感分布",
                    color=fine_counts.index,
                    color_discrete_map={EMOTION_STYLES[i]["name"]: EMOTION_STYLES[i]["color"] for i in range(6)},
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">粗粒度分布</div>', unsafe_allow_html=True)
                coarse_counts = result_df["coarse_emotion"].value_counts()
                fig_coarse = px.pie(
                    values=coarse_counts.values,
                    names=coarse_counts.index,
                    title="粗粒度情感分布",
                    color=coarse_counts.index,
                    color_discrete_map={"正面": "#2ecc71", "负面": "#e74c3c", "中性": "#95a5a6"},
                )
                st.plotly_chart(fig_coarse, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">详细结果</div>', unsafe_allow_html=True)
                st.dataframe(result_df, use_container_width=True)
                csv = result_df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="下载完整结果",
                    data=csv,
                    file_name="emotion_analysis_results.csv",
                    mime="text/csv",
                )
                st.markdown("</div>", unsafe_allow_html=True)

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
      .about-root {
        width: 100%;
        display: flex;
        justify-content: flex-start;
      }
      .about-card-box {
        background: #ffffff;
        border-radius: 16px;
        border: 1.5px solid #d0d5dd;
        box-shadow: 0 12px 28px rgba(17, 24, 39, 0.08);
        padding: 18px 22px;
        min-height: 720px;         /* 明显拉高白框，尽量贴近底部 */
        width: 100%;               /* 让白框占满主内容宽度 */
        color: #1f2d3d;
        font-size: clamp(1.0rem, 0.6rem + 0.8vw, 1.35rem);  /* 字号随宽度明显变大 */
      }
      .about-section-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 900;
        margin-bottom: 8px;
      }
      .about-grid {
        display: grid;
        grid-template-columns: 80px 1fr 80px 1fr;
        row-gap: 6px;
        column-gap: 10px;
      }
      .about-divider {
        border-top: 1px dashed #e5e7eb;
        margin: 8px 0 4px 0;
      }
      .about-badge-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 4px;
      }
      .about-badge {
        padding: 4px 10px;
        border-radius: 999px;
        font-weight: 900;
        font-size: 0.85rem;
      }
      .about-badge-pos {
        background: #e9fbf3;
        border: 1px solid #bfe8d6;
      }
      .about-badge-neg {
        background: #ffefef;
        border: 1px solid #ffd1d1;
      }
      .about-badge-neu {
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
      }
      .about-tech-list > div {
        margin-bottom: 2px;
      }
      @media (max-width: 900px) {
        .about-card-box {
          padding: 16px 14px;      /* 小屏时适当收紧内边距 */
        }
        .about-grid {
          grid-template-columns: 80px 1fr;  /* 窄屏改为两列，避免挤压 */
        }
      }
    </style>
  </head>
  <body>
    <div class="about-root">
      <div class="about-card-box">
        <div>
          <div class="about-section-title">
            <span style="font-size:1.1rem;">🖋️</span>
            <span>细粒度情感说明</span>
          </div>
          <div class="about-grid">
            <div><strong>😊 开心</strong></div><div>积极、愉悦、满足</div>
            <div><strong>😲 惊讶</strong></div><div>意外、震惊、诧异</div>
            <div><strong>😢 悲伤</strong></div><div>失落、痛苦、泪坠</div>
            <div><strong>😨 恐惧</strong></div><div>害怕、担心、焦虑</div>
            <div><strong>😠 生气</strong></div><div>愤怒、不满、恼火</div>
            <div><strong>🤢 厌恶</strong></div><div>反感、鄙视、嫌弃</div>
          </div>
        </div>

        <div class="about-divider"></div>

        <div>
          <div class="about-section-title" style="margin-top:4px;">
            <span style="font-size:1.1rem;">📊</span>
            <span>粗粒度映射说明</span>
          </div>
          <div style="font-size:0.95rem;">
            <div class="about-badge-row">
              <span class="about-badge about-badge-pos">正面</span>
              <span>😊 开心</span>
            </div>
            <div class="about-badge-row">
              <span class="about-badge about-badge-neg">负面</span>
              <span>😢 悲伤　😠 生气　😨 恐惧　🤢 厌恶</span>
            </div>
            <div class="about-badge-row">
              <span class="about-badge about-badge-neu">中性</span>
              <span>😲 惊讶</span>
            </div>
          </div>
        </div>

        <div class="about-divider"></div>

        <div>
          <div class="about-section-title" style="margin-top:4px;">
            <span style="font-size:1.1rem;">🛠️</span>
            <span>技术栈说明</span>
          </div>
          <div class="about-tech-list">
            <div><strong>模型：</strong>BERT-base-Chinese</div>
            <div><strong>框架：</strong>PyTorch + Transformers</div>
            <div><strong>界面：</strong>Streamlit + Plotly</div>
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