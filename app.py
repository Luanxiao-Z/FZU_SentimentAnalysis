# ==========================================
# 细粒度情感分析系统 - Streamlit 入口文件
# 负责：页面配置、CSS 加载、模型初始化
# ==========================================

import streamlit as st
import importlib.util
from pathlib import Path
from src.config import ASSETS_DIR, MODEL_PATH
from src.model_handler import EmotionModelHandler
from src.utils import load_css

# ------------------------------------------
# 页面配置
# ------------------------------------------
st.set_page_config(
    page_title="细粒度情感分析系统",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------
# 加载状态管理
# ------------------------------------------
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# ------------------------------------------
# 加载 CSS
# ------------------------------------------
css_path = ASSETS_DIR / 'style.css'
css_content = load_css(str(css_path))
if css_content:
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# ------------------------------------------
# 模型单例模式
# ------------------------------------------
@st.cache_resource
def get_model_handler():
    """获取模型处理器单例"""
    handler = EmotionModelHandler(model_path=str(MODEL_PATH))
    handler.load_model()
    return handler

# 初始化模型并显示加载进度
if not st.session_state.model_loaded:
    # 创建加载界面容器
    loading_container = st.container()
    
    with loading_container:
        # 显示加载动画和文字
        st.markdown("""
        <div class="loading-overlay">
            <div class="loading-content">
                <div class="loading-spinner">
                    <div class="spinner-ring"></div>
                    <div class="spinner-ring"></div>
                    <div class="spinner-ring"></div>
                </div>
                <div class="loading-text">正在初始化情感分析模型</div>
                <div class="loading-subtext">首次加载可能需要几秒钟，请耐心等待...</div>
                <div class="loading-progress">
                    <div class="progress-bar">
                        <div class="progress-fill"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 加载模型
        with st.spinner(''):
            st.session_state.handler = get_model_handler()
            st.session_state.model_loaded = True
        
        # 刷新页面以显示主界面
        st.rerun()

# ------------------------------------------
# 侧边栏导航
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
# 动态导入页面模块
# ------------------------------------------
def load_page_module(page_name: str):
    """动态加载页面模块"""
    pages_dir = Path(__file__).parent / 'pages'
    
    # 根据页面名称确定文件名
    file_map = {
        "单条文本分析": "1_单条文本分析.py",
        "批量文本分析": "2_批量文本分析.py",
        "关于系统": "3_关于系统.py",
    }
    
    filename = file_map.get(page_name)
    if not filename:
        return None
    
    page_path = pages_dir / filename
    
    # 动态导入
    spec = importlib.util.spec_from_file_location("page_module", page_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module

# ------------------------------------------
# 主界面：内容区（按导航切换）
# ------------------------------------------
# 只在模型加载完成后显示主界面
if st.session_state.model_loaded:
    st.markdown('<div class="main-header">细粒度情感分析系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">面向中文文本的 6 类细粒度情感识别与可视化</div>', unsafe_allow_html=True)
    
    # 加载当前页面模块并渲染
    page_module = load_page_module(st.session_state.page)
    if page_module:
        if hasattr(page_module, 'render_page'):
            # 判断页面是否需要 handler 参数
            import inspect
            sig = inspect.signature(page_module.render_page)
            params = list(sig.parameters.keys())
            
            if len(params) > 0:
                # 需要 handler 参数的页面（单条分析、批量分析）
                page_module.render_page(st.session_state.handler)
            else:
                # 不需要参数的页面（关于系统）
                page_module.render_page()
        else:
            st.error(f"页面 '{st.session_state.page}' 未找到")

# ------------------------------------------
# 页脚
# ------------------------------------------
pass