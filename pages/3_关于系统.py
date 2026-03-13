# ==========================================
# 关于系统页面
# ==========================================

import streamlit as st
import streamlit.components.v1 as components


def render_page():
    """渲染关于系统页面"""
    # 顶部标题
    st.markdown(
        "<div class='section-title'>📚 关于系统</div>",
        unsafe_allow_html=True,
    )

    # 输出一整块 HTML
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
            <div class="about-subtitle">细粒度（6 类）情感识别 + 粗粒度（正/负/中性）映射与可视化</div>
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
    # 渲染 HTML
    components.html(about_html, height=900, scrolling=False)