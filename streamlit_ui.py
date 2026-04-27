
import streamlit as st
from AdaphotoRet_run import search_photos

st.set_page_config(page_title="AdaphotoRet · 记忆相册", page_icon="🖼️", layout="wide")

# ---------- 自定义 CSS (完整天蓝泡泡风格) ----------
CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #D4E8FC 0%, #B2D4F5 100%);
    }
    
    .welcome-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 85vh;
        text-align: center;
        padding: 2rem 1rem;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(25px);
        border-radius: 48px;
        margin: 1.5rem;
        box-shadow: 0 25px 50px -8px rgba(30, 80, 130, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    .book {
        width: 240px;
        height: 320px;
        position: relative;
        perspective: 1500px;
        margin: 0 auto;
    }
    .book .page {
        display: block;
        width: 120px;
        height: 320px;
        background-color: rgba(255, 255, 255, 0.7);
        position: absolute;
        top: 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        background-image: linear-gradient(to right, rgba(160, 190, 220, 0.1) 0%, transparent 10%);
        border: 1px solid rgba(255, 255, 255, 0.8);
        transform-origin: left center;
        animation: flipPage 3s ease-in-out forwards;
    }
    .book .page:first-child { left: 0; border-radius: 4px 0 0 4px; border-right: none; }
    .book .page:last-child { left: 120px; border-radius: 0 4px 4px 0; border-left: none; transform-origin: right center; animation: flipPageRight 3s ease-in-out forwards; }
    .book::after {
        content: '';
        position: absolute;
        top: 2%; left: 50%;
        width: 4px; height: 96%;
        background: linear-gradient(to right, rgba(0,0,0,0.1), transparent);
        transform: translateX(-50%);
        z-index: 2;
    }
    @keyframes flipPage {
        0% { transform: rotateY(0deg); }
        30% { transform: rotateY(-25deg); box-shadow: -5px 10px 15px rgba(0,0,0,0.1); }
        70% { transform: rotateY(-5deg); }
        100% { transform: rotateY(0deg); }
    }
    @keyframes flipPageRight {
        0% { transform: rotateY(0deg); }
        30% { transform: rotateY(25deg); box-shadow: 5px 10px 15px rgba(0,0,0,0.1); }
        70% { transform: rotateY(5deg); }
        100% { transform: rotateY(0deg); }
    }
    
    .main-title {
        font-family: 'Georgia', 'Times New Roman', serif;
        font-size: 5.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF7675, #FDCB6E, #00CEC9, #A29BFE, #6C5CE7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 4px;
        margin-bottom: 0.8rem;
        animation: fadeInUp 1.5s;
    }
    .sub-title {
        font-family: 'Georgia', serif;
        font-style: italic;
        font-size: 1.5rem;
        color: #2C3E50;
        margin-top: -10px;
        animation: fadeInUp 1.8s;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .search-card {
        background: rgba(255, 255, 255, 0.35);
        backdrop-filter: blur(20px);
        border-radius: 32px;
        padding: 1.5rem 1.5rem;
        box-shadow: 0 20px 40px -10px rgba(30, 80, 130, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.6);
    }
    
    .polaroid {
        background: transparent !important;
        padding: 0 !important;
        border-radius: 20px;
        box-shadow: none !important;
        border: none !important;
        transition: transform 0.2s ease;
        margin-bottom: 10px;
    }
    .polaroid:hover {
        transform: scale(1.02);
    }
    
    .stButton > button {
        background: transparent;
        border: 2px solid rgba(255,255,255,0.6);
        color: #2C3E50 !important;
        font-weight: bold;
        border-radius: 60px;
        padding: 0.7rem 3rem;
        font-size: 1.2rem;
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.25);
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.4);
        border-color: rgba(255,255,255,0.9);
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.12);
    }
    
    /* 输入框样式 */
    div[data-testid="stTextInput"] input {
        background: rgba(255, 255, 255, 0.35) !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        border-radius: 40px !important;
        padding: 14px 24px !important;
        font-size: 1.05rem !important;
        backdrop-filter: blur(15px);
        color: #1e293b !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #74B9FF !important;
        box-shadow: 0 0 0 3px rgba(116, 185, 255, 0.3) !important;
    }

    .stImage {
        background: transparent !important;
    }
    .stImage > img {
        border-radius: 16px !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    }

    .art-quote {
        background: linear-gradient(135deg, #6C5CE7, #00CEC9, #FDCB6E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    .tips-card {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(15px);
        border-radius: 24px;
        padding: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.5);
        color: #1e293b;
        font-size: 0.92rem;
        line-height: 1.7;
    }

    .little-guy {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        margin-top: 1.5rem;
        position: relative;
    }
    .speech-bubble {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 6px 10px;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        font-size: 0.75rem;
        color: #1e293b;
        text-align: center;
        word-break: break-word;
        max-width: 90px;
        line-height: 1.3;
        margin-bottom: 4px;
    }
    .speech-bubble::after {
        content: '';
        position: absolute;
        bottom: -6px;
        left: 50%;
        margin-left: -6px;
        border-width: 6px;
        border-style: solid;
        border-color: rgba(255, 255, 255, 0.7) transparent transparent transparent;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- 会话状态 ----------
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ---------- 首页 ----------
if st.session_state.page == "welcome":
    st.markdown("""
    <div class="welcome-box">
        <div class="book">
            <span class="page"></span>
            <span class="page"></span>
        </div>
        <h1 class="main-title">AdaphotoRet</h1>
        <p class="sub-title">Finding the moments you remember,<br>with a heart that truly understands</p>
        <div style="margin: 2rem 0; font-size: 1.1rem; color: #2d3748; line-height: 2; max-width: 700px;">
            <p style="font-size: 1.2rem; font-weight: 600;">“能拍就拍，能照就照，想炫的一定要去炫，十年后，再好的相机和技术，也拍不出如此般模样。”</p>
            <p style="font-style: italic; opacity: 0.8;">“We photographers deal in things which are continually vanishing, and when they have vanished there is no contrivance on earth can make them come back again.” — Henri Cartier-Bresson</p>
            <p style="font-size: 2rem; margin: 0.5rem 0;">📷 ✨ 🌊</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    with col2:
        if st.button("✨ 开启记忆之旅 ✨", use_container_width=True, key="start_btn"):
            st.session_state.page = "search"
            st.rerun()

# ---------- 检索页 ----------
else:
    col_back, col_title = st.columns([0.8, 8])
    with col_back:
        if st.button("🏠", key="back_home", help="返回首页"):
            st.session_state.page = "welcome"
            st.session_state.last_results = None
            st.rerun()

    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <h2 style="margin-bottom: 0.3rem;">🖼️ 用描述唤醒记忆</h2>
        <p style="font-style: italic; font-size: 1rem; color: #2C3E50; margin-top: 0.2rem;">
            “照片能捕捉住转瞬即逝的美丽，将永恒定格。” —— 马克·吐温
        </p>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1.2, 2], gap="large")

    with left_col:
        with st.container():
            st.markdown('<div class="search-card">', unsafe_allow_html=True)
            # === 新增引导语（放在输入框上方）===
            st.markdown("""
            <div style="text-align: center; margin-bottom: 0.8rem; color: #2C3E50; font-size: 1rem;">
                <span style="font-size: 1.2rem;">✨</span> 
                <b>请在下方描述你心中的画面</b> 
                <span style="font-size: 1.2rem;">✨</span><br>
                <span style="font-size: 0.85rem; opacity: 0.8;">
                    📝 说出场景、人物、宠物、地点……越具体，记忆越清晰
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            query = st.text_input(
                "输入描述",
                placeholder="例如：菜花田里的小狗；下雨天的老街道；一群在海滩打排球的西方人",
                label_visibility="collapsed"
            )
            search_clicked = st.button("🔍 开始寻找", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        if st.session_state.last_results is None:
            st.markdown("""
            <div class="tips-card">
                <p style="font-weight: 700; margin-top: 0; font-size: 1.05rem;">💡 如何获得最准确的结果？</p>
                <ul style="padding-left: 1.2rem; margin-bottom: 0;">
                    <li>🖊️ 描述尽量具体：场景、人物数量、天气、动作、颜色</li>
                    <li>🐾 如果是宠物，可以说出品种、毛色、姿态（如“蓝白猫”、“橘猫在晒太阳”）</li>
                    <li>🌍 提到城市或地标（如“成都九眼桥的夜景”）</li>
                    <li>🎭 加入情感或氛围（如“开心的生日派对”、“忧伤的雨天”）</li>
                    <li>📸 不要怕口语化，系统能理解“小狗”、“花花”等昵称</li>
                </ul>
                <p style="margin-bottom: 0; margin-top: 0.5rem;">✨ 试试看，让记忆瞬间重现！</p>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.last_results is not None:
            top1, top2, top3, _, _ = st.session_state.last_results
            if top1 is None:
                st.warning("未找到匹配的照片，试试更具体的描述？")
            else:
                guy_col, img1, img2, img3 = st.columns([0.6, 0.8, 0.8, 0.8])
                with guy_col:
                    st.markdown("""
                    <div class="little-guy">
                        <div class="speech-bubble">看看是你心目中的那张照片吗</div>
                        <svg width="70" height="120" viewBox="0 0 70 120" style="display:block; margin: 0 auto;">
                            <circle cx="35" cy="18" r="11" stroke="#2C3E50" stroke-width="2" fill="none"/>
                            <circle cx="31" cy="16" r="1.5" fill="#2C3E50"/>
                            <circle cx="39" cy="16" r="1.5" fill="#2C3E50"/>
                            <path d="M30 21 Q35 25 40 21" stroke="#2C3E50" stroke-width="1.5" fill="none"/>
                            <line x1="35" y1="29" x2="35" y2="65" stroke="#2C3E50" stroke-width="2.5"/>
                            <line x1="35" y1="42" x2="20" y2="55" stroke="#2C3E50" stroke-width="2.5"/>
                            <line x1="35" y1="42" x2="58" y2="37" stroke="#2C3E50" stroke-width="2.5"/>
                            <line x1="58" y1="37" x2="65" y2="34" stroke="#2C3E50" stroke-width="2"/>
                            <line x1="58" y1="37" x2="65" y2="40" stroke="#2C3E50" stroke-width="2"/>
                            <line x1="35" y1="65" x2="25" y2="90" stroke="#2C3E50" stroke-width="2.5"/>
                            <line x1="35" y1="65" x2="45" y2="90" stroke="#2C3E50" stroke-width="2.5"/>
                        </svg>
                    </div>
                    """, unsafe_allow_html=True)

                with img1:
                    if top1:
                        st.markdown('<div class="polaroid">', unsafe_allow_html=True)
                        st.image(top1, use_container_width=True)
                        st.caption("🥇 最佳匹配")
                        st.markdown('</div>', unsafe_allow_html=True)
                with img2:
                    if top2:
                        st.markdown('<div class="polaroid">', unsafe_allow_html=True)
                        st.image(top2, use_container_width=True)
                        st.caption("🥈 第二候选")
                        st.markdown('</div>', unsafe_allow_html=True)
                with img3:
                    if top3:
                        st.markdown('<div class="polaroid">', unsafe_allow_html=True)
                        st.image(top3, use_container_width=True)
                        st.caption("🥉 第三候选")
                        st.markdown('</div>', unsafe_allow_html=True)

    if search_clicked and query:
        with st.spinner("🧠 正在解析语义，筛选最匹配的瞬间..."):
            top1, top2, top3, report_md, table_data = search_photos(query)
            st.session_state.last_results = (top1, top2, top3, report_md, table_data)
            st.session_state.last_query = query
            st.rerun()

    if st.session_state.last_results is not None:
        _, _, _, report_md, table_data = st.session_state.last_results
        if report_md:
            st.markdown("---")
            st.markdown("### 📋 可解释推理报告")
            st.markdown(report_md)
        if table_data:
            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "0": "图片路径",
                    "1": st.column_config.NumberColumn("匹配度", format="%d"),
                    "2": "规则贡献分解"
                }
            )