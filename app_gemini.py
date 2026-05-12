import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import tempfile
import os
import platform
import matplotlib.font_manager as fm
import random

# --- [1. 시스템 및 한글 폰트 설정] ---
st.set_page_config(page_title="V-RAP: AI Vocal Analyzer", layout="wide")

@st.cache_resource
def set_korean_font():
    linux_font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(linux_font_path):
        font_prop = fm.FontProperties(fname=linux_font_path)
        plt.rc('font', family=font_prop.get_name())
    else:
        if platform.system() == 'Windows': plt.rc('font', family='Malgun Gothic')
        elif platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# --- [2. 보안 설정: API 키] ---
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("⚠️ 시크릿 설정(Secrets)에서 GOOGLE_API_KEY를 입력해주세요.")

# --- [3. 핵심 분석 함수 (YIN 알고리즘으로 렉 제거)] ---
def analyze_vocal_fast(y, sr):
    # 연산량을 줄이기 위해 yin 사용 (pyin보다 훨씬 빠름)
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
    avg_f0 = np.nanmean(f0)
    
    C4_HZ = 261.63
    gender = "남성형" if avg_f0 < C4_HZ else "여성형"
    if avg_f0 >= 261.63:
        vocal_range = "소프라노" if avg_f0 >= 440 else "알토"
    else:
        vocal_range = "테너" if avg_f0 >= 130 else "베이스"
    return avg_f0, gender, vocal_range

# --- [4. CSS 레이아웃 (중앙 정렬 및 여백 조절)] ---
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
        .report-box { 
            background-color: #1E1E1E; padding: 10px; border-radius: 10px; 
            border: 1px solid #333; text-align: center;
            display: flex; flex-direction: column; justify-content: center; align-items: center;
        }
        h1 { margin-top: -45px; margin-bottom: 5px; font-size: 1.8rem; }
        .stAudioInput { margin-bottom: -15px; }
    </style>
""", unsafe_allow_html=True)

st.title("🎼 너의 목소리가 보여")
tab1, tab2 = st.tabs(["🔍 초고속 분석", "🎯 주파수 게임"])

# --- [탭 1: 분석 모드] ---
with tab1:
    col_cap, col_btn = st.columns([3, 1])
    with col_cap: st.caption("보컬 데이터 분석 및 AI 리포트")
    with col_btn:
        if st.button("🎹 C4 기준음", use_container_width=True):
            st.audio(np.sin(2 * np.pi * 261.63 * np.arange(22050) / 44100), format="audio/wav", sample_rate=44100, autoplay=True)

    audio_data_1 = st.audio_input("목소리를 녹음해주세요", key="input_analysis")

    if audio_data_1:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data_1.getvalue())
            tmp_path = tmp_file.name

        try:
            with st.spinner('⚡ 분석 중...'):
                y, sr = librosa.load(tmp_path, sr=16000, duration=3) # 3초만 로드해서 렉 방지
                avg_f0, gender_type, range_type = analyze_vocal_fast(y, sr)

            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1: st.markdown(f"<div class='report-box'><p style='margin:0; font-size:0.7rem; color:#AAA;'>성별</p><h3 style='margin:0;'>{gender_type}</h3></div>", unsafe_allow_html=True)
            with res_c2: st.markdown(f"<div class='report-box'><p style='margin:0; font-size:0.7rem; color:#AAA;'>음역대</p><h3 style='margin:0;'>{range_type}</h3></div>", unsafe_allow_html=True)
            with res_c3: st.markdown(f"<div class='report-box'><p style='margin:0; font-size:0.7rem; color:#AAA;'>평균 주파수</p><h3 style='margin:0;'>{avg_f0:.1f}Hz</h3></div>", unsafe_allow_html=True)

            col_graph, col_ai = st.columns([1, 1], gap="small")
            with col_graph:
                # [수정] 분석 그래프 높이를 1.5로 대폭 축소
                fig1, ax = plt.subplots(figsize=(4, 1.5))
                librosa.display.waveshow(y, sr=sr, ax=ax, color='#0064FF', alpha=0.5)
                ax.set_title('보컬 파형', fontsize=8); ax.tick_params(labelsize=6)
                plt.tight_layout(); st.pyplot(fig1)

            with col_ai:
                st.markdown("<h4 style='margin-top:10px; margin-bottom:5px; font-size:0.9rem;'>📊 Gemini AI 보컬 리포트</h4>", unsafe_allow_html=True)
                with st.spinner('🤖 AI 작성 중...'):
                    # NotFound 에러 방지를 위해 1.5-flash 모델 사용
                    model = genai.GenerativeModel("gemini-3.1-flash-lite")
                    prompt = f"데이터: {avg_f0:.1f}Hz, {gender_type}, {range_type}. 판정이유, 어울리는 동물, 추천 국내가수(아이유/김동률 제외하고 인디/록/아이돌 중 랜덤하게 1명)를 3줄 이내로 매우 짧게 작성해줘."
                    response = model.generate_content(prompt)
                    st.info(response.text)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

# --- [탭 2: 게임 모드] ---
with tab2:
    if 'target_hz' not in st.session_state:
        st.session_state.target_hz = round(random.uniform(150.0, 300.0), 1)

    st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>🎯 {st.session_state.target_hz} Hz</h1>", unsafe_allow_html=True)
    
    col_g1, col_g2 = st.columns([3, 1])
    with col_g1: st.caption("💡 목소리로 타겟을 맞추세요! (오차범위 ±20Hz)")
    with col_g2:
        if st.button("🔄 타겟 변경", use_container_width=True):
            st.session_state.target_hz = round(random.uniform(150.0, 300.0), 1); st.rerun()

    audio_data_2 = st.audio_input("소리를 내주세요", key="input_game")

    if audio_data_2:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data_2.getvalue()); game_tmp_path = tmp_file.name

        try:
            # 렉 최소화 (3초 제한 및 yin 사용)
            y, sr = librosa.load(game_tmp_path, sr=16000, duration=3)
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
            avg_f0 = np.nanmean(f0)

            if not np.isnan(avg_f0):
                diff = abs(avg_f0 - st.session_state.target_hz)
                
                c1, c2 = st.columns(2)
                with c1: st.markdown(f"<div class='report-box'>내 기록<h2 style='margin:0;'>{avg_f0:.1f} Hz</h2></div>", unsafe_allow_html=True)
                with c2:
                    # 오차범위 20Hz 적용
                    if diff <= 20: 
                        st.balloons()
                        st.markdown(f"<div class='report-box' style='border-color:#00CC66;'><h2 style='color:#00CC66; margin:0;'>SUCCESS!</h2></div>", unsafe_allow_html=True)
                    else: st.markdown(f"<div class='report-box' style='border-color:#FF4B4B;'><h2 style='color:#FF4B4B; margin:0;'>TRY AGAIN</h2></div>", unsafe_allow_html=True)
                
                # [수정] 게임 그래프 초슬림화 및 여백 제거 (잘림 방지)
                graph_col, info_col = st.columns([2, 1])
                with graph_col:
                    fig2, ax = plt.subplots(figsize=(5, 0.6))
                    ax.plot(y[:8000], color='#1F3A5A', linewidth=0.5) 
                    ax.axis('off'); plt.tight_layout(pad=0); st.pyplot(fig2)
                
                with info_col:
                    acc = max(0, min(100, (1 - (diff/60)) * 100))
                    st.markdown(f"<p style='font-size:0.8rem; margin:0;'>정확도: {acc:.1f}% | 오차: {diff:.1f}Hz</p>", unsafe_allow_html=True)
                    st.progress(acc/100)
            else: st.warning("소리가 너무 작습니다. 조금 더 크게 소리 내주세요!")
        finally:
            if os.path.exists(game_tmp_path): os.remove(game_tmp_path)
