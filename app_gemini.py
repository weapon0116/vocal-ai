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

# --- [3. 핵심 함수 (초광속 YIN 알고리즘)] ---
def analyze_vocal_ultra_fast(y, sr):
    # 렉 방지를 위해 데이터 길이를 3초로 제한하고 yin 사용
    y_short = y[:sr*3]
    f0 = librosa.yin(y_short, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
    avg_f0 = np.nanmean(f0)
    
    C4_HZ = 261.63
    gender = "남성형" if avg_f0 < C4_HZ else "여성형"
    if avg_f0 >= 261.63:
        vocal_range = "소프라노" if avg_f0 >= 440 else "알토"
    else:
        vocal_range = "테너" if avg_f0 >= 130 else "베이스"
    return avg_f0, gender, vocal_range

# --- [4. CSS 디자인 (산으로 갔던 그 멋진 UI)] ---
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        .report-box { 
            background-color: #1E1E1E; padding: 20px; border-radius: 15px; 
            border: 1px solid #333; text-align: center;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        }
        .big-text { font-size: 3rem !important; font-weight: 800; color: #FF4B4B; }
        .success-text { font-size: 4rem !important; color: #00CC66; font-weight: 900; }
        h1 { font-size: 2.5rem; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🎼 너의 목소리가 보여")
tab1, tab2 = st.tabs(["🔍 보컬 분석 리포트", "🎯 주파수 챌린지"])

# --- [탭 1: 분석 모드] ---
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.write("당신의 목소리를 물리적으로 분석하고 AI가 매칭 결과를 알려줍니다.")
    with col_r:
        if st.button("🎹 C4 기준음 듣기", use_container_width=True):
            st.audio(np.sin(2 * np.pi * 261.63 * np.arange(22050) / 44100), format="audio/wav", sample_rate=44100, autoplay=True)

    audio_data = st.audio_input("마이크에 '아~' 소리를 내주세요.", key="ana_input")

    if audio_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data.getvalue())
            tmp_path = tmp_file.name

        try:
            with st.spinner('⚡ 분석 중...'):
                y, sr = librosa.load(tmp_path, sr=16000)
                avg_f0, gender_type, range_type = analyze_vocal_ultra_fast(y, sr)

            # [UI] 산으로 갔던 그 멋진 디자인 결과창
            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1: st.markdown(f"<div class='report-box'><p>성별 판정</p><h3>{gender_type}</h3></div>", unsafe_allow_html=True)
            with res_c2: st.markdown(f"<div class='report-box'><p>음역대</p><h3>{range_type}</h3></div>", unsafe_allow_html=True)
            with res_c3: st.markdown(f"<div class='report-box'><p>평균 주파수</p><h3>{avg_f0:.1f} Hz</h3></div>", unsafe_allow_html=True)

            st.divider()
            
            graph_col, ai_col = st.columns([1.2, 0.8])
            with graph_col:
                # [그래프] 뺴먹지 않고 2개 다 넣었습니다!
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
                librosa.display.waveshow(y, sr=sr, ax=ax1, color='#0064FF', alpha=0.5)
                ax1.set_title('그래프 1: 음성 에너지 파형', fontsize=12)
                
                D = np.abs(librosa.stft(y, n_fft=2048))
                ax2.plot(librosa.fft_frequencies(sr=sr, n_fft=2048), np.mean(D, axis=1), color='#1F3A5A', linewidth=2)
                ax2.set_xlim(0, 1000); ax2.set_title('그래프 2: 주파수 성분 분석', fontsize=12)
                ax2.axvline(x=261.63, color='green', linestyle=':', label='가온 도 (C4)')
                if not np.isnan(avg_f0):
                    ax2.axvline(x=avg_f0, color='red', linestyle='--', label=f'내 목소리: {avg_f0:.1f}Hz')
                ax2.legend()
                plt.tight_layout(); st.pyplot(fig)

            with ai_col:
                st.markdown("### 🤖 AI 매칭 리포트")
                with st.spinner('Gemini 3.1 Flash-Lite가 작성 중...'):
                    # 모델 강제 고정
                    model = genai.GenerativeModel("gemini-3.1-flash-lite")
                    prompt = f"데이터: {avg_f0:.1f}Hz, {gender_type}, {range_type}. 판정이유, 어울리는 동물, 추천 국내가수(아이유/김동률 제외하고 랜덤하게)를 3줄 이내로 매우 짧게 작성."
                    response = model.generate_content(prompt)
                    st.success(response.text)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

# --- [탭 2: 게임 모드 (렉 방지 및 그래프 제거 버전)] ---
with tab2:
    if 'target_hz' not in st.session_state:
        st.session_state.target_hz = round(random.uniform(160.0, 300.0), 1)

    st.markdown("<p style='text-align:center;'>목표 주파수를 맞추세요!</p>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center; font-size:5rem; font-weight:900; color:#FF4B4B;'>{st.session_state.target_hz} Hz</div>", unsafe_allow_html=True)
    
    if st.button("🔄 타겟 변경", use_container_width=True):
        st.session_state.target_hz = round(random.uniform(160.0, 300.0), 1); st.rerun()

    game_audio = st.audio_input("지금 바로 소리를 내보세요!", key="game_input")

    if game_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(game_audio.getvalue()); g_path = tmp_file.name

        try:
            # 렉 줄이려고 그래프 연산 다 뺐습니다!
            y, sr = librosa.load(g_path, sr=16000, duration=2)
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
            avg_f0 = np.nanmean(f0)

            if not np.isnan(avg_f0):
                diff = abs(avg_f0 - st.session_state.target_hz)
                
                # 그래프 대신 결과를 아주 크게 보여줌
                st.markdown(f"<div class='report-box'>내 목소리 주파수<br><span class='big-text'>{avg_f0:.1f} Hz</span></div>", unsafe_allow_html=True)
                
                if diff <= 20:
                    st.balloons()
                    st.markdown("<div style='text-align:center;'><span class='success-text'>🎉 SUCCESS!</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align:center; font-size:2rem; color:#888; margin-top:20px;'>오차: {diff:.1f} Hz<br>좀 더 노력해보세요!</div>", unsafe_allow_html=True)
            else:
                st.warning("소리가 너무 작아 감지되지 않았습니다.")
        finally:
            if os.path.exists(g_path): os.remove(g_path)
