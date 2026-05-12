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

# --- [3. 핵심 함수] ---
def analyze_vocal_ultra_fast(y, sr):
    # 렉을 최소화하기 위해 yin 알고리즘 사용
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
    avg_f0 = np.nanmean(f0)
    C4_HZ = 261.63
    gender = "남성형" if avg_f0 < C4_HZ else "여성형"
    if avg_f0 >= 261.63:
        vocal_range = "소프라노" if avg_f0 >= 440 else "알토"
    else:
        vocal_range = "테너" if avg_f0 >= 130 else "베이스"
    return avg_f0, gender, vocal_range

# --- [4. CSS 디자인 (힙한 네온 UI)] ---
st.markdown("""
    <style>
        /* 분석 리포트 박스 */
        .report-box { 
            background-color: #1E1E1E; padding: 20px; border-radius: 15px; 
            border: 1px solid #444; text-align: center;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
            margin-bottom: 10px;
        }
        /* 게임 모드 전용 대형 전광판 */
        .game-display {
            background: linear-gradient(145deg, #121212, #252525);
            padding: 40px; border-radius: 25px;
            border: 2px solid #FF4B4B; text-align: center;
            margin: 20px 0px;
        }
        .target-val { font-size: 6rem !important; font-weight: 900; color: #FF4B4B; text-shadow: 0 0 20px rgba(255,75,75,0.5); }
        .my-val { font-size: 5rem !important; font-weight: 800; color: #00BFFF; }
        .success-banner { font-size: 5rem !important; color: #00FF88; font-weight: 900; text-shadow: 0 0 30px rgba(0,255,136,0.6); }
        .fail-banner { font-size: 3rem !important; color: #FFD700; font-weight: 700; }
        
        /* 탭 폰트 조절 */
        .stTabs [data-baseweb="tab"] { font-size: 1.2rem; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

st.title("🎼 너의 목소리가 보여")
tab1, tab2 = st.tabs(["🔍 프로페셔널 분석", "🎯 1:1 주파수 챌린지"])

# --- [탭 1: 분석 모드 (디자인 유지 & 그래프 2개)] ---
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l: st.write("목소리 파형과 주파수를 정밀 분석하여 나만의 보컬 프로필을 생성합니다.")
    with col_r:
        if st.button("🎹 C4 기준음 재생", use_container_width=True):
            st.audio(np.sin(2 * np.pi * 261.63 * np.arange(22050) / 44100), format="audio/wav", sample_rate=44100, autoplay=True)

    audio_data = st.audio_input("마이크에 소리를 내주세요", key="ana_input")

    if audio_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data.getvalue()); tmp_path = tmp_file.name
        try:
            with st.spinner('⚡ 데이터 사이언싱 중...'):
                y, sr = librosa.load(tmp_path, sr=16000)
                avg_f0, gender_type, range_type = analyze_vocal_ultra_fast(y, sr)

            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1: st.markdown(f"<div class='report-box'><p>보컬 타입</p><h2>{gender_type}</h2></div>", unsafe_allow_html=True)
            with res_c2: st.markdown(f"<div class='report-box'><p>메인 음역대</p><h2>{range_type}</h2></div>", unsafe_allow_html=True)
            with res_c3: st.markdown(f"<div class='report-box'><p>평균 주파수</p><h2>{avg_f0:.1f} Hz</h2></div>", unsafe_allow_html=True)

            g_col, a_col = st.columns([1.2, 0.8])
            with g_col:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
                librosa.display.waveshow(y, sr=sr, ax=ax1, color='#0064FF', alpha=0.5)
                ax1.set_title('보컬 에너지 파형', fontsize=12)
                
                D = np.abs(librosa.stft(y, n_fft=2048))
                ax2.plot(librosa.fft_frequencies(sr=sr, n_fft=2048), np.mean(D, axis=1), color='#1F3A5A', linewidth=2)
                ax2.set_xlim(0, 1000); ax2.set_title('주파수 분포', fontsize=12)
                ax2.axvline(x=261.63, color='green', linestyle=':', label='기준 도(C4)')
                if not np.isnan(avg_f0): ax2.axvline(x=avg_f0, color='red', linestyle='--', label=f'내 주파수')
                ax2.legend(); plt.tight_layout(); st.pyplot(fig)

            with a_col:
                st.markdown("### 🤖 AI 매칭 리포트")
                model = genai.GenerativeModel("gemini-3.1-flash-lite")
                prompt = f"데이터: {avg_f0:.1f}Hz, {gender_type}, {range_type}. 판정이유, 어울리는 동물, 추천 국내가수(아이유/김동률 제외)를 3줄 이내로 위트있게 작성."
                response = model.generate_content(prompt)
                st.success(response.text)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

# --- [탭 2: 게임 모드 (렉 최소화 & 전광판 디자인)] ---
with tab2:
    if 'target_hz' not in st.session_state:
        st.session_state.target_hz = round(random.uniform(160.0, 300.0), 1)

    # 전광판 UI
    st.markdown(f"""
        <div class='game-display'>
            <p style='color:#888; font-size:1.5rem;'>TARGET FREQUENCY</p>
            <div class='target-val'>{st.session_state.target_hz} Hz</div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 타겟 변경 (NEW GOAL)", use_container_width=True):
        st.session_state.target_hz = round(random.uniform(160.0, 300.0), 1); st.rerun()

    # 렉을 줄이기 위해 게임용 입력을 아주 간단하게 처리
    game_audio = st.audio_input("목소리로 타겟을 맞추세요!", key="game_input")

    if game_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(game_audio.getvalue()); g_path = tmp_file.name

        try:
            # 렉 방지를 위해 아주 짧게 분석 (2초)
            y, sr = librosa.load(g_path, sr=16000, duration=2)
            f0 = librosa.yin(y, fmin=100, fmax=500)
            avg_f0 = np.nanmean(f0)

            if not np.isnan(avg_f0):
                diff = abs(avg_f0 - st.session_state.target_hz)
                
                # 결과 박스 대형화
                st.markdown(f"""
                    <div class='report-box'>
                        <p style='color:#AAA;'>나의 주파수</p>
                        <div class='my-val'>{avg_f0:.1f} Hz</div>
                    </div>
                """, unsafe_allow_html=True)
                
                if diff <= 20:
                    st.balloons()
                    st.markdown("<div style='text-align:center;' class='success-banner'>PERFECT! 🎉</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align:center;' class='fail-banner'>오차: {diff:.1f} Hz<br>조금만 더 힘내세요!</div>", unsafe_allow_html=True)
                    st.progress(max(0, 100 - int(diff*2)))
            else:
                st.warning("소리가 감지되지 않았습니다.")
        finally:
            if os.path.exists(g_path): os.remove(g_path)
