import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import librosa
import librosa.display
import tempfile
import os
import platform
import matplotlib.font_manager as fm
import random
import scipy.io.wavfile as wav
import io

# --- [1. 시스템 설정 및 폰트] ---
st.set_page_config(page_title="V-RAP AI", layout="wide")

@st.cache_resource
def set_korean_font():
    linux_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(linux_path):
        plt.rc('font', family=fm.FontProperties(fname=linux_path).get_name())
    else:
        plt.rc('font', family='Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# --- [2. API 키 설정] ---
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("API 키를 확인해주세요.")

# --- [3. CSS 디자인 (컴팩트 & 세련)] ---
st.markdown("""
    <style>
        .report-box { background-color: #1E1E1E; padding: 15px; border-radius: 12px; border: 1px solid #444; text-align: center; }
        .game-display { background: linear-gradient(145deg, #121212, #252525); padding: 20px; border-radius: 20px; border: 2px solid #FF4B4B; text-align: center; margin-bottom: 15px; }
        .target-val { font-size: 3.5rem !important; font-weight: 900; color: #FF4B4B; }
        .my-val { font-size: 3rem !important; font-weight: 800; color: #00BFFF; }
        .banner { font-size: 2.5rem !important; font-weight: 900; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🎼 너의 목소리가 보여")
tab1, tab2 = st.tabs(["🔍 전문 분석", "🎯 렉 제로 게임"])

# --- [탭 1: 전문 분석 모드] ---
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l: st.caption("파형과 주파수를 정밀 분석하여 AI 보컬 리포트를 생성합니다.")
    with col_r:
        if st.button("🎹 C4 기준음", use_container_width=True):
            st.audio(np.sin(2 * np.pi * 261.63 * np.arange(22050) / 44100), format="audio/wav", sample_rate=44100, autoplay=True)

    audio_data = st.audio_input("분석을 위해 소리를 내주세요", key="ana_input")

    if audio_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data.getvalue())
            tmp_path = tmp_file.name
        try:
            with st.spinner('⚡ 정밀 분석 중...'):
                y, sr = librosa.load(tmp_path, sr=16000)
                f0 = librosa.yin(y, fmin=80, fmax=500)
                avg_f0 = np.nanmean(f0)
                
                gender = "남성형" if avg_f0 < 261.63 else "여성형"
                v_range = "소프라노" if avg_f0 >= 440 else ("알토" if avg_f0 >= 261.63 else ("테너" if avg_f0 >= 130 else "베이스"))

            # 결과 UI
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f"<div class='report-box'><small>타입</small><h3>{gender}</h3></div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='report-box'><small>음역대</small><h3>{v_range}</h3></div>", unsafe_allow_html=True)
            with c3: st.markdown(f"<div class='report-box'><small>주파수</small><h3>{avg_f0:.1f} Hz</h3></div>", unsafe_allow_html=True)

            g_col, a_col = st.columns([1.3, 0.7])
            with g_col:
                # [범례 포함 그래프]
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
                librosa.display.waveshow(y, sr=sr, ax=ax1, color='#0064FF', alpha=0.5)
                ax1.set_title('보컬 에너지 파형')
                
                D = np.abs(librosa.stft(y, n_fft=2048))
                ax2.plot(librosa.fft_frequencies(sr=sr, n_fft=2048), np.mean(D, axis=1), color='#1F3A5A', label='주파수 밀도')
                ax2.axvline(x=261.63, color='green', linestyle=':', linewidth=2, label='가온 도 (C4)')
                if not np.isnan(avg_f0):
                    ax2.axvline(x=avg_f0, color='red', linestyle='--', linewidth=2, label=f'내 목소리 ({avg_f0:.1f}Hz)')
                ax2.set_xlim(0, 1000); ax2.legend(loc='upper right'); plt.tight_layout()
                st.pyplot(fig)

            with a_col:
                st.markdown("### 🤖 AI 보컬 리포트")
                model = genai.GenerativeModel("gemini-3.1-flash-lite")
                prompt = f"데이터: {avg_f0:.1f}Hz, {gender}, {v_range}. 판정이유, 닮은 동물, 추천가수(아이유/김동률 제외)를 위트있게 3줄 이내 작성."
                response = model.generate_content(prompt)
                st.info(response.text)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

# --- [탭 2: 렉 제로 게임 모드 (초광속 엔진)] ---
with tab2:
    if 'target_hz' not in st.session_state:
        st.session_state.target_hz = round(random.uniform(160.0, 300.0), 1)

    st.markdown(f"""
        <div class='game-display'>
            <p style='color:#888; margin-bottom:0;'>MISSION TARGET</p>
            <div class='target-val'>{st.session_state.target_hz} Hz</div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 타겟 변경", use_container_width=True):
        st.session_state.target_hz = round(random.uniform(160.0, 300.0), 1)
        st.rerun()

    game_audio = st.audio_input("도전! (짧고 명확하게 '아~')", key="game_input")

    if game_audio:
        try:
            # 1. 초광속 로드 (scipy로 메모리 직접 읽기)
            sr, y = wav.read(io.BytesIO(game_audio.read()))
            if len(y.shape) > 1: y = y[:, 0] # 모노 변환
            
            # 2. 데이터 다이어트 (0.5초만 추출 + 부동소수점 변환)
            y = y[:int(sr * 0.5)].astype(float)
            y -= np.mean(y)
            
            # 3. 초광속 자기상관 피치 검출 (Numpy 최적화)
            corr = np.correlate(y, y, mode='full')[len(y)-1:]
            d = np.diff(corr)
            start = np.where(d > 0)[0][0] if len(np.where(d > 0)[0]) > 0 else 0
            peak = np.argmax(corr[start:]) + start
            
            if peak > 0:
                avg_f0 = sr / peak
                # 유효 범위 내 결과 출력
                if 80 < avg_f0 < 500:
                    diff = abs(avg_f0 - st.session_state.target_hz)
                    st.markdown(f"<div class='report-box'><small>나의 기록</small><div class='my-val'>{avg_f0:.1f} Hz</div></div>", unsafe_allow_html=True)
                    
                    if diff <= 20:
                        st.balloons()
                        st.markdown("<div style='text-align:center; color:#00FF88;' class='banner'>🎉 SUCCESS!</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align:center; color:#FFD700;' class='banner'>오차: {diff:.1f} Hz</div>", unsafe_allow_html=True)
                else:
                    st.warning("선명하게 다시 소리 내주세요!")
            else:
                st.warning("소리가 너무 작습니다.")
        except Exception:
            st.error("렉이 발생했습니다. 다시 녹음 버튼을 눌러주세요.")
