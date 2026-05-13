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

# --- [1. 기본 설정 및 폰트] ---
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

# --- [2. 보안 설정] ---
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("⚠️ GOOGLE_API_KEY를 설정해주세요.")

# --- [3. 분석 함수] ---
def analyze_vocal_fast(y, sr):
    # 속도를 위해 fmin, fmax 범위를 살짝 좁힘
    f0 = librosa.yin(y, fmin=80, fmax=600)
    avg_f0 = np.nanmean(f0)
    gender = "남성형" if avg_f0 < 261.63 else "여성형"
    vocal_range = "소프라노" if avg_f0 >= 440 else ("알토" if avg_f0 >= 261.63 else ("테너" if avg_f0 >= 130 else "베이스"))
    return avg_f0, gender, vocal_range

# --- [4. CSS 디자인] ---
st.markdown("""
    <style>
        .report-box { 
            background-color: #F8F9FA; padding: 15px; border-radius: 15px; 
            border: 1px solid #DEE2E6; text-align: center; margin-bottom: 15px;
        }
        .report-box h3 { color: #111; margin: 5px 0; font-size: 1.8rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🎼 너의 목소리가 보여")

audio_data = st.audio_input("마이크에 소리를 내주세요", key="ana_input")

if audio_data:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_data.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # --- [1단계: 로컬 데이터 분석 및 그래프 출력 (매우 빠름)] ---
        y, sr = librosa.load(tmp_path, sr=16000)
        avg_f0, gender_type, range_type = analyze_vocal_fast(y, sr)

        st.success("✅ 분석 완료! 시각화 데이터를 먼저 확인하세요.")

        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1: st.markdown(f"<div class='report-box'><small>타입</small><h3>{gender_type}</h3></div>", unsafe_allow_html=True)
        with res_c2: st.markdown(f"<div class='report-box'><small>음역대</small><h3>{range_type}</h3></div>", unsafe_allow_html=True)
        with res_c3: st.markdown(f"<div class='report-box'><small>주파수</small><h3>{avg_f0:.1f} Hz</h3></div>", unsafe_allow_html=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), facecolor='white')
        librosa.display.waveshow(y, sr=sr, ax=ax1, color='#007BFF', alpha=0.6)
        ax1.set_facecolor('#F8F9FA')
        
        D = np.abs(librosa.stft(y, n_fft=2048))
        ax2.plot(librosa.fft_frequencies(sr=sr, n_fft=2048), np.mean(D, axis=1), color='#333')
        ax2.set_xlim(0, 1000)
        ax2.set_facecolor('#F8F9FA')
        ax2.axvline(x=261.63, color='#28A745', linestyle=':')
        ax2.axvline(x=avg_f0, color='#DC3545', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)

        # --- [2단계: AI 리포트 생성 (별도 진행)] ---
        st.markdown("---")
        with st.spinner('🤖 AI 코치가 리포트를 작성 중입니다...'):
            # 가장 빠른 Flash 모델 사용
            model = genai.GenerativeModel("gemini-3.1-flash-lite") 
            prompt = f"데이터: {avg_f0:.1f}Hz, {gender_type}, {range_type}. 1.판정이유, 2.동물, 3.가수(1명)를 3줄로 위트있게 작성."
            response = model.generate_content(prompt)
            st.markdown("### 🤖 AI Vocal Coach Report")
            st.info(response.text)

    except Exception as e:
        st.error(f"오류 발생: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
