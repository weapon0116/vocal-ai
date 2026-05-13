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

# --- [3. 핵심 분석 함수] ---
def analyze_vocal_fast(y, sr):
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
    avg_f0 = np.nanmean(f0)
    C4_HZ = 261.63
    gender = "남성형" if avg_f0 < C4_HZ else "여성형"
    vocal_range = "소프라노" if avg_f0 >= 440 else ("알토" if avg_f0 >= 261.63 else ("테너" if avg_f0 >= 130 else "베이스"))
    return avg_f0, gender, vocal_range

# --- [4. CSS 디자인] ---
st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
        .report-box { 
            background-color: #1E1E1E; padding: 15px; border-radius: 12px; 
            border: 1px solid #444; text-align: center; margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎼 너의 목소리가 보여")
st.caption("목소리를 녹음하면 AI가 정밀 분석하여 리포트를 생성합니다.")

# --- [메인 레이아웃] ---
col_l, col_r = st.columns([2, 1])
with col_r:
    if st.button("🎹 C4 기준음 듣기", use_container_width=True):
        st.audio(np.sin(2 * np.pi * 261.63 * np.arange(22050) / 44100), format="audio/wav", sample_rate=44100, autoplay=True)

audio_data = st.audio_input("마이크에 소리를 내주세요", key="ana_input")

if audio_data:
    # 1. 파일 임시 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_data.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # 2. 빙글빙글(Spinner) 효과 적용 구간
        with st.spinner('🚀 AI가 당신의 목소리를 정밀 분석 중입니다. 잠시만 기다려주세요...'):
            # 오디오 로드 및 데이터 분석
            y, sr = librosa.load(tmp_path, sr=16000)
            avg_f0, gender_type, range_type = analyze_vocal_fast(y, sr)
            
            # AI 리포트 생성 (모델 호출)
            model = genai.GenerativeModel("gemini-3.1-flash-lite")
            prompt = f"데이터: {avg_f0:.1f}Hz, {gender_type}, {range_type}. 다른 말은 하지 않고 1. 판정이유, 2. 비슷한 동물, 3. 비슷한 가수(1명)를 3줄로 형식에 맞추어 너무 길지 않게 약간 위트있게 작성."
            response = model.generate_content(prompt)
            report_text = response.text

        # 3. 결과 출력 (분석이 끝난 후 한꺼번에 짠!)
        st.success("✅ 분석 완료!")
        
        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1: st.markdown(f"<div class='report-box'><small>타입</small><h3>{gender_type}</h3></div>", unsafe_allow_html=True)
        with res_c2: st.markdown(f"<div class='report-box'><small>음역대</small><h3>{range_type}</h3></div>", unsafe_allow_html=True)
        with res_c3: st.markdown(f"<div class='report-box'><small>주파수</small><h3>{avg_f0:.1f} Hz</h3></div>", unsafe_allow_html=True)

        g_col, a_col = st.columns([1.3, 0.7])
        with g_col:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
            
            # 그래프 1: 파형
            librosa.display.waveshow(y, sr=sr, ax=ax1, color='#0064FF', alpha=0.5)
            ax1.set_title('보컬 에너지 파형', fontsize=12)
            
            # 그래프 2: 주파수 분석 및 범례
            D = np.abs(librosa.stft(y, n_fft=2048))
            ax2.plot(librosa.fft_frequencies(sr=sr, n_fft=2048), np.mean(D, axis=1), color='#1F3A5A', label='주파수 밀도')
            ax2.set_xlim(0, 2000)
            ax2.set_title('주파수 성분 분석', fontsize=12)
            
            ax2.axvline(x=261.63, color='green', linestyle=':', linewidth=2, label='가온 도 (C4)')
            if not np.isnan(avg_f0):
                ax2.axvline(x=avg_f0, color='red', linestyle='--', linewidth=2, label=f'내 목소리 ({avg_f0:.1f}Hz)')
            
            ax2.legend(loc='upper right', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

        with a_col:
            st.markdown("### 🤖 AI 분석 결과")
            st.info(report_text)

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
