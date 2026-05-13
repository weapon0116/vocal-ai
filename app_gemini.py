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

# --- [4. CSS 디자인 (화이트 모드 최적화)] ---
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        .report-box { 
            background-color: #F8F9FA; padding: 15px; border-radius: 15px; 
            border: 1px solid #DEE2E6; text-align: center; margin-bottom: 15px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.05);
        }
        .report-box h3 { color: #111; margin: 5px 0; font-size: 1.8rem; }
        .report-box small { color: #666; font-weight: 600; }
        h1 { font-weight: 800 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🎼 너의 목소리가 보여")
st.caption("목소리를 분석하여 시각적 데이터와 AI 리포트를 제공합니다.")

# --- [메인 레이아웃] ---
col_l, col_r = st.columns([2, 1])
with col_r:
    if st.button("🎹 C4 기준음 듣기", use_container_width=True):
        st.audio(np.sin(2 * np.pi * 261.63 * np.arange(22050) / 44100), format="audio/wav", sample_rate=44100, autoplay=True)

audio_data = st.audio_input("마이크에 소리를 내주세요", key="ana_input")

if audio_data:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_data.getvalue())
        tmp_path = tmp_file.name
    
    try:
        with st.spinner('🚀 데이터를 시각화하고 AI 리포트를 생성하는 중입니다...'):
            # 1. 분석 수행
            y, sr = librosa.load(tmp_path, sr=16000)
            avg_f0, gender_type, range_type = analyze_vocal_fast(y, sr)
            
            # 2. 제미나이 리포트 생성
            model = genai.GenerativeModel("gemini-3.1-flash-lite") # 최신 모델명 권장
            prompt = f"데이터: {avg_f0:.1f}Hz, {gender_type}, {range_type}. 다른 말은 하지 않고 1. 판정이유, 2. 비슷한 동물, 3. 비슷한 가수(1명)를 3줄로 형식에 맞추어 위트있게 작성."
            response = model.generate_content(prompt)
            report_text = response.text

        # ---------------------------------------------------------
        # [수정 포인트] 결과물 배치 순서 변경
        # ---------------------------------------------------------
        st.success("✅ 분석 완료!")

        # 상단: 3단 요약 박스
        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1: st.markdown(f"<div class='report-box'><small>보컬 타입</small><h3>{gender_type}</h3></div>", unsafe_allow_html=True)
        with res_c2: st.markdown(f"<div class='report-box'><small>음역대 판정</small><h3>{range_type}</h3></div>", unsafe_allow_html=True)
        with res_c3: st.markdown(f"<div class='report-box'><small>평균 주파수</small><h3>{avg_f0:.1f} Hz</h3></div>", unsafe_allow_html=True)

        # 중단: 그래프 영역 (화면을 넓게 사용)
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), facecolor='white')
        
        # 파형 그래프
        librosa.display.waveshow(y, sr=sr, ax=ax1, color='#007BFF', alpha=0.6)
        ax1.set_title('보컬 에너지 파형 (Time Domain)', fontsize=11, fontweight='bold')
        ax1.set_facecolor('#F8F9FA')
        
        # 주파수 그래프
        D = np.abs(librosa.stft(y, n_fft=2048))
        ax2.plot(librosa.fft_frequencies(sr=sr, n_fft=2048), np.mean(D, axis=1), color='#333', label='주파수 밀도')
        ax2.set_xlim(0, 1500)
        ax2.set_title('주파수 성분 분석 (Frequency Domain)', fontsize=11, fontweight='bold')
        ax2.set_facecolor('#F8F9FA')
        ax2.axvline(x=261.63, color='#28A745', linestyle=':', linewidth=2, label='C4 (Standard)')
        if not np.isnan(avg_f0):
            ax2.axvline(x=avg_f0, color='#DC3545', linestyle='--', linewidth=2, label=f'내 목소리 ({avg_f0:.1f}Hz)')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        st.pyplot(fig)

        # 하단: AI 제미나이 상세 리포트
        st.markdown("---")
        st.markdown("### 🤖 AI Vocal Coach Report")
        st.info(report_text)

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
