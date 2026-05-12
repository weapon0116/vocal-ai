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
st.set_page_config(page_title="vocal_ai", layout="wide")

@st.cache_resource
def set_korean_font():
    # 1. 리눅스 서버(Streamlit Cloud) 나눔폰트 경로
    linux_font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    
    if os.path.exists(linux_font_path):
        # 서버 환경: 설치된 나눔고딕 등록
        font_prop = fm.FontProperties(fname=linux_font_path)
        plt.rc('font', family=font_prop.get_name())
    else:
        # 로컬 환경 (Windows/Mac)
        if platform.system() == 'Windows':
            plt.rc('font', family='Malgun Gothic')
        elif platform.system() == 'Darwin':
            plt.rc('font', family='AppleGothic')
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# --- [2. 보안 설정: API 키] ---
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("⚠️ 시크릿 설정(Secrets)에서 GOOGLE_API_KEY를 확인해주세요.")

# --- [3. 핵심 함수] ---
def analyze_gender_by_c4(avg_f0):
    C4_HZ = 261.63
    if np.isnan(avg_f0): return "측정 불가", "알 수 없음"
    
    gender = "남성형 (Male-like)" if avg_f0 < C4_HZ else "여성형 (Female-like)"
    if avg_f0 >= 261.63:
        vocal_range = "소프라노 (Soprano)" if avg_f0 >= 440 else "알토 (Alto)"
    else:
        vocal_range = "테너 (Tenor)" if avg_f0 >= 130 else "베이스 (Bass)"
    return gender, vocal_range

def play_piano_c4():
    sr, duration, f0 = 44100, 1.0, 261.63
    t = np.linspace(0, duration, int(sr * duration), False)
    tone = (1.0 * np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * f0 * 2 * t))
    tone = tone * np.exp(-4 * t)
    return tone / np.max(np.abs(tone)), sr

# --- [4. 메인 화면 구성 (수정 전 재생바 스타일)] ---
st.title("🎼 너의 목소리가 보여")

if st.button("🎹 기준점: 가온 도(C4) 듣기"):
    audio_buffer, sr_p = play_piano_c4()
    # 수정 전 스타일: 재생바가 그대로 노출됩니다.
    st.audio(audio_buffer, format="audio/wav", sample_rate=sr_p, autoplay=True)

st.divider()

audio_data = st.audio_input("가온 도(C4) 음으로 마이크에 '아~'소리를 내주세요.")

if audio_data:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_data.getvalue())
        tmp_path = tmp_file.name

    try:
        with st.spinner('🔍 주파수 분석 중...'):
            y, sr = librosa.load(tmp_path, sr=16000)
            y, _ = librosa.effects.trim(y)
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
            avg_f0 = np.nanmean(f0)
            gender_type, range_type = analyze_gender_by_c4(avg_f0)

        # 결과 표시
        st.subheader(f"📊 분석 결과: {gender_type} / {range_type}")
        st.write(f"평균 주파수: **{avg_f0:.2f} Hz** (가온 도 261.63 Hz 대비 {'낮음' if avg_f0 < 261.63 else '높음'})")
        
        col1, col2 = st.columns([1.2, 0.8], gap="medium")
        
        with col1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            librosa.display.waveshow(y, sr=sr, ax=ax1, color='#0064FF', alpha=0.8)
            ax1.set_title('그래프 1: 음성 에너지 파형', fontsize=12)
            
            n_fft = 2048
            D = np.abs(librosa.stft(y, n_fft=n_fft))
            avg_D = np.mean(D, axis=1)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            ax2.plot(freqs, avg_D, color='#1F3A5A', linewidth=2)
            ax2.set_xlim(0, 2000)
            ax2.set_title('그래프 2: 주파수 성분 분석 (진동수 vs 진폭)', fontsize=12)
            ax2.set_xlabel('진동수 (Hz)')
            ax2.set_ylabel('진폭 (Amplitude)')
            
            # 범례에 한글 적용
            ax2.axvline(x=261.63, color='green', linestyle=':', label='가온 도 (C4)')
            if not np.isnan(avg_f0):
                ax2.axvline(x=avg_f0, color='red', linestyle='--', label=f'내 목소리 피크: {avg_f0:.1f}Hz')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            with st.spinner('🤖 Gemini AI 보컬 리포트 생성 중...'):
                model = genai.GenerativeModel("gemini-3.1-flash-lite")
                sample_file = genai.upload_file(path=tmp_path)
                
                prompt = f"""
                분석 데이터: {avg_f0:.2f}Hz ({gender_type}).
                1. [물리적 판정]: 왜 이 목소리가 {gender_type}로 분류되는지 가온 도(261.63Hz)와 비교하여 설명해라.
                2. [동물/가수 매칭]: 성별과 음색에 딱 맞는 동물 1마리와 가수 1명을 추천해라.
                짧고 명확하게 답변해라.
                """
                response = model.generate_content([sample_file, prompt])
                st.info("📑 Gemini AI 매칭 리포트")
                st.success(response.text)

    except Exception as e:
        st.error(f"오류 발생: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
