import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import tempfile
import os

# --- [설정] 와이드 모드 및 여백 극단적 최적화 ---
st.set_page_config(page_title="vocal_ai", page_icon="🎤", layout="wide")

st.markdown(f"""
    <style>
    /* 1. 전체 디자인 위로 올리기 (상단 여백 제거) */
    .main .block-container {{
        padding-top: 0.5rem !important; 
        padding-bottom: 0rem !important;
        max-width: 95% !important;
    }}
    
    /* 2. 요소 간 간격 최소화 */
    [data-testid="stVerticalBlock"] > div {{
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin-top: -0.2rem !important;
    }}

    /* 3. 평균 주파수 폰트 크기 미세 확대 */
    .hz-font {{
        font-size: 1.2rem !important;
        font-weight: 500;
        margin-top: -0.5rem !important; /* 위쪽 공백 제거 */
    }}

    .blue-text {{ color: #0064FF !important; font-weight: bold; }}
    h1 {{ font-size: 2rem !important; margin-bottom: 0.3rem !important; }}
    h3 {{ font-size: 1.4rem !important; margin-bottom: 0.3rem !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- [설정] Gemini API 키 ---
GOOGLE_API_KEY = "AIzaSyDH4kTbM7iFPMKafjWE65tgjDEZqq-6kAg" 
genai.configure(api_key=GOOGLE_API_KEY)

st.markdown('<h1 style="color: #0064FF; text-align: center;">🎤 음역대 분석 x Gemini AI</h1>', unsafe_allow_html=True)

# --- 성부 판정 로직 ---
def get_vocal_range(pitch):
    if pitch >= 440: return "소프라노 (Soprano)"
    elif 174 <= pitch < 350: return "알토 (Alto)"
    elif 130 <= pitch < 174: return "테너 (Tenor)"
    elif pitch < 130: return "베이스 (Bass)"
    else: return "판독 불가"

# 입력창
_, input_col, _ = st.columns([1, 1.5, 1])
with input_col:
    audio_data = st.audio_input("3초간 마이크에 '아~' 소리를 내주세요!")

if audio_data:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_data.getvalue())
        tmp_path = tmp_file.name

    # 좌우 배치
    col1, col2 = st.columns([1.1, 0.9], gap="medium")

    with col1:
        with st.spinner('분석 중...'):
            y, sr = librosa.load(tmp_path, sr=16000)
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
            avg_f0 = np.nanmean(f0)
            result = get_vocal_range(avg_f0)

            st.subheader("📊 음성 에너지 분석")
            fig, ax = plt.subplots(2, 1, figsize=(10, 4.2)) # 높이 소폭 추가 축소
            librosa.display.waveshow(y, sr=sr, ax=ax[0], color='#0064FF')
            ax[0].set_title('Voice Waveform', fontsize=9)
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', x_axis='time', ax=ax[1])
            ax[1].set_title('Mel-Spectrogram', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

    with col2:
        st.subheader("🔍 판독 결과")
        st.markdown(f'### 판정: <span class="blue-text">{result}</span>', unsafe_allow_html=True)
        # 평균 주파수 폰트 키우고 위쪽 공백 제거
        st.markdown(f'<div class="hz-font">평균 주파수: <b>{avg_f0:.2f} Hz</b></div>', unsafe_allow_html=True)
        
        st.divider()
        
        with st.spinner('🤖 Gemini 분석 중...'):
            try:
                model = genai.GenerativeModel(model_name="gemini-3.1-flash-lite")
                sample_file = genai.upload_file(path=tmp_path)
                
                prompt = """
                당신은 칭찬에 후한 보컬 코치입니다. 서론 없이 '결론'만 말하세요.
                모든 답변은 짧고 강렬하게, 장점만 언급하세요.

                형식:
                1. 키워드: 단어 3개
                2. 한줄평: 10자 내외의 극찬
                3. 매력: 핵심 장점 한 단어
                """
                response = model.generate_content([sample_file, prompt])
                
                st.info("🤖 Gemini AI 음색 리포트")
                st.success(response.text)
                
            except Exception as e:
                st.error(f"Gemini 분석 실패: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
