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
        .game-display {
            background: linear-gradient(145deg, #121212, #252525);
            padding: 20px; border-radius: 20px;
            border: 2px solid #FF4B4B; text-align: center; margin: 10px 0px;
        }
        .target-val { font-size: 3.5rem !important; font-weight: 900; color: #FF4B4B; }
        .my-val { font-size: 3rem !important; font-weight: 800; color: #00BFFF; }
        .banner { font-size: 2.5rem !important; font-weight: 900; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🎼 너의 목소리가 보여")
tab1, tab2 = st.tabs(["🔍 보컬 분석", "🎯 주파수 게임"])

# --- [탭 1: 분석 모드 (범례 복구 완료)] ---
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l: st.caption("파형과 주파수 성분을 분석하여 당신의 보컬 특성을 진단합니다.")
    with col_r:
        if st.button("🎹 C4 기준음", use_container_width=True):
            st.audio(np.sin(2 * np.pi * 261.63 * np.arange(22050) / 44100), format="audio/wav", sample_rate=44100, autoplay=True)

    audio_data = st.audio_input("마이크에 소리를 내주세요", key="ana_input")

    if audio_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data.getvalue()); tmp_path = tmp_file.name
        try:
            with st.spinner('⚡ 분석 중...'):
                y, sr = librosa.load(tmp_path, sr=16000)
                avg_f0, gender_type, range_type = analyze_vocal_fast(y, sr)

            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1: st.markdown(f"<div class='report-box'><small>타입</small><h3>{gender_type}</h3></div>", unsafe_allow_html=True)
            with res_c2: st.markdown(f"<div class='report-box'><small>음역대</small><h3>{range_type}</h3></div>", unsafe_allow_html=True)
            with res_c3: st.markdown(f"<div class='report-box'><small>주파수</small><h3>{avg_f0:.1f} Hz</h3></div>", unsafe_allow_html=True)

            g_col, a_col = st.columns([1.3, 0.7])
            with g_col:
                # [범례 핵심 복구 구간]
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
                
                # 그래프 1
                librosa.display.waveshow(y, sr=sr, ax=ax1, color='#0064FF', alpha=0.5)
                ax1.set_title('보컬 에너지 파형', fontsize=12)
                
                # 그래프 2 (주파수 분석 + 범례 추가)
                D = np.abs(librosa.stft(y, n_fft=2048))
                ax2.plot(librosa.fft_frequencies(sr=sr, n_fft=2048), np.mean(D, axis=1), color='#1F3A5A', label='주파수 밀도')
                ax2.set_xlim(0, 1000)
                ax2.set_title('주파수 성분 분석', fontsize=12)
                
                # 가이드 라인 및 범례 설정
                ax2.axvline(x=261.63, color='green', linestyle=':', linewidth=2, label='가온 도 (C4: 261.6Hz)')
                if not np.isnan(avg_f0):
                    ax2.axvline(x=avg_f0, color='red', linestyle='--', linewidth=2, label=f'내 목소리 피크 ({avg_f0:.1f}Hz)')
                
                ax2.legend(loc='upper right', fontsize=10) # 범례 강제 출력
                plt.tight_layout()
                st.pyplot(fig)

            with a_col:
                st.markdown("### 🤖 AI 리포트")
                model = genai.GenerativeModel("gemini-3.1-flash-lite")
                prompt = f"데이터: {avg_f0:.1f}Hz, {gender_type}, {range_type}. 판정이유, 어울리는 동물, 추천 국내가수(아이유/김동률 제외)를 3줄 이내로 위트있게 작성."
                response = model.generate_content(prompt)
                st.info(response.text)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

# --- [탭 2: 게임 모드 (연산 부하 최소화)] ---
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

    # 1. 오디오 입력 받기
    game_audio = st.audio_input("도전!", key="game_input")

    if game_audio:
        # 파일 저장 및 로딩 최적화
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(game_audio.getvalue())
            g_path = tmp_file.name
        
        try:
            # 2. 분석 최적화: sr을 8000으로 낮춰서 데이터 양을 절반으로 줄임
            y, sr = librosa.load(g_path, sr=8000, duration=1.5) 
            
            # 3. YIN 연산 최적화: hop_length를 키워서 연산 횟수 대폭 감소
            # 이 정도만 해도 렉은 거의 사라집니다.
            f0 = librosa.yin(y, fmin=100, fmax=400, sr=sr, hop_length=1024)
            avg_f0 = np.nanmean(f0)

            if not np.isnan(avg_f0):
                diff = abs(avg_f0 - st.session_state.target_hz)
                st.markdown(f"<div class='report-box'><small>나의 주파수</small><div class='my-val'>{avg_f0:.1f} Hz</div></div>", unsafe_allow_html=True)
                
                if diff <= 20:
                    st.balloons()
                    st.markdown("<div style='text-align:center; color:#00FF88;' class='banner'>🎉 SUCCESS!</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align:center; color:#FFD700;' class='banner'>TRY AGAIN! (오차:{diff:.1f}Hz)</div>", unsafe_allow_html=True)
            else:
                st.warning("소리를 조금 더 길게 내주세요!")
        except Exception:
            st.error("분석 중 오류가 발생했습니다.")
        finally:
            if os.path.exists(g_path):
                os.remove(g_path)
