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

# --- [4. 메인 화면 구성] ---
st.title("🎼 너의 목소리가 보여")

# 탭으로 페이지 분리 (한 페이지에 다 안 나와도 된다고 하셔서 가독성 높임)
tab1, tab2 = st.tabs(["🔍 초심 분석 모드", "🎯 주파수 맞추기 게임"])

# --- [탭 1: 분석 모드 (원본 코드 기능 완벽 복구)] ---
with tab1:
    if st.button("🎹 기준점: 가온 도(C4) 듣기", key="btn_c4"):
        audio_buffer, sr_p = play_piano_c4()
        st.audio(audio_buffer, format="audio/wav", sample_rate=sr_p, autoplay=True)

    audio_data = st.audio_input("마이크에 '아~' 소리를 내주세요.", key="input_analysis")

    if audio_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data.getvalue())
            tmp_path = tmp_file.name

        try:
            with st.spinner('🔍 분석 중...'):
                y, sr = librosa.load(tmp_path, sr=16000)
                y, _ = librosa.effects.trim(y)
                # 렉 방지를 위해 yin 사용 (초심 로직 유지)
                f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
                avg_f0 = np.nanmean(f0)
                gender_type, range_type = analyze_gender_by_c4(avg_f0)

            st.subheader(f"📊 분석 결과: {gender_type} / {range_type}")
            st.write(f"평균 주파수: **{avg_f0:.2f} Hz**")
            
            col1, col2 = st.columns([1.2, 0.8], gap="medium")
            
            with col1:
                # [그래프 복구] 원본 스타일의 2단 그래프 (파형 + 주파수 분석)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # 그래프 1: 파형
                librosa.display.waveshow(y, sr=sr, ax=ax1, color='#0064FF', alpha=0.8)
                ax1.set_title('그래프 1: 음성 에너지 파형', fontsize=12)
                
                # 그래프 2: 주파수 분석 (원본 로직 그대로)
                n_fft = 2048
                D = np.abs(librosa.stft(y, n_fft=n_fft))
                avg_D = np.mean(D, axis=1)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
                
                ax2.plot(freqs, avg_D, color='#1F3A5A', linewidth=2)
                ax2.set_xlim(0, 1000) # 가독성을 위해 범위를 1000Hz로 조정
                ax2.set_title('그래프 2: 주파수 성분 분석 (진동수 vs 진폭)', fontsize=12)
                ax2.set_xlabel('진동수 (Hz)')
                ax2.set_ylabel('진폭 (Amplitude)')
                
                ax2.axvline(x=261.63, color='green', linestyle=':', label='가온 도 (C4)')
                if not np.isnan(avg_f0):
                    ax2.axvline(x=avg_f0, color='red', linestyle='--', label=f'내 목소리 피크: {avg_f0:.1f}Hz')
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                with st.spinner('🤖 AI 리포트 생성 중...'):
                    model = genai.GenerativeModel("gemini-3.1-flash-lite")
                    # 중복 방지를 위한 프롬프트 강화
                    prompt = f"""
                    분석 데이터: {avg_f0:.2f}Hz, {gender_type}, {range_type}.
                    다음 형식을 엄격히 지켜서 매우 짧게 리포트를 써줘:
                    1. 물리적 판정: (261.63Hz와 비교하여 설명)
                    2. 닮은 동물: (단어 1개)
                    3. 추천 가수: (계속 같은 사람 말하지 말)
                    """
                    response = model.generate_content(prompt)
                    st.info("📑 AI 매칭 리포트")
                    st.success(response.text)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

# --- [탭 2: 주파수 맞추기 게임] ---
with tab2:
    if 'target_hz' not in st.session_state:
        st.session_state.target_hz = round(random.uniform(150.0, 310.0), 1)

    st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>🎯 타겟: {st.session_state.target_hz} Hz</h1>", unsafe_allow_html=True)
    
    c_btn1, c_btn2 = st.columns([3, 1])
    with c_btn1: st.write("미션: 목소리를 조절해서 위 주파수를 맞춰보세요! (오차범위 ±20Hz)")
    with c_btn2: 
        if st.button("🔄 타겟 바꾸기"):
            st.session_state.target_hz = round(random.uniform(150.0, 310.0), 1); st.rerun()

    game_audio = st.audio_input("타겟 주파수에 맞춰 소리를 내주세요", key="input_game")

    if game_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(game_audio.getvalue())
            game_path = tmp_file.name

        try:
            y, sr = librosa.load(game_path, sr=16000, duration=3)
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
            avg_f0 = np.nanmean(f0)

            if not np.isnan(avg_f0):
                diff = abs(avg_f0 - st.session_state.target_hz)
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("내 기록", f"{avg_f0:.1f} Hz", f"{avg_f0 - st.session_state.target_hz:.1f} Hz")
                with col_res2:
                    # 오차범위 20Hz 적용
                    if diff <= 20:
                        st.balloons()
                        st.success("🎉 대성공! 완벽한 음감입니다!")
                    else:
                        st.error(f"실패! 오차가 {diff:.1f}Hz입니다. 다시 도전해보세요.")

                # 게임용 그래프 (이상하지 않게 가독성 중점)
                fig_game, ax_g = plt.subplots(figsize=(10, 3))
                D = np.abs(librosa.stft(y, n_fft=1024))
                ax_g.plot(librosa.fft_frequencies(sr=sr, n_fft=1024), np.mean(D, axis=1), color='#1F3A5A')
                ax_g.axvline(x=st.session_state.target_hz, color='orange', linewidth=3, label='TARGET')
                ax_g.axvline(x=avg_f0, color='red', linestyle='--', linewidth=2, label='YOU')
                ax_g.set_xlim(0, 500); ax_g.legend(); ax_g.set_title("주파수 매칭 분석")
                st.pyplot(fig_game)
            else:
                st.warning("소리가 너무 작습니다.")
        finally:
            if os.path.exists(game_path): os.remove(game_path)
