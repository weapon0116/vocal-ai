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
st.set_page_config(page_title="vocal_ai", layout="wide")

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

# --- [4. 메인 화면 구성] ---
st.title("🎼 너의 목소리가 보여")

# 상단 탭으로 모드 분리
tab1, tab2 = st.tabs(["🔍 정밀 분석 모드", "🎁 주파수 맞추기 게임"])

# --- [탭 1: 정밀 분석 모드 (기존 기능)] ---
with tab1:
    st.header("정밀 분석 및 AI 리포트")
    if st.button("🎹 기준점: 가온 도(C4) 듣기"):
        audio_buffer, sr_p = play_piano_c4()
        st.audio(audio_buffer, format="audio/wav", sample_rate=sr_p, autoplay=True)

    st.divider()
    audio_data_1 = st.audio_input("마이크에 소리를 내주세요. (분석용)", key="input_analysis")

    if audio_data_1:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data_1.getvalue())
            tmp_path = tmp_file.name

        try:
            with st.spinner('🔍 주파수 분석 중...'):
                y, sr = librosa.load(tmp_path, sr=16000)
                y, _ = librosa.effects.trim(y)
                f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
                avg_f0 = np.nanmean(f0)
                gender_type, range_type = analyze_gender_by_c4(avg_f0)

            st.subheader(f"📊 분석 결과: {gender_type} / {range_type}")
            st.write(f"평균 주파수: **{avg_f0:.2f} Hz**")
            
            col1, col2 = st.columns([1.2, 0.8], gap="medium")
            with col1:
                fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                librosa.display.waveshow(y, sr=sr, ax=ax1, color='#0064FF', alpha=0.8)
                ax1.set_title('음성 에너지 파형')
                
                n_fft = 2048
                D = np.abs(librosa.stft(y, n_fft=n_fft))
                avg_D = np.mean(D, axis=1)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
                ax2.plot(freqs, avg_D, color='#1F3A5A', linewidth=2)
                ax2.set_xlim(0, 2000)
                ax2.axvline(x=261.63, color='green', linestyle=':', label='가온 도 (C4)')
                if not np.isnan(avg_f0):
                    ax2.axvline(x=avg_f0, color='red', linestyle='--', label=f'내 목소리: {avg_f0:.1f}Hz')
                ax2.legend()
                st.pyplot(fig1)

            with col2:
                with st.spinner('🤖 Gemini AI 보컬 리포트 생성 중...'):
                    model = genai.GenerativeModel("gemini-3.1-flash-lite")
                    sample_file = genai.upload_file(path=tmp_path)
                    prompt = f"분석 데이터: {avg_f0:.2f}Hz ({gender_type}). 물리적 판정 이유와 어울리는 동물/가수를 짧고 굵게 리포트해줘."
                    response = model.generate_content([sample_file, prompt])
                    st.info("📑 Gemini AI 매칭 리포트")
                    st.success(response.text)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

# --- [탭 2: 주파수 맞추기 게임 (이벤트용)] ---
with tab2:
    st.header("🎯 타겟 주파수를 맞춰라!")
    st.write("랜덤으로 생성된 주파수와 내 목소리를 일치시켜보세요! (오차 ±5Hz 이내 성공)")

    # 세션 상태 초기화
    if 'target_hz' not in st.session_state:
        st.session_state.target_hz = round(random.uniform(120.0, 350.0), 2)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("오늘의 타겟", f"{st.session_state.target_hz} Hz")
    with c2:
        if st.button("🔄 타겟 변경하기"):
            st.session_state.target_hz = round(random.uniform(120.0, 350.0), 2)
            st.rerun()

    st.divider()
    audio_data_2 = st.audio_input("마이크에 소리를 내주세요. (게임용)", key="input_game")

    if audio_data_2:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data_2.getvalue())
            game_tmp_path = tmp_file.name

        try:
            y, sr = librosa.load(game_tmp_path, sr=16000)
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
            avg_f0 = np.nanmean(f0)

            if not np.isnan(avg_f0):
                diff = abs(avg_f0 - st.session_state.target_hz)
                st.subheader(f"🎤 당신의 기록: {avg_f0:.2f} Hz")
                
                if diff <= 5:
                    st.balloons()
                    st.success(f"🎊 성공!! (차이: {diff:.2f}Hz) 경품을 수령하세요!")
                else:
                    st.error(f"아쉽습니다! (차이: {diff:.2f}Hz) 조금만 더 노력해보세요!")
                
                # 게임용 심플 그래프
                fig2, ax = plt.subplots(figsize=(10, 4))
                n_fft = 2048
                D = np.abs(librosa.stft(y, n_fft=n_fft))
                avg_D = np.mean(D, axis=1)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
                ax.plot(freqs, avg_D, color='#1F3A5A')
                ax.set_xlim(0, 1000)
                ax.axvline(x=st.session_state.target_hz, color='orange', label=f'타겟: {st.session_state.target_hz}Hz')
                ax.axvline(x=avg_f0, color='red', linestyle='--', label=f'내 목소리: {avg_f0:.1f}Hz')
                ax.legend()
                st.pyplot(fig2)
            else:
                st.warning("소리가 너무 작거나 감지되지 않았습니다.")
        finally:
            if os.path.exists(game_tmp_path): os.remove(game_tmp_path)
