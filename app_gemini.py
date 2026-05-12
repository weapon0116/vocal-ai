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
    # 리눅스 서버(Streamlit Cloud) 나눔폰트 경로
    linux_font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(linux_font_path):
        font_prop = fm.FontProperties(fname=linux_font_path)
        plt.rc('font', family=font_prop.get_name())
    else:
        # 로컬 환경 (Windows/Mac)
        if platform.system() == 'Windows':
            plt.rc('font', family='Malgun Gothic')
        elif platform.system() == 'Darwin':
            plt.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# --- [2. 보안 설정: API 키] ---
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("⚠️ 시크릿 설정(Secrets)에서 GOOGLE_API_KEY를 확인해주세요.")

# --- [3. 핵심 분석 함수] ---
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

# --- [4. 전역 레이아웃 설정 (CSS)] ---
st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }
        .report-box { 
            background-color: #1E1E1E; 
            padding: 10px; 
            border-radius: 8px; 
            border: 1px solid #333; 
            margin-bottom: 10px; 
            text-align: center;
        }
        h1 { margin-top: -30px; }
    </style>
""", unsafe_allow_html=True)

# --- [5. 메인 화면 구성] ---
st.title("🎼 너의 목소리가 보여")

tab1, tab2 = st.tabs(["🔍 정밀 분석 모드", "🎯 주파수 맞추기 게임"])

# --- [탭 1: 정밀 분석 모드] ---
with tab1:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.write("주파수 데이터를 기반으로 당신의 보컬 성별과 음역대를 정밀 진단합니다.")
    with c2:
        if st.button("🎹 C4 기준음 듣기", use_container_width=True):
            audio_buffer, sr_p = play_piano_c4()
            st.audio(audio_buffer, format="audio/wav", sample_rate=sr_p, autoplay=True)

    audio_data_1 = st.audio_input("목소리를 녹음해주세요 (분석용)", key="input_analysis")

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

            # 상단 대시보드
            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1:
                st.markdown(f"<div class='report-box'><p style='margin:0; font-size:0.8rem; color:#AAA;'>판정 성별</p><h3 style='margin:0;'>{gender_type.split(' ')[0]}</h3></div>", unsafe_allow_html=True)
            with res_c2:
                st.markdown(f"<div class='report-box'><p style='margin:0; font-size:0.8rem; color:#AAA;'>음역대</p><h3 style='margin:0;'>{range_type.split(' ')[0]}</h3></div>", unsafe_allow_html=True)
            with res_c3:
                st.markdown(f"<div class='report-box'><p style='margin:0; font-size:0.8rem; color:#AAA;'>평균 주파수</p><h3 style='margin:0;'>{avg_f0:.2f} Hz</h3></div>", unsafe_allow_html=True)

            col_graph, col_ai = st.columns([1.2, 1], gap="medium")
            with col_graph:
                fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
                librosa.display.waveshow(y, sr=sr, ax=ax1, color='#0064FF', alpha=0.6)
                ax1.set_title('음성 파형', fontsize=9)
                
                n_fft = 2048
                D = np.abs(librosa.stft(y, n_fft=n_fft))
                avg_D = np.mean(D, axis=1)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
                ax2.plot(freqs, avg_D, color='#1F3A5A', linewidth=1.5)
                ax2.set_xlim(0, 1500)
                ax2.axvline(x=261.63, color='green', linestyle=':', label='C4')
                if not np.isnan(avg_f0):
                    ax2.axvline(x=avg_f0, color='red', linestyle='--', label=f'{avg_f0:.1f}Hz')
                ax2.legend(prop={'size': 7})
                ax2.set_title('주파수 성분 분석', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig1)

            with col_ai:
                with st.spinner('🤖 AI 보컬 리포트 생성 중...'):
                    model = genai.GenerativeModel("gemini-3.1-flash-lite")
                    sample_file = genai.upload_file(path=tmp_path)
                    prompt = f"분석 데이터: {avg_f0:.2f}Hz ({gender_type}). 물리적 판정 이유와 어울리는 동물/가수를 리스트 형식으로 매우 짧게 작성해줘."
                    response = model.generate_content([sample_file, prompt])
                    st.markdown("### 📑 AI 분석 리포트")
                    st.success(response.text)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

# --- [탭 2: 주파수 맞추기 게임] ---
with tab2:
    if 'target_hz' not in st.session_state:
        st.session_state.target_hz = round(random.uniform(140.0, 300.0), 2)

    st.markdown(f"<h1 style='text-align: center; color: #FF4B4B; font-size: 3.5rem; margin-bottom: 0;'>🎯 {st.session_state.target_hz} Hz</h1>", unsafe_allow_html=True)
    
    col_ctrl, col_btn = st.columns([3, 1])
    with col_ctrl:
        st.write(f"💡 **미션:** 목소리를 내어 타겟 주파수를 맞추세요! (허용 오차: ±10Hz)")
    with col_btn:
        if st.button("🔄 타겟 변경", use_container_width=True, key="reset_game"):
            st.session_state.target_hz = round(random.uniform(140.0, 300.0), 2)
            st.rerun()

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
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown(f"""
                        <div style='background-color: #1E1E1E; padding: 10px; border-radius: 8px; text-align: center; border: 1px solid #444;'>
                            <p style='margin:0; font-size: 1rem; color: #AAA;'>나의 기록</p>
                            <h2 style='margin:0; font-size: 2.2rem;'>{avg_f0:.2f} Hz</h2>
                        </div>
                    """, unsafe_allow_html=True)
                
                with res_col2:
                    if diff <= 10:
                        st.balloons()
                        st.markdown(f"<h1 style='color: #00CC66; margin-top: 10px; text-align: center;'>🎊 성공! (-{diff:.2f})</h1>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h1 style='color: #FF4B4B; margin-top: 10px; text-align: center;'>❌ 실패 (+{diff:.2f})</h1>", unsafe_allow_html=True)
                
                # 게임 모드 최적화 그래프 (높이 축소)
                fig2, ax = plt.subplots(figsize=(10, 2.8)) 
                n_fft = 2048
                D = np.abs(librosa.stft(y, n_fft=n_fft))
                avg_D = np.mean(D, axis=1)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
                
                ax.plot(freqs, avg_D, color='#1F3A5A', linewidth=2)
                ax.set_xlim(0, 600) 
                ax.axvline(x=st.session_state.target_hz, color='orange', linewidth=3, label=f'TARGET ({st.session_state.target_hz}Hz)')
                ax.axvline(x=avg_f0, color='red', linestyle='--', linewidth=2, label=f'YOU ({avg_f0:.1f}Hz)')
                ax.legend(prop={'size': 8}, loc='upper right')
                ax.tick_params(labelsize=8)
                plt.tight_layout()
                st.pyplot(fig2)
                
                acc = max(0, min(100, (1 - (diff/50)) * 100))
                st.progress(acc/100, text=f"정확도: {acc:.1f}%")

            else:
                st.warning("소리를 감지하지 못했습니다. 다시 시도해주세요.")
        finally:
            if os.path.exists(game_tmp_path): os.remove(game_tmp_path)
