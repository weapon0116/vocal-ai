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
    gender = "남성형" if avg_f0 < C4_HZ else "여성형"
    if avg_f0 >= 261.63:
        vocal_range = "소프라노" if avg_f0 >= 440 else "알토"
    else:
        vocal_range = "테너" if avg_f0 >= 130 else "베이스"
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
            padding: 12px; 
            border-radius: 10px; 
            border: 1px solid #333; 
            margin-bottom: 5px; 
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        h1 { margin-top: -45px; margin-bottom: 10px; }
        .stAudioInput { margin-bottom: -15px; }
        .success-text { color: #00CC66; font-weight: bold; }
        .fail-text { color: #FF4B4B; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- [5. 메인 화면 구성] ---
st.title("🎼 너의 목소리가 보여")

tab1, tab2 = st.tabs(["🔍 정밀 분석 모드", "🎯 주파수 맞추기 게임"])

# --- [탭 1: 정밀 분석 모드] ---
with tab1:
    c1, c2 = st.columns([3, 1])
    with c1: st.caption("당신의 보컬 데이터를 AI가 정밀 분석합니다.")
    with c2:
        if st.button("🎹 C4 기준음", use_container_width=True):
            audio_buffer, sr_p = play_piano_c4()
            st.audio(audio_buffer, format="audio/wav", sample_rate=sr_p, autoplay=True)

    audio_data_1 = st.audio_input("목소리를 녹음해주세요", key="input_analysis")

    if audio_data_1:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data_1.getvalue())
            tmp_path = tmp_file.name

        try:
            with st.spinner('🔍 정밀 분석 중... (잠시만 기다려주세요)'):
                y, sr = librosa.load(tmp_path, sr=16000)
                y, _ = librosa.effects.trim(y)
                f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
                avg_f0 = np.nanmean(f0)
                gender_type, range_type = analyze_gender_by_c4(avg_f0)

            # 1. 상단 요약 (정렬 및 음역대 수정)
            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1: st.markdown(f"<div class='report-box'><p style='margin:0; font-size:0.75rem; color:#AAA;'>성별</p><h3 style='margin:0;'>{gender_type}</h3></div>", unsafe_allow_html=True)
            with res_c2: st.markdown(f"<div class='report-box'><p style='margin:0; font-size:0.75rem; color:#AAA;'>음역대</p><h3 style='margin:0;'>{range_type}</h3></div>", unsafe_allow_html=True)
            with res_c3: st.markdown(f"<div class='report-box'><p style='margin:0; font-size:0.75rem; color:#AAA;'>평균 주파수</p><h3 style='margin:0;'>{avg_f0:.1f} Hz</h3></div>", unsafe_allow_html=True)

            # 2. 하단 그래프 & AI 리포트
            col_graph, col_ai = st.columns([1.1, 0.9], gap="medium")
            with col_graph:
                fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3.8), gridspec_kw={'height_ratios': [1, 1.2]})
                librosa.display.waveshow(y, sr=sr, ax=ax1, color='#0064FF', alpha=0.5)
                ax1.set_title('음성 파형', fontsize=8); ax1.tick_params(labelsize=6)
                
                n_fft = 2048; D = np.abs(librosa.stft(y, n_fft=n_fft)); avg_D = np.mean(D, axis=1)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
                ax2.plot(freqs, avg_D, color='#1F3A5A', linewidth=1); ax2.set_xlim(0, 1000)
                if not np.isnan(avg_f0): ax2.axvline(x=avg_f0, color='red', linestyle='--', linewidth=1)
                ax2.set_title('주파수 분석', fontsize=8); ax2.tick_params(labelsize=6)
                plt.tight_layout(pad=0.5); st.pyplot(fig1)

            with col_ai:
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                st.markdown("<h4 style='margin-bottom: 5px; border-left: 5px solid #00CC66; padding-left: 10px;'>📊 Gemini AI 보컬 리포트</h4>", unsafe_allow_html=True)
                with st.spinner('🤖 AI가 분석 내용을 작성하고 있습니다...'):
                    model = genai.GenerativeModel("gemini-3.1-flash-lite")
                    sample_file = genai.upload_file(path=tmp_path)
                    prompt = f"데이터: {avg_f0:.1f}Hz, {gender_type}. 판정 이유와 어울리는 동물/가수를 3줄 이내 리스트로 짧고 강렬하게 작성해줘."
                    response = model.generate_content([sample_file, prompt])
                    st.success(response.text)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

# --- [탭 2: 주파수 맞추기 게임] ---
with tab2:
    if 'target_hz' not in st.session_state:
        st.session_state.target_hz = round(random.uniform(140.0, 300.0), 2)

    st.markdown(f"<h1 style='text-align: center; color: #FF4B4B; font-size: 3rem; margin-bottom: 0;'>🎯 {st.session_state.target_hz} Hz</h1>", unsafe_allow_html=True)
    
    col_ctrl, col_btn = st.columns([3, 1])
    with col_ctrl: st.caption("💡 목소리로 타겟 주파수를 맞추세요! (계산 속도 최적화 완료)")
    with col_btn:
        if st.button("🔄 타겟 변경", use_container_width=True):
            st.session_state.target_hz = round(random.uniform(140.0, 300.0), 2); st.rerun()

    audio_data_2 = st.audio_input("마이크에 소리를 내주세요.", key="input_game")

    if audio_data_2:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data_2.getvalue()); game_tmp_path = tmp_file.name

        try:
            # [속도 최적화] 게임용은 빠른 yin 알고리즘 사용
            y, sr = librosa.load(game_tmp_path, sr=16000)
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
            avg_f0 = np.nanmean(f0)

            if not np.isnan(avg_f0):
                diff = abs(avg_f0 - st.session_state.target_hz)
                
                # 상단 결과창
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown(f"<div class='report-box'><p style='margin:0; font-size: 0.8rem; color: #AAA;'>나의 기록</p><h2 style='margin:0; font-size: 2.2rem;'>{avg_f0:.1f} Hz</h2></div>", unsafe_allow_html=True)
                with res_col2:
                    if diff <= 10:
                        st.balloons(); st.markdown(f"<div class='report-box' style='border-color: #00CC66;'><h2 class='success-text' style='margin:0;'>🎊 성공!</h2></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='report-box' style='border-color: #FF4B4B;'><h2 class='fail-text' style='margin:0;'>❌ 실패</h2></div>", unsafe_allow_html=True)
                
                # 그래프 좌측 압축 + 정보 우측 배치
                graph_col, info_col = st.columns([1.6, 1], gap="small")
                
                with graph_col:
                    # 그래프 높이 더 축소 및 눈금 정리
                    fig2, ax = plt.subplots(figsize=(6, 1.4)) 
                    D = np.abs(librosa.stft(y, n_fft=1024)); avg_D = np.mean(D, axis=1) # n_fft 축소로 속도 향상
                    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
                    ax.plot(freqs, avg_D, color='#1F3A5A', linewidth=1.2); ax.set_xlim(0, 600) 
                    ax.axvline(x=st.session_state.target_hz, color='orange', linewidth=2.5, label='TARGET')
                    ax.axvline(x=avg_f0, color='red', linestyle='--', linewidth=1.5, label='YOU')
                    ax.legend(prop={'size': 6}, loc='upper right'); ax.tick_params(labelsize=5)
                    plt.tight_layout(pad=0.1); st.pyplot(fig2)
                
                with info_col:
                    acc = max(0, min(100, (1 - (diff/50)) * 100))
                    st.markdown(f"""
                        <div style='margin-top: 5px; padding: 12px; background: #262730; border-radius: 8px; border: 1px solid #444;'>
                            <p style='margin:0; font-size: 0.9rem; font-weight: bold;'>🎯 실시간 스코어</p>
                            <hr style='margin: 8px 0; border: 0.5px solid #555;'>
                            <p style='margin:0; font-size: 0.85rem;'>정확도: <span class='success-text'>{acc:.1f}%</span></p>
                            <p style='margin:0; font-size: 0.85rem;'>오차범위: <span class='fail-text'>{diff:.1f} Hz</span></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(acc/100)
            else: st.warning("소리가 감지되지 않았습니다. 조금 더 크게 소리를 내주세요!")
        finally:
            if os.path.exists(game_tmp_path): os.remove(game_tmp_path)
