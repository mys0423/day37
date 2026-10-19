import asyncio
import hashlib
import io
import json
import os
import struct
import tempfile
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # .env 로딩

# 입력 파일 최대 크기를 10Mbytes로 제한
MAX_BYTES = 10 * 1024 * 1024

# 실행이력을 저장할 jsonl파일 경로
LOG_PATH = Path(__file__).with_name('voice_runs_campus.jsonl')

# 시설 운영 정보 딕셔너리
FACILITIES = {
    "도서관": "도서관은 본관 옆 건물이며 평일 9시부터 18시까지 운영합니다.",
    "학생식당": "학생식당은 학생회관 1층에 있으며 11시부터 14시까지 운영합니다.",
    "취업지원센터": "취업지원센터는 학생회관 2층이며 취업 상담을 제공합니다.",
    "행정실": "행정실은 본관 1층이며 각종 증명서 발급을 안내합니다.",
}

# 실제 WAV 파일인지 확인하고 기본 정보를 추출하는 함수
def validate_wav(audio):
    try:
        with wave.open(io.BytesIO(audio), 'rb') as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frames = wav.getnframes()
            compression = wav.getcomptype()
    except (wave.Error, EOFError, OSError) as e:
        raise ValueError('유효한 비압축 PCM WAV 파일을 입력해 주세요.') from e

    if compression != 'NONE' or channels not in (1, 2) or sample_rate < 0 or frames <= 0:
        raise ValueError('모노/스테레오 비압축 PCM WAV 파일이 필요합니다.')

    return {
        'channels': channels,
        'sample_rate': sample_rate,
        'bit_depth': sample_width * 8,
        'duration_sec': round(frames / sample_rate, 2)
    }


# faster-whisper 모델 로딩
@st.cache_resource(show_spinner='Whisper 모델을 로딩하고 있습니다...')
def load_stt_model():
    from faster_whisper import WhisperModel
    return WhisperModel('base', device='CPU', compute_type='int8')


# STT 인식 수행 (데모 모드 지원)
def transcribe_wav(audio, use_real_stt):
    model = load_stt_model()
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(audio)
        wav_path = Path(f.name)

    try:
        segments, info = model.transcribe(
            str(wav_path),
            language='ko',
            vad_filter=True
        )

        transcript = ''.join(segment.text.strip() for segment in segments).strip()

        return transcript

    finally:
        wav_path.unlink(missing_ok=True)


# 시설명 키워드를 찾아 안내하는 응답 생성 함수
def build_reply(transcript):
    clean = ''.join(transcript.split())
    if not clean:
        return '음성 인식을 하지 못했습니다. 다시 말씀해 주세요.'

    # 질문에 시설명이 포함되어 있는지 확인
    for facility, info in FACILITIES.items():
        if facility in transcript:
            return info

    return '안내 가능한 시설은 도서관, 학생식당, 취업지원센터, 행정실입니다.'


# Edge TTS 실제 합성 함수
async def generate_tts_mp3(text, voice):
    import edge_tts
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        mp3_path = Path(f.name)
    try:
        await edge_tts.Communicate(text=text, voice=voice).save(str(mp3_path))
        return mp3_path.read_bytes()
    finally:
        mp3_path.unlink(missing_ok=True)


# TTS 합성 수행 (데모 모드 지원)
def synthesize_speech(text, voice_id, use_real_tts):
    if not use_real_tts:
        time.sleep(0.5)  # 가상 처리 시간
        # 재생 가능한 1초 무음 WAV 생성 (비프음 데모 역할)
        out = io.BytesIO()
        with wave.open(out, 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(struct.pack('h', 0) * 16000)
        return out.getvalue(), 'audio/wav'

    mp3_bytes = asyncio.run(generate_tts_mp3(text, voice_id))
    return mp3_bytes, 'audio/mpeg'


# JSONL 및 DB 저장
def store_meta(meta):
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(json.dumps(meta, ensure_ascii=False) + '\n')

    database_url = os.getenv('DATABASE_URL', '').strip()
    if not database_url:
        return

    try:
        import psycopg
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                # 독립적인 실습을 위해 campus_voice_runs 테이블 사용
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS campus_voice_runs (
                        request_id TEXT PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL,
                        status TEXT NOT NULL,
                        stt_ms DOUBLE PRECISION NOT NULL,
                        reply_ms DOUBLE PRECISION NOT NULL,
                        tts_ms DOUBLE PRECISION NOT NULL,
                        total_ms DOUBLE PRECISION NOT NULL,
                        model_version TEXT NOT NULL,
                        audio_sha256 TEXT NOT NULL
                    )
                ''')
                cur.execute('''
                    INSERT INTO campus_voice_runs
                    (request_id, created_at, status, stt_ms, reply_ms, tts_ms, total_ms, model_version, audio_sha256)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    meta['request_id'], meta['created_at'], meta['status'],
                    meta['stt_ms'], meta['reply_ms'], meta['tts_ms'],
                    meta['total_ms'], meta['model_version'], meta['audio_sha256']
                ))
    except Exception as e:
        st.warning(f'DB 실행 이력 저장 실패: {type(e).__name__}: {e}')


# 파이프라인 함수
def run_pipeline(audio, use_real_stt, use_real_tts, voice_id):
    request_id = str(uuid.uuid4())
    digest = hashlib.sha256(audio).hexdigest()
    marks = {'stt_ms': 0.0, 'reply_ms': 0.0, 'tts_ms': 0.0}
    started = time.perf_counter()
    result = None
    error = None

    try:
        t0 = time.perf_counter()
        transcript = transcribe_wav(audio, use_real_stt)
        marks['stt_ms'] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        reply = build_reply(transcript)
        marks['reply_ms'] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        output, mime = synthesize_speech(reply, voice_id, use_real_tts)
        marks['tts_ms'] = (time.perf_counter() - t0) * 1000

        result = (transcript, reply, output, mime)
    except Exception as e:
        error = e

    elapsed = (time.perf_counter() - started) * 1000
    model_info = f"STT:{'real' if use_real_stt else 'demo'}/TTS:{'real' if use_real_tts else 'demo'}/{voice_id}"

    meta = {
        'request_id': request_id,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'error' if error else 'success',
        **{key: round(value, 2) for key, value in marks.items()},
        'total_ms': round(elapsed, 2),
        'model_version': model_info,
        'audio_sha256': digest
    }

    store_meta(meta)
    return result, marks, meta, error


# Streamlit UI
def main():
    st.set_page_config(page_title='캠퍼스 시설 안내', layout='centered')
    st.title('캠퍼스 시설 안내 음성 도우미')
    st.caption('마이크로 시설 운영에 대해 질문하면 AI가 안내해 드립니다.')

    with st.form('voice_form'):
        uploaded = st.audio_input('마이크로 질문을 녹음하세요 (예: 도서관 운영시간 알려줘)')

        cols = st.columns(2)
        with cols[0]:
            use_real_stt = st.toggle('실제 STT 모델 사용 (faster-whisper)', value=False)
        with cols[1]:
            use_real_tts = st.toggle('실제 TTS 모델 사용 (edge-tts)', value=False)

        voice_id = st.selectbox('TTS 목소리', ['ko-KR-SunHiNeural', 'ko-KR-InJoonNeural'])
        submitted = st.form_submit_button('안내 방송 생성')

    # 하단: 최근 이력 표시
    st.divider()
    st.subheader('최근 처리 이력')
    if LOG_PATH.exists():
        try:
            df = pd.read_json(LOG_PATH, lines=True)
            # 최근 5건 추출 및 보기 좋게 컬럼 정리
            st.dataframe(df.tail(5)[['created_at', 'status', 'total_ms', 'model_version']], use_container_width=True)
        except Exception:
            st.write('이력을 불러올 수 없습니다.')
    else:
        st.write('저장된 실행 이력이 없습니다.')

    if not submitted:
        return

    if uploaded is None:
        st.error('먼저 마이크로 음성을 녹음해 주세요.')
        return

    audio = uploaded.getvalue()

    if not audio or len(audio) > MAX_BYTES:
        st.error(f'음성 파일 크기는 {MAX_BYTES}바이트 이하여야 합니다.')
        return

    try:
        audio_info = validate_wav(audio)
    except ValueError as e:
        st.error(str(e))
        return

    with st.spinner('음성 분석 및 응답 생성 중...'):
        result, marks, meta, error = run_pipeline(audio, use_real_stt, use_real_tts, voice_id)

    if error:
        st.error(f'처리 오류: {type(error).__name__}: {error}')
    else:
        transcript, reply, output, mime = result

        st.success('처리가 완료되었습니다!')

        st.subheader('인식된 질문')
        st.info(transcript)

        st.subheader('AI 안내')
        st.write(reply)
        st.audio(output, format=mime)

        st.subheader('소요 시간 분석')
        st.metric('총 처리시간', f"{meta['total_ms']:.0f} ms")

        chart_df = pd.DataFrame([marks])
        st.bar_chart(chart_df)


if __name__ == '__main__':
    main()