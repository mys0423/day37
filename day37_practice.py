import asyncio
import hashlib
import io
import json
import math
import struct
import tempfile
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import sys

# 최신 av와 faster-whisper 사이의 metadata_errors 충돌을 원천 차단합니다.
import av
_original_open = av.open
def _patched_open(*args, **kwargs):
    kwargs.pop('metadata_errors', None)  # 에러를 유발하는 옵션만 몰래 제거
    return _original_open(*args, **kwargs)
av.open = _patched_open


# 상위 폴더 경로 추가 (include, util 모듈 인식용)
sys.path.append(str(Path(__file__).resolve().parent.parent))


ROOT = Path(__file__).resolve().parent

LOG_PATH = ROOT / "voice_runs_campus.jsonl"

# 에러 수정: 정상적인 곱셈 연산으로 변경
MAX_BYTES = 10 * 1024 * 1024

FACILITIES = {
    "도서관": "도서관은 본관 옆 건물이며 평일 9시부터 18시까지 운영합니다.",
    "학생식당": "학생식당은 학생회관 1층에 있으며 11시부터 14시까지 운영합니다.",
    "취업지원센터": "취업지원센터는 학생회관 2층이며 취업 상담을 제공합니다.",
    "행정실": "행정실은 본관 1층이며 각종 증명서 발급을 안내합니다.",
}


# 문장 길이로 길이를 정하는 시험용 WAV 생성 함수
def make_tone(text):
    rate = 16000  # 1초당 샘플 16,000개
    duration = min(1.5, max(0.3, len(text) * 0.025))
    count = int(rate * duration)
    buffer = io.BytesIO()  # 메모리에 바이너리 데이터를 기록함

    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)  # 단일 채널(모노)
        writer.setsampwidth(2)  # 샘플당 2바이트 = 16비트 PCM
        writer.setframerate(rate)  # 샘플링 레이트 설정

        # 에러 수정: 곱셈 기호(*) 복구 및 괄호 정리
        samples = (int(4500 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(count))
        writer.writeframes(b"".join(struct.pack("<h", s) for s in samples))

    return buffer.getvalue(), "audio/wav"


# 녹음된 WAV의 전사 결과를 반환하는 함수
def recognize(audio, demo_text, use_real):
    if not use_real:
        return demo_text.strip()  # 데모는 녹음 내용을 인식하지 않음
    from faster_whisper import WhisperModel
    model = load_whisper()  # 모델은 캐시하여 재다운로드를 줄임
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
        file.write(audio)
        path = Path(file.name)
    try:
        # language='ko': 한국어 인식, beam_size: 탐색 후보의 폭
        segments, _ = model.transcribe(str(path), language="ko", beam_size=3)
        return " ".join(part.text.strip() for part in segments).strip()
    finally:
        path.unlink(missing_ok=True)  # 예외 시에도 임시 WAV를 삭제


# CPU 기반 Whisper tiny 모델을 한 번 생성하고 캐싱하는 함수
@st.cache_resource(show_spinner="Whisper 모델 준비 중...")
def load_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel("tiny", device="cpu", compute_type="int8")


# 질문 내 시설 이름을 기준으로 간단한 안내문을 반환하는 함수
def answer_question(transcript):
    clean = " ".join(transcript.split())  # 여러 공백을 하나로 축소
    if not clean:
        return "음성이 인식되지 않았습니다. 다시 말씀해 주세요."
    for name, description in FACILITIES.items():
        if name in clean:
            return description  # 발견한 첫 시설의 정보를 반환
    choices = ", ".join(FACILITIES)
    return f"해당 시설 정보는 없습니다. 안내 가능한 곳은 {choices}입니다."


# 온라인 edge-tts로 문장을 합성한 MP3 바이트를 반환하는 함수
async def edge_audio(text):
    import edge_tts
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as file:
        path = Path(file.name)
    try:
        # Communicate(text, voice): TTS 합성 요청 객체 생성
        await edge_tts.Communicate(
            text=text, voice="ko-KR-SunHiNeural"
        ).save(str(path))  # MP3를 지정한 파일에 저장
        return path.read_bytes(), "audio/mpeg"
    finally:
        path.unlink(missing_ok=True)


# 실제 TTS 여부에 따라 (음성 bytes, MIME 타입)을 반환하는 함수
def speak(text, use_real):
    return asyncio.run(edge_audio(text)) if use_real else make_tone(text)


# 개인 발화와 녹음 원본을 제외한 실행 메타데이터를 JSONL로 저장하는 함수
def save_log(metadata):
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metadata, ensure_ascii=False) + "\n")


# JSONL 마지막 limit건을 DataFrame으로 변환하는 함수
def recent_logs(limit=10):
    if not LOG_PATH.exists():
        return pd.DataFrame()  # 이력 파일이 없다면 빈 데이터프레임
    with LOG_PATH.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    return pd.DataFrame(rows[-limit:][::-1])  # 가장 최근 실행부터 표시


def main():
    st.set_page_config(page_title="캠퍼스 음성 도우미", layout="centered")
    st.title("캠퍼스 시설 음성 안내")
    st.caption("시설 안내 내용은 교육용 가상 정보입니다.")

    with st.form("campus_voice"):
        audio_input = st.audio_input("시설에 관해 질문을 녹음하세요")
        demo_text = st.text_input(
            "데모 STT 예시 문장", "도서관 운영시간 알려줘"
        )
        use_stt = st.checkbox("실제 STT 사용 (faster-whisper)")
        use_tts = st.checkbox("실제 TTS 사용 (edge-tts)")
        submitted = st.form_submit_button("음성 안내 실행")

    if submitted:
        audio = audio_input.getvalue() if audio_input else b""

        if not audio:
            st.error("음성을 먼저 녹음해 주세요.")
            return

        if len(audio) > MAX_BYTES:
            st.error("입력 파일은 10MiB를 초과할 수 없습니다.")
            return

        timings = {"stt_ms": 0.0, "reply_ms": 0.0, "tts_ms": 0.0}
        req_id = str(uuid.uuid4())  # 실행을 구분할 고유 요청 ID
        digest = hashlib.sha256(audio).hexdigest()  # 원본 대신 해시 저장
        start = time.perf_counter()  # 벽시계 오차가 없는 시간 측정
        outcome = None
        error = None

        try:
            t0 = time.perf_counter()
            transcript = recognize(audio, demo_text, use_stt)
            timings["stt_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            reply = answer_question(transcript)
            timings["reply_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            sound, mime = speak(reply, use_tts)
            timings["tts_ms"] = (time.perf_counter() - t0) * 1000

            outcome = (transcript, reply, sound, mime)

        except Exception as exc:
            error = exc  # 라이브러리 미설치·네트워크 오류 등 전달

        total = (time.perf_counter() - start) * 1000

        metadata = {
            "request_id": req_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "error" if error else "success",
            "stt_model": "whisper-tiny" if use_stt else "demo-stt",
            "tts_model": "edge-tts" if use_tts else "demo-beep",
            "audio_sha256": digest,
            **{key: round(value, 2) for key, value in timings.items()},
            "total_ms": round(total, 2),
        }

        save_log(metadata)  # 성공/실패 여부와 무관하게 메타데이터 저장

        if error:
            st.error(f"실행 실패: {type(error).__name__}: {error}")
        else:
            transcript, reply, sound, mime = outcome
            st.write("**STT 인식 문장:**", transcript)
            st.write("**안내 답변:**", reply)
            st.audio(sound, format=mime)  # 브라우저에서 음성 재생

        st.metric("총 처리시간", f"{total:.0f} ms")
        st.bar_chart(pd.Series(timings))  # STT/응답/TTS 단계별 시간
        st.json(metadata)  # 개인정보 없는 실행 메타데이터
        print("음성 처리 완료", req_id, metadata["status"])

    st.subheader("최근 처리 이력")
    logs = recent_logs(10)
    if logs.empty:
        st.info("아직 실행 이력이 없습니다.")
    else:
        st.dataframe(logs, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()