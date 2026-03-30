#
# Plivo + Gemini Live Phone Call Bot
# Voice-to-voice with per-call context isolation and speech interruption
#

import os
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from google.genai import Client
from google.genai.types import ActivityEnd, ActivityStart, Blob, HttpOptions, ThinkingConfig
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiVADParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

load_dotenv(override=True)

# How much audio to pre-buffer before Silero fires (captures utterance start).
# Silero needs start_secs=0.2s of speech before triggering, so buffering
# ~300ms gives us comfortable coverage for the beginning of each utterance.
_PRE_BUFFER_MS = 300


# ── Shared Google AI client ────────────────────────────────────────────────────

_shared_gemini_client: Optional[Client] = None


def get_shared_client() -> Client:
    global _shared_gemini_client
    if _shared_gemini_client is None:
        _shared_gemini_client = Client(
            api_key=os.getenv("GOOGLE_API_KEY"),
            http_options=HttpOptions(api_version="v1beta"),
        )
        logger.info("Gemini API client created (shared across calls)")
    return _shared_gemini_client


async def prewarm_gemini():
    """Pre-warm DNS + TLS and initialise the shared Gemini client at startup."""
    import aiohttp

    get_shared_client()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://generativelanguage.googleapis.com/",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                logger.info(
                    f"Google API endpoint reachable (HTTP {resp.status}) — DNS + TLS pre-warmed"
                )
    except Exception as e:
        logger.warning(f"Gemini pre-warm ping failed (non-fatal): {e}")

    logger.info("Gemini pre-warm complete — server ready to handle calls")


# ── Service subclass: shared client + telephony turn management ───────────────

class _PhoneBotGeminiService(GeminiLiveLLMService):
    """GeminiLiveLLMService customised for telephony.

    Key overrides
    -------------
    1. ``create_client`` — injects the process-wide shared Client.

    2. ``_handle_session_ready`` — sends the initial greeting as soon as the
       Gemini session is established, avoiding the race condition where
       on_client_connected fires before the session is ready.

    3. ``_send_user_audio`` — implements pre-buffered, gated audio delivery:
       - While the user is NOT speaking: accumulate a rolling 300 ms ring
         buffer but DO NOT forward audio to Gemini (prevents noise
         accumulation that causes growing TTFB per turn).
       - When the user starts speaking: flush the pre-buffer first (so Gemini
         hears the beginning of the utterance that Silero's 200 ms onset
         window would otherwise miss), then forward all subsequent frames.
       - When the user stops speaking: stop forwarding until next turn.

    4. ``_handle_user_started_speaking`` / ``_handle_user_stopped_speaking`` —
       send ``activity_start`` / ``activity_end`` via ``send_realtime_input``
       (the correct API when server-side VAD is disabled; ``send_client_content``
       cannot be mixed with the realtime-input audio stream).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_sending_audio = False
        self._pre_buffer: list = []          # [(audio_bytes, sample_rate), …]
        self._pre_buffer_bytes = 0
        self._pre_buffer_max_bytes = 0       # computed on first audio frame

    def create_client(self):
        self._client = get_shared_client()

    async def _handle_session_ready(self, session):
        """Trigger greeting once Gemini session is live."""
        await super()._handle_session_ready(session)
        if self._session and not self._disconnecting:
            try:
                from google.genai.types import Content, Part
                greeting_turn = Content(
                    role="user",
                    parts=[Part(text="Greet the caller warmly and ask how you can help them today.")],
                )
                await self._session.send_client_content(turns=[greeting_turn], turn_complete=True)
                logger.debug("Sent initial greeting to Gemini")
            except Exception as e:
                logger.warning(f"Initial greeting send failed: {e}")

    async def _send_user_audio(self, frame):
        """Pre-buffer audio; only send to Gemini during active speech windows."""
        if self._disconnecting or not self._session:
            return

        # Compute ring-buffer size on the first frame we see.
        if not self._pre_buffer_max_bytes:
            bytes_per_ms = (frame.sample_rate * frame.num_channels * 2) // 1000
            self._pre_buffer_max_bytes = bytes_per_ms * _PRE_BUFFER_MS

        if not self._is_sending_audio:
            # Maintain a rolling pre-buffer; discard oldest frames when full.
            self._pre_buffer.append((bytes(frame.audio), frame.sample_rate))
            self._pre_buffer_bytes += len(frame.audio)
            while self._pre_buffer_bytes > self._pre_buffer_max_bytes and self._pre_buffer:
                removed, _ = self._pre_buffer.pop(0)
                self._pre_buffer_bytes -= len(removed)
            return

        # Forwarding mode — send frame to Gemini.
        try:
            await self._session.send_realtime_input(
                audio=Blob(data=frame.audio, mime_type=f"audio/pcm;rate={frame.sample_rate}")
            )
        except Exception as e:
            await self._handle_send_error(e)

    async def _handle_user_started_speaking(self, frame):
        self._user_is_speaking = True
        # Snapshot and clear pre-buffer BEFORE any awaits so that audio frames
        # arriving while we yield to the event loop go directly to Gemini
        # (not into a pre-buffer that we're about to discard).
        pre_buf = list(self._pre_buffer)
        self._pre_buffer = []
        self._pre_buffer_bytes = 0
        self._is_sending_audio = True  # gate open — live frames go straight to Gemini
        if self._session and not self._disconnecting:
            try:
                await self._session.send_realtime_input(activity_start=ActivityStart())
                for audio_bytes, sample_rate in pre_buf:
                    await self._session.send_realtime_input(
                        audio=Blob(data=audio_bytes, mime_type=f"audio/pcm;rate={sample_rate}")
                    )
                logger.debug(f"activity_start + flushed {len(pre_buf)} pre-buffer frames")
            except Exception as e:
                logger.warning(f"activity_start/pre-buffer flush failed: {e}")

    async def _handle_user_stopped_speaking(self, frame):
        self._user_is_speaking = False
        self._user_audio_buffer = bytearray()
        # Close the gate BEFORE the await so no stray audio frames slip through.
        self._is_sending_audio = False
        self._pre_buffer = []
        self._pre_buffer_bytes = 0
        await self.start_ttfb_metrics()
        if self._session and not self._disconnecting:
            try:
                await self._session.send_realtime_input(activity_end=ActivityEnd())
                logger.debug("Sent activity_end to Gemini (user stopped speaking via Silero)")
            except Exception as e:
                logger.warning(f"activity_end send failed: {e}")


# ── Turn signal converter ──────────────────────────────────────────────────────

class _TurnSignalConverter(FrameProcessor):
    """Converts Silero VAD frames to the speaking frames GeminiLive understands.

    ``VADProcessor`` emits ``VADUserStartedSpeakingFrame`` /
    ``VADUserStoppedSpeakingFrame``.  GeminiLive's ``process_frame`` handles
    ``UserStartedSpeakingFrame`` / ``UserStoppedSpeakingFrame`` (no "VAD"
    prefix) — those are the ones that update its internal speaking state and
    trigger the activity handlers.  This converter bridges the gap.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.push_frame(UserStartedSpeakingFrame(), direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self.push_frame(UserStoppedSpeakingFrame(), direction)

        await self.push_frame(frame, direction)


# ── Early Interruption Processor ──────────────────────────────────────────────

class EarlyInterruptor(FrameProcessor):
    """Clears Plivo's audio buffer the moment caller speech is detected —
    but ONLY while the bot is actively speaking.

    Tuning via .env
    ---------------
    INTERRUPTION_ENERGY_THRESHOLD  RMS threshold (default 600).
        Phone speech is typically 500–3000 RMS; line noise is below 300 RMS.
    """

    def __init__(
        self,
        stream_id: str,
        energy_threshold: int = 600,
        hold_frames: int = 3,
        cooldown_frames: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._stream_id = stream_id
        self._threshold = energy_threshold
        self._hold = hold_frames
        self._cooldown = cooldown_frames
        self._bot_speaking = False
        self._high_energy_count = 0
        self._cooldown_count = 0
        self._fired = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.UPSTREAM:
            if isinstance(frame, BotStartedSpeakingFrame):
                self._bot_speaking = True
                self._fired = False
                self._high_energy_count = 0
                self._cooldown_count = 0
                logger.debug("EarlyInterruptor: armed (bot started speaking)")
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._bot_speaking = False
                self._fired = False
                logger.debug("EarlyInterruptor: disarmed (bot stopped speaking)")

        elif isinstance(frame, InputAudioRawFrame) and self._bot_speaking:
            audio = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0

            if rms > self._threshold:
                self._high_energy_count += 1
                self._cooldown_count = 0
                if self._high_energy_count >= self._hold and not self._fired:
                    self._fired = True
                    logger.debug(
                        f"EarlyInterruptor: caller speech (rms={rms:.0f}) "
                        f"during bot speech → clearAudio"
                    )
                    await self.push_frame(
                        OutputTransportMessageUrgentFrame(
                            message={"event": "clearAudio", "streamId": self._stream_id}
                        ),
                        FrameDirection.DOWNSTREAM,
                    )
            else:
                self._high_energy_count = 0
                self._cooldown_count += 1
                if self._cooldown_count >= self._cooldown:
                    self._fired = False

        await self.push_frame(frame, direction)


# ── Bot entry point ────────────────────────────────────────────────────────────

async def run_bot(
    websocket,
    stream_id: str,
    call_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
):
    """Run the Gemini Live bot for a single Plivo call."""

    effective_prompt = system_prompt or os.getenv(
        "SYSTEM_PROMPT",
        (
            "You are a friendly and helpful phone assistant. "
            "Keep your answers concise — you are speaking on a phone call. "
            "Avoid special characters, bullet points, or markdown formatting. "
            "Respond naturally as if in a real conversation."
        ),
    )

    logger.info(f"Starting bot | stream_id={stream_id} call_id={call_id}")

    # ── Serializer ────────────────────────────────────────────────────────────
    serializer = PlivoFrameSerializer(
        stream_id=stream_id,
        call_id=call_id,
        auth_id=os.getenv("PLIVO_AUTH_ID"),
        auth_token=os.getenv("PLIVO_AUTH_TOKEN"),
        params=PlivoFrameSerializer.InputParams(auto_hang_up=True),
    )

    # ── Transport ─────────────────────────────────────────────────────────────
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=serializer,
        ),
    )

    # ── Silero VAD — detects speech/silence on phone audio (reliable at 8→16kHz)
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(
            stop_secs=0.8,    # 800ms silence → end of turn (merges rapid speech bursts)
            start_secs=0.2,   # 200ms of speech → confirmed speech start
            min_volume=0.0,   # disable EBU R128 volume gate — phone audio at 8kHz
                              # measures very low on that scale; rely on Silero only
        ))
    )

    # ── VAD frame converter ───────────────────────────────────────────────────
    turn_converter = _TurnSignalConverter()

    # ── Early interruptor ─────────────────────────────────────────────────────
    early_interruptor = EarlyInterruptor(
        stream_id=stream_id,
        energy_threshold=int(os.getenv("INTERRUPTION_ENERGY_THRESHOLD", "600")),
    )

    # ── Gemini Live ───────────────────────────────────────────────────────────
    llm = _PhoneBotGeminiService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        settings=GeminiLiveLLMService.Settings(
            system_instruction=effective_prompt,
            voice="Puck",  # Aoede | Charon | Fenrir | Kore | Puck
            vad=GeminiVADParams(
                disabled=True,  # Silero owns all VAD
            ),
            thinking=ThinkingConfig(thinking_budget=0),  # disable reasoning → low latency
        ),
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            turn_converter,
            early_interruptor,
            llm,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ── Events ────────────────────────────────────────────────────────────────

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected | stream_id={stream_id}")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected | stream_id={stream_id}")
        await task.cancel()

    @transport.event_handler("on_session_timeout")
    async def on_session_timeout(transport, client):
        logger.warning(f"Session timeout | stream_id={stream_id}")
        await task.cancel()

    # ── Run ───────────────────────────────────────────────────────────────────
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

    logger.info(f"Bot finished | stream_id={stream_id}")