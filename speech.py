"""
AutoType v1.02 — Speech Engine
STT: faster-whisper (persistent model, configurable unload timer) — currently unused by start.py
TTS: Kokoro-82M (https://huggingface.co/hexgrad/Kokoro-82M)

Requirements:
  pip install kokoro soundfile numpy
  pip install faster-whisper                  # only if STT routes are re-enabled
"""
import os, json, time, threading, tempfile, io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "speech")
os.makedirs(DATA_DIR, exist_ok=True)
CONFIG_PATH = os.path.join(DATA_DIR, "speech_config.json")

# ══════════════════════════════════════════════════════════════════
# STT — faster-whisper (persistent model)  [legacy, kept for re-use]
# ══════════════════════════════════════════════════════════════════
_whisper_model = None
_whisper_lock = threading.Lock()
_whisper_last_use = 0.0
_unload_timer = None

def _get_whisper(model_size="base", model_path="", device="auto", compute_type="auto"):
    global _whisper_model, _whisper_last_use
    with _whisper_lock:
        if _whisper_model is not None:
            _whisper_last_use = time.time()
            return _whisper_model
        from faster_whisper import WhisperModel
        path = model_path.strip() if model_path.strip() else model_size
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        _whisper_model = WhisperModel(path, device=device, compute_type=compute_type)
        _whisper_last_use = time.time()
        return _whisper_model

def unload_whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            del _whisper_model
            _whisper_model = None
            return True
    return False

def transcribe_audio(audio_bytes, content_type, config):
    global _whisper_last_use
    model = _get_whisper(
        model_size=config.get("whisper_model", "base"),
        model_path=config.get("whisper_model_path", ""),
        device=config.get("whisper_device", "auto"),
        compute_type=config.get("whisper_compute", "auto")
    )
    ct = (content_type or "").lower()
    if "wav" in ct: ext = ".wav"
    elif "ogg" in ct: ext = ".ogg"
    elif "mp4" in ct or "m4a" in ct: ext = ".m4a"
    else: ext = ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=DATA_DIR)
    try:
        tmp.write(audio_bytes); tmp.close()
        segments, info = model.transcribe(
            tmp.name,
            language=config.get("language", None) or None,
            beam_size=config.get("beam_size", 5),
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=600, speech_pad_ms=300)
        )
        text = " ".join([seg.text.strip() for seg in segments])
        _whisper_last_use = time.time()
        return text.strip()
    finally:
        try: os.unlink(tmp.name)
        except: pass

# ══════════════════════════════════════════════════════════════════
# TTS — Kokoro-82M
# ══════════════════════════════════════════════════════════════════
# Voice catalogue from
# https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
# Each entry: id, label, lang_code ('a'=American English, 'b'=British English)
KOKORO_VOICES = [
    # American English — Female
    {"id": "af_heart",    "name": "Heart (American F)",    "lang": "a"},
    {"id": "af_alloy",    "name": "Alloy (American F)",    "lang": "a"},
    {"id": "af_aoede",    "name": "Aoede (American F)",    "lang": "a"},
    {"id": "af_bella",    "name": "Bella (American F)",    "lang": "a"},
    {"id": "af_jessica",  "name": "Jessica (American F)",  "lang": "a"},
    {"id": "af_kore",     "name": "Kore (American F)",     "lang": "a"},
    {"id": "af_nicole",   "name": "Nicole (American F)",   "lang": "a"},
    {"id": "af_nova",     "name": "Nova (American F)",     "lang": "a"},
    {"id": "af_river",    "name": "River (American F)",    "lang": "a"},
    {"id": "af_sarah",    "name": "Sarah (American F)",    "lang": "a"},
    {"id": "af_sky",      "name": "Sky (American F)",      "lang": "a"},
    # American English — Male
    {"id": "am_adam",     "name": "Adam (American M)",     "lang": "a"},
    {"id": "am_echo",     "name": "Echo (American M)",     "lang": "a"},
    {"id": "am_eric",     "name": "Eric (American M)",     "lang": "a"},
    {"id": "am_fenrir",   "name": "Fenrir (American M)",   "lang": "a"},
    {"id": "am_liam",     "name": "Liam (American M)",     "lang": "a"},
    {"id": "am_michael",  "name": "Michael (American M)",  "lang": "a"},
    {"id": "am_onyx",     "name": "Onyx (American M)",     "lang": "a"},
    {"id": "am_puck",     "name": "Puck (American M)",     "lang": "a"},
    {"id": "am_santa",    "name": "Santa (American M)",    "lang": "a"},
    # British English — Female
    {"id": "bf_alice",    "name": "Alice (British F)",     "lang": "b"},
    {"id": "bf_emma",     "name": "Emma (British F)",      "lang": "b"},
    {"id": "bf_isabella", "name": "Isabella (British F)",  "lang": "b"},
    {"id": "bf_lily",     "name": "Lily (British F)",      "lang": "b"},
    # British English — Male
    {"id": "bm_daniel",   "name": "Daniel (British M)",    "lang": "b"},
    {"id": "bm_fable",    "name": "Fable (British M)",     "lang": "b"},
    {"id": "bm_george",   "name": "George (British M)",    "lang": "b"},
    {"id": "bm_lewis",    "name": "Lewis (British M)",     "lang": "b"},
]
_VOICE_LANG = {v["id"]: v["lang"] for v in KOKORO_VOICES}

_kokoro_pipelines = {}     # lang_code -> KPipeline instance
_kokoro_lock = threading.Lock()

def get_tts_voices():
    """Return the Kokoro voice catalogue."""
    return list(KOKORO_VOICES)

def _get_kokoro_pipeline(lang_code):
    """Lazy-load a KPipeline per language code; cached across calls."""
    with _kokoro_lock:
        p = _kokoro_pipelines.get(lang_code)
        if p is not None:
            return p
        from kokoro import KPipeline
        p = KPipeline(lang_code=lang_code)
        _kokoro_pipelines[lang_code] = p
        return p

def _audio_to_numpy(audio):
    """Normalize whatever Kokoro yields (torch tensor or numpy array) to float32 numpy."""
    import numpy as np
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype="float32")
    return audio

def kokoro_ready(voice="af_sky"):
    """Return True if the KPipeline for this voice's language is already loaded."""
    lang_code = _VOICE_LANG.get(voice, "a")
    return lang_code in _kokoro_pipelines

def preload_kokoro(voice="af_sky"):
    """Eagerly load the KPipeline for this voice. Raises on failure."""
    lang_code = _VOICE_LANG.get(voice, "a")
    _get_kokoro_pipeline(lang_code)
    return True

def speak_to_wav(text, config):
    """Generate a 24 kHz mono WAV (bytes) from text using Kokoro.

    Raises on backend failure so the caller can surface a useful error.
    """
    text = (text or "").strip()
    if not text:
        return None
    voice = (config.get("tts_voice") or "af_sky").strip()
    speed = float(config.get("tts_speed", 1.0) or 1.0)
    lang_code = _VOICE_LANG.get(voice, "a")
    import numpy as np
    import soundfile as sf
    pipeline = _get_kokoro_pipeline(lang_code)
    generator = pipeline(text, voice=voice, speed=speed)
    chunks = []
    for _gs, _ps, audio in generator:
        chunks.append(_audio_to_numpy(audio))
    if not chunks:
        return None
    full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    buf = io.BytesIO()
    sf.write(buf, full, 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()

# ══════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════
# Defaults: TTS auto-playback OFF, per-message playback button HIDDEN.
DEFAULT_CONFIG = {
    # Kokoro TTS
    "tts_auto_speak": False,        # auto-play assistant replies (master toggle for voice feedback)
    "show_playback_button": False,  # show 🔊 button on assistant messages
    "tts_voice": "af_sky",
    "tts_speed": 1.0,
    # Legacy STT (unused by current UI but kept for compatibility)
    "whisper_model": "base",
    "whisper_model_path": "",
    "whisper_device": "auto",
    "whisper_compute": "auto",
    "language": "",
    "beam_size": 5,
    "auto_send": True,
}

def load_speech_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg.update(json.load(f))
        except: pass
    return cfg

def save_speech_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
