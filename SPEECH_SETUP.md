# AutoType v1.02 — Speech Setup Guide

## Overview
AutoType uses **Kokoro-82M** for Text-to-Speech. Voice feedback is **off by default** —
enable it from the **Speech** settings tab.

---

## Install

```
pip install kokoro soundfile numpy
```

Kokoro depends on `espeak-ng` at runtime for the phonemizer. Install it on your OS:

- Windows: `winget install eSpeak-NG.eSpeak-NG`
- macOS: `brew install espeak-ng`
- Debian/Ubuntu: `sudo apt install espeak-ng`

The first call to TTS downloads the Kokoro-82M model weights (~330 MB) from
HuggingFace and caches them locally.

---

## Settings (Speech tab)

| Setting               | Default | Effect                                                              |
|-----------------------|---------|---------------------------------------------------------------------|
| Auto Voice Playback   | OFF     | Master toggle. When ON, every assistant reply is spoken aloud.      |
| Show Playback Button  | OFF     | When ON, each assistant reply gets a 🔊 button to play it manually. |
| Voice                 | af_sky  | Kokoro voice ID (see catalogue below).                              |
| Speed                 | 1.00×   | Playback speed multiplier (0.5×–2.0×).                              |

Click **TEST VOICE** to preview the selected voice.

---

## Voice catalogue

Full list and audio samples:
<https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md>

**American English (lang_code `a`)** — `af_*` female, `am_*` male
`af_heart`, `af_alloy`, `af_aoede`, `af_bella`, `af_jessica`, `af_kore`,
`af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`,
`am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`,
`am_onyx`, `am_puck`, `am_santa`

**British English (lang_code `b`)** — `bf_*` female, `bm_*` male
`bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`,
`bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

The lang_code is selected automatically from the voice prefix (`a*`/`b*`).

---

## Troubleshooting

**`ModuleNotFoundError: kokoro`** → `pip install kokoro`.
**`OSError: espeak-ng not installed`** → install espeak-ng (see above).
**First TTS request is slow** → model weights are downloading; subsequent calls are fast.
**No audio plays** → the browser may be blocking autoplay; click anywhere in the page first,
then retry.
