# Say the word...

A minimal, floating macOS app for voice-to-text. Hold the **Fn key** to record, release to transcribe and paste into any app.

Built with Whisper (local, offline transcription) and a frosted-glass UI inspired by visionOS.

## Features

- **Hold Fn to record** -- transcribed text is automatically pasted into the active app
- **40+ languages** -- full Whisper language support with searchable dropdown
- **Hands-free mode** -- double-clap to start/stop recording
- **Mini mode** -- collapse to a tiny pill that stays out of the way
- **Audio pause/resume** -- automatically pauses Spotify, Music, and browser audio while recording
- **Transcription history** -- click any previous transcription to copy it
- **Dot-matrix animations** -- Ndot font with scatter/dissolve effects driven by your voice

## Download

1. Go to [Releases](https://github.com/dl-jorgobe/voice-to-text/releases) and download the latest `.zip`
2. Unzip and drag **Say the word...** to your Applications folder
3. Open the app -- it will guide you through first-run setup

## First-run setup

On first launch, the app will:

1. **Install whisper-cli** via Homebrew (if not already installed)
2. **Download the Whisper model** (465 MB, one-time)

You also need to grant two permissions in **System Settings > Privacy & Security**:

- **Accessibility** -- so the app can detect the Fn key
- **Microphone** -- so the app can record your voice

And in **System Settings > Keyboard**, set "Press fn key to" to **Do Nothing**.

## Build from source

If you prefer to build it yourself:

```bash
git clone https://github.com/dl-jorgobe/voice-to-text.git
cd voice-to-text
./install.sh
open "Say the word.app"
```

Or build a standalone `.app` bundle:

```bash
pip3 install py2app
python3 setup.py py2app
open "dist/Say the word....app"
```

## Configuration

Edit `config.json` to set default languages and hint words:

```json
{
  "languages": ["EN", "DA"],
  "whisper_codes": { "EN": "en", "DA": "da" },
  "hint_words": ["your", "custom", "words"]
}
```

- **hint_words** -- words Whisper should prioritize (names, brands, jargon). Leave empty `[]` if not needed.

## Requirements

- macOS (Apple Silicon or Intel)
- Homebrew (installed automatically if missing)
