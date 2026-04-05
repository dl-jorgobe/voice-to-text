# Voice to Text

A minimal, floating macOS app for voice-to-text. Hold the **Fn key** to record, release to transcribe and paste into any app.

Built with Whisper (local, offline transcription) and a frosted-glass UI inspired by iOS/visionOS.

## Features

- **Hold Fn to record** — transcribed text is automatically pasted into the active app
- **Language toggle** — switch between two languages with one click
- **Hands-free mode** — double-clap to start/stop recording
- **Audio pause/resume** — automatically pauses Spotify, Music, and browser audio while recording
- **Transcription history** — click any previous transcription to copy it

## Install

```bash
git clone https://github.com/dl-jorgobe/voice-to-text.git
cd voice-to-text
./install.sh
```

The install script handles Homebrew, whisper-cli, Python packages, and the Whisper model download.

## Setup

Before using the app:

1. **System Settings > Privacy & Security > Accessibility** — grant access to Voice to Text
2. **System Settings > Keyboard** — set "Press fn key to" to **Do Nothing**

## Launch

Double-click `Voice to Text.app`, or:

```bash
open "Voice to Text.app"
```

## Configuration

Edit `config.json` to set your languages and hint words:

```json
{
  "languages": ["EN", "DA"],
  "whisper_codes": {
    "EN": "en",
    "DA": "da",
    "SE": "sv"
  },
  "hint_words": ["Jorgobé", "Søren", "skincare"]
}
```

- **languages** — the two labels shown in the toggle (e.g. `["EN", "SE"]` for English/Swedish)
- **whisper_codes** — maps each label to the Whisper language code
- **hint_words** — words Whisper should listen for (names, brands, jargon). Leave empty `[]` if not needed

## Requirements

- macOS
- Homebrew
- Python 3
- whisper-cli (`brew install whisper-cpp`)
