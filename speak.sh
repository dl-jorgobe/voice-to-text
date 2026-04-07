#!/bin/bash
# Voice to Text — Text-to-Speech for hands-free mode
# Uses macOS built-in "say" command (free, no API needed)
# Change VOICE to any installed voice: say -v '?' to list all
#
# For higher quality, replace this with ElevenLabs:
#   curl -s --request POST \
#     --url "https://api.elevenlabs.io/v1/text-to-speech/YOUR_VOICE_ID/stream" \
#     --header "xi-api-key: YOUR_KEY" \
#     --header "Content-Type: application/json" \
#     --data "{\"text\": \"$TEXT\"}" -o /tmp/voice.mp3 && afplay /tmp/voice.mp3

VOICE="Samantha"  # Good default. Try also: Daniel (British), Karen (Australian), Alva (Swedish)

TEXT="$1"
if [ -z "$TEXT" ]; then
  echo "Usage: speak.sh \"text to speak\""
  exit 1
fi

say -v "$VOICE" "$TEXT" &
SAY_PID=$!

# Kill speech if interrupted (e.g. user starts recording)
trap "kill $SAY_PID 2>/dev/null" EXIT
wait $SAY_PID 2>/dev/null
