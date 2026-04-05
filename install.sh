#!/bin/bash
set -e

echo "Voice to Text — installer"
echo "========================="
echo ""

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This app only runs on macOS."
    exit 1
fi

# Check / install Homebrew
if ! command -v brew &>/dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install whisper-cli
if ! command -v whisper-cli &>/dev/null; then
    echo "Installing whisper-cli..."
    brew install whisper-cpp
else
    echo "whisper-cli already installed."
fi

# Install Python 3 if needed
if ! command -v python3 &>/dev/null; then
    echo "Installing Python 3..."
    brew install python3
else
    echo "Python 3 already installed."
fi

# Install Python dependencies
echo "Installing Python packages..."
pip3 install --quiet sounddevice numpy pyobjc-framework-Quartz pyobjc-framework-Cocoa

# Download whisper model
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models"
MODEL_PATH="$MODEL_DIR/ggml-small.bin"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Downloading Whisper model (465 MB)..."
    mkdir -p "$MODEL_DIR"
    curl -L -o "$MODEL_PATH" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
    echo "Model downloaded."
else
    echo "Whisper model already present."
fi

echo ""
echo "Installation complete!"
echo ""
echo "Before using the app, you need to:"
echo "  1. Open System Settings > Privacy & Security > Accessibility"
echo "     and grant access to 'Voice to Text' (or Terminal if running from CLI)"
echo "  2. Open System Settings > Keyboard and set 'Press fn key to' → 'Do Nothing'"
echo ""
echo "To launch: double-click 'Voice to Text.app' or run: open 'Voice to Text.app'"
