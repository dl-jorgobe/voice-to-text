"""
py2app build script for Say the word...

Usage:
    python3.12 setup.py py2app

Creates a standalone .app bundle in dist/ with Python and all
dependencies embedded. Users only need whisper-cli (installed
automatically on first run) and the whisper model (downloaded
automatically on first run).
"""

from setuptools import setup

APP = ['voice_app.py']
DATA_FILES = [
    'config.json',
    'Ndot55.ttf',
    'speak.sh',
]

OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'Say the word.app/Contents/Resources/AppIcon.icns',
    'plist': {
        'CFBundleName': 'Say the word...',
        'CFBundleDisplayName': 'Say the word...',
        'CFBundleIdentifier': 'com.saytheword.app',
        'CFBundleVersion': '2.0',
        'CFBundleShortVersionString': '2.0',
        'LSUIElement': False,
        'NSMicrophoneUsageDescription': 'Say the word... needs microphone access to record your voice for transcription.',
    },
    'packages': ['numpy', 'sounddevice'],
    'includes': [
        'AppKit', 'Quartz', 'CoreText', 'Foundation',
        'objc',
    ],
    'frameworks': [],
    'excludes': ['tkinter', 'matplotlib', 'scipy', 'PIL', 'pytest'],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
