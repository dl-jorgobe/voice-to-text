#!/usr/bin/env python3
"""
Voice to Text — minimal floating macOS window.
Hold Fn to record, release to transcribe and paste.
Muted blue-grey card UI with Ndot dot-matrix text animation.
"""

import os
import sys
import json
import wave
import tempfile
import subprocess
import threading
import logging
import time
import math

import numpy as np
import sounddevice as sd

import objc
import AppKit
import Quartz
from AppKit import (
    NSApplication, NSApp, NSWindow, NSView, NSColor, NSFont,
    NSTextField, NSButton,
    NSVisualEffectView, NSVisualEffectBlendingModeBehindWindow,
    NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable, NSWindowStyleMaskFullSizeContentView,
    NSWindowStyleMaskBorderless,
    NSBackingStoreBuffered, NSFloatingWindowLevel,
    NSMakeRect, NSApplicationActivationPolicyAccessory,
    NSViewWidthSizable, NSViewHeightSizable,
    NSCenterTextAlignment,
    NSLineBreakByTruncatingTail, NSLineBreakByWordWrapping,
    NSFontWeightMedium, NSFontWeightSemibold, NSFontWeightRegular,
    NSFontWeightLight,
    NSControlStateValueOn, NSControlStateValueOff,
)
from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventSetFlags,
    CGEventPost,
    kCGHIDEventTap,
    kCGEventFlagMaskCommand,
    kCGEventSourceStateHIDSystemState,
    CGEventSourceCreate,
)
from Quartz.CoreGraphics import (
    CGEventMaskBit,
    kCGEventFlagsChanged,
    CGEventGetFlags,
)

# ── Load bundled Ndot font ─────────────────────────────────────────────
import CoreText
import Foundation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NDOT_FONT_PATH = os.path.join(SCRIPT_DIR, "Ndot55.ttf")
if os.path.exists(NDOT_FONT_PATH):
    font_url = Foundation.NSURL.fileURLWithPath_(NDOT_FONT_PATH)
    CoreText.CTFontManagerRegisterFontsForURL(
        font_url, CoreText.kCTFontManagerScopeProcess, None
    )

# ── Config ──────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

try:
    with open(CONFIG_PATH) as f:
        CONFIG = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Warning: Could not load {CONFIG_PATH} ({e}), using defaults")
    CONFIG = {"languages": ["EN"], "whisper_codes": {"EN": "en"}, "hint_words": []}

LANG_LABELS = CONFIG.get("languages", ["EN"])
WHISPER_CODES = CONFIG.get("whisper_codes", {"EN": "en"})
HINT_WORDS = CONFIG.get("hint_words", [])

# Store models in ~/Library/Application Support so they persist across app updates
_APP_SUPPORT = os.path.join(os.path.expanduser("~"), "Library", "Application Support", "Say the word")
_MODELS_DIR = os.path.join(_APP_SUPPORT, "models")
# Also check the script directory for backwards compatibility (dev mode)
_DEV_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "ggml-small.bin")
MODEL_PATH = _DEV_MODEL_PATH if os.path.exists(_DEV_MODEL_PATH) else os.path.join(_MODELS_DIR, "ggml-small.bin")
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"

def _find_whisper_cli():
    """Find whisper-cli, checking common locations."""
    result = subprocess.run(["which", "whisper-cli"], capture_output=True, text=True)
    if result.stdout.strip():
        return result.stdout.strip()
    for path in ["/opt/homebrew/bin/whisper-cli", "/usr/local/bin/whisper-cli"]:
        if os.path.exists(path):
            return path
    return None

WHISPER_CMD = _find_whisper_cli()

def _first_run_setup():
    """Check dependencies and download model on first run. Returns True if ready."""
    global WHISPER_CMD, MODEL_PATH
    ready = True

    # Check for Homebrew
    has_brew = subprocess.run(["which", "brew"], capture_output=True).returncode == 0

    # Check / install whisper-cli
    if not WHISPER_CMD:
        if has_brew:
            result = subprocess.run(
                ["osascript", "-e",
                 'display dialog "Say the word... needs to install whisper-cli (speech engine).\n\nThis is a one-time setup." '
                 'buttons {"Cancel", "Install"} default button "Install" with title "First Run Setup"'],
                capture_output=True, text=True,
            )
            if "Install" in result.stdout:
                # Install in Terminal so user can see progress
                subprocess.run([
                    "osascript", "-e",
                    'tell application "Terminal" to do script "brew install whisper-cpp && echo Done — you can close this window"'
                ], capture_output=True)
                subprocess.run([
                    "osascript", "-e",
                    'display dialog "Installing whisper-cli in Terminal.\n\nReopen this app when it finishes." '
                    'buttons {"OK"} default button "OK" with title "Installing..."'
                ], capture_output=True)
                sys.exit(0)
            else:
                sys.exit(0)
        else:
            subprocess.run([
                "osascript", "-e",
                'display alert "Say the word..." message "Homebrew and whisper-cli are required.\n\n'
                'Install Homebrew first:\n/bin/bash -c \\"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\\"\n\n'
                'Then: brew install whisper-cpp" as critical'
            ], capture_output=True)
            sys.exit(1)

    # Download whisper model if missing
    if not os.path.exists(MODEL_PATH):
        result = subprocess.run(
            ["osascript", "-e",
             'display dialog "Say the word... needs to download the speech model (465 MB).\n\nThis is a one-time download." '
             'buttons {"Cancel", "Download"} default button "Download" with title "First Run Setup"'],
            capture_output=True, text=True,
        )
        if "Download" not in result.stdout:
            sys.exit(0)

        # Use Application Support for downloaded models
        MODEL_PATH = os.path.join(_MODELS_DIR, "ggml-small.bin")
        os.makedirs(_MODELS_DIR, exist_ok=True)
        print("Downloading whisper model...")
        dl_result = subprocess.run(
            ["curl", "-L", "--progress-bar", "-o", MODEL_PATH, MODEL_URL],
            timeout=600,
        )
        if dl_result.returncode != 0 or not os.path.exists(MODEL_PATH):
            subprocess.run([
                "osascript", "-e",
                'display alert "Download failed" message "Could not download the speech model. Check your internet connection and try again." as critical'
            ], capture_output=True)
            sys.exit(1)

    return True
SAMPLE_RATE = 16000
CHANNELS = 1
FN_FLAG = 1 << 23

# ── Layout dimensions ──────────────────────────────────────────────────
WIN_WIDTH = 300
WIN_HEIGHT = 290
MINI_WIDTH = 220
MINI_HEIGHT = 42
CORNER_RADIUS = 36

# ── Color palette (neutral frosted glass — original style) ─────────────
# Card background tint — very subtle warm-neutral
BG_R, BG_G, BG_B = 0.45, 0.45, 0.48
BG_ALPHA = 0.15
# Darker inner fill
INNER_R, INNER_G, INNER_B = 0.18, 0.17, 0.22
# Text colors
TEXT_PRIMARY = (1.0, 1.0, 1.0, 1.0)         # white
TEXT_SECONDARY = (1.0, 1.0, 1.0, 0.45)      # muted white
# Button pill (dark)
BTN_R, BTN_G, BTN_B, BTN_A = 0.15, 0.17, 0.20, 0.85

# ── Logging ─────────────────────────────────────────────────────────────
# Use Application Support for logs when running as bundled app
if getattr(sys, 'frozen', False):
    os.makedirs(_APP_SUPPORT, exist_ok=True)
    LOG_PATH = os.path.join(_APP_SUPPORT, "voice.log")
else:
    LOG_PATH = os.path.join(SCRIPT_DIR, "voice.log")
logging.basicConfig(filename=LOG_PATH, level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s: %(message)s")


# ── Passthrough view (for decorative overlays that shouldn't eat clicks) ─
class PassthroughView(NSView):
    def hitTest_(self, point):
        return None


# ── Idle dot view ───────────────────────────────────────────────────────
class CircleView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(CircleView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._color = NSColor.systemGrayColor()
        return self

    def setColor_(self, color):
        self._color = color
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        path = AppKit.NSBezierPath.bezierPathWithOvalInRect_(self.bounds())
        self._color.setFill()
        path.fill()


# ── Waveform view (used in mini mode) ──────────────────────────────────
class WaveformView(NSView):
    NUM_BARS = 7
    BAR_WIDTH = 4
    BAR_GAP = 3
    MIN_HEIGHT = 4
    MAX_HEIGHT = 32

    def initWithFrame_(self, frame):
        self = objc.super(WaveformView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._levels = [0.0] * self.NUM_BARS
        self._color = NSColor.whiteColor()
        return self

    def setLevels_(self, levels):
        self._levels = levels
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        bounds = self.bounds()
        total_w = self.NUM_BARS * self.BAR_WIDTH + (self.NUM_BARS - 1) * self.BAR_GAP
        start_x = (bounds.size.width - total_w) / 2

        for i, level in enumerate(self._levels):
            h = self.MIN_HEIGHT + level * (self.MAX_HEIGHT - self.MIN_HEIGHT)
            x = start_x + i * (self.BAR_WIDTH + self.BAR_GAP)
            y = (bounds.size.height - h) / 2
            bar_rect = NSMakeRect(x, y, self.BAR_WIDTH, h)
            path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                bar_rect, self.BAR_WIDTH / 2, self.BAR_WIDTH / 2
            )
            self._color.setFill()
            path.fill()


# ── Dot Text View (Ndot font with animated dots) ──────────────────────
class DotTextView(NSView):
    """Renders Ndot text normally when idle.
    When recording, each letter's dots scatter outward from that letter's center,
    driven by audio volume. Dots settle back smoothly when recording stops."""

    DOT_RADIUS = 1.2

    def initWithFrame_(self, frame):
        self = objc.super(DotTextView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._text = ""
        self._dots = []              # [(x, y), ...] base positions in view coords
        self._scatter_dirs = []      # [(dx, dy), ...] normalized direction from letter center
        self._dot_speeds = []        # per-dot random speed multiplier (0.5–2.0)
        self._dot_drift = []         # per-dot subtle random drift angle offset
        self._spread = 0.0           # current spread amount (0 = text, 1 = fully scattered)
        self._target_spread = 0.0    # target spread (driven by audio)
        self._anim_time = 0.0        # elapsed time for drift animation
        self._color = NSColor.colorWithCalibratedRed_green_blue_alpha_(*TEXT_PRIMARY)
        self._font_size = 24
        self._animating = False
        self._font = NSFont.fontWithName_size_("Ndot 55", 24) or \
                     NSFont.fontWithName_size_("Ndot 57 Aligned", 24) or \
                     NSFont.systemFontOfSize_(24)
        self._attr_str = None
        self._text_origin = AppKit.NSMakePoint(0, 0)
        self._text_y_center = None   # if set, override vertical centering
        self._max_scatter = 80.0     # max pixels dots can scatter
        self._visible_dots = -1      # -1 = all visible, 0..N = how many to show
        self._dot_order = []         # randomized order for reveal/dissolve
        self._visible_set = None     # precomputed set for O(1) lookups
        return self

    def setText_(self, text):
        self._text = text
        self._build_attr_str()
        self._extract_dots_per_letter()
        self._visible_dots = -1  # all visible
        self._visible_set = None
        import random
        self._dot_order = list(range(len(self._dots)))
        random.shuffle(self._dot_order)
        self.setNeedsDisplay_(True)

    def dissolve(self, callback=None):
        """Animate dots disappearing one by one, fast. Calls callback when done."""
        if not self._dots:
            if callback:
                callback()
            return
        total = len(self._dots)
        import random
        self._dot_order = list(range(total))
        random.shuffle(self._dot_order)
        self._visible_dots = total
        self._rebuild_visible_set()

        def _run():
            # Remove dots in batches for speed
            batch = max(1, total // 8)  # ~8 steps
            while self._visible_dots > 0:
                self._visible_dots = max(0, self._visible_dots - batch)
                self._rebuild_visible_set()
                def _r(): self.setNeedsDisplay_(True)
                AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(_r)
                time.sleep(0.015)
            if callback:
                AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(callback)
        threading.Thread(target=_run, daemon=True).start()

    def reveal(self):
        """Animate dots appearing one by one, fast."""
        if not self._dots:
            return
        total = len(self._dots)
        import random
        self._dot_order = list(range(total))
        random.shuffle(self._dot_order)
        self._visible_dots = 0
        self._rebuild_visible_set()

        def _run():
            batch = max(1, total // 8)
            while self._visible_dots < total:
                self._visible_dots = min(total, self._visible_dots + batch)
                self._rebuild_visible_set()
                def _r(): self.setNeedsDisplay_(True)
                AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(_r)
                time.sleep(0.015)
            self._visible_dots = -1  # all visible
            self._visible_set = None
            def _r(): self.setNeedsDisplay_(True)
            AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(_r)
        threading.Thread(target=_run, daemon=True).start()

    def setColor_(self, color):
        self._color = color
        self._build_attr_str()
        self.setNeedsDisplay_(True)

    def setAnimating_(self, animating):
        self._animating = animating
        if not animating:
            self._target_spread = 0.0
        self.setNeedsDisplay_(True)

    def setSpread_(self, spread):
        """Set target scatter spread (0.0 = text, 1.0 = max scatter)."""
        self._target_spread = min(1.0, max(0.0, spread))

    def updateAnimation(self):
        """Smooth interpolation toward target spread. Call at ~20fps."""
        self._anim_time += 0.05
        if self._animating:
            self._spread += (self._target_spread - self._spread) * 0.45
        else:
            # Smooth settle back
            self._spread += (0.0 - self._spread) * 0.12
            if self._spread < 0.005:
                self._spread = 0.0
        self.setNeedsDisplay_(True)

    def _build_attr_str(self):
        """Build the attributed string for direct text rendering."""
        attrs = {
            AppKit.NSFontAttributeName: self._font,
            AppKit.NSForegroundColorAttributeName: self._color,
        }
        self._attr_str = AppKit.NSAttributedString.alloc().initWithString_attributes_(
            self._text, attrs
        )
        size = self._attr_str.size()
        view_w = self.bounds().size.width
        view_h = self.bounds().size.height
        x = (view_w - size.width) / 2
        if self._text_y_center is not None:
            y = self._text_y_center - size.height / 2
        else:
            y = (view_h - size.height) / 2
        self._text_origin = AppKit.NSMakePoint(x, y)

    def _extract_dots_per_letter(self):
        """Render each letter individually to get per-letter dot groups,
        then compute scatter directions from each letter's centroid."""
        all_dots = []
        all_dirs = []
        all_speeds = []
        all_drifts = []

        # Get per-character x advances to know where each letter sits
        attrs = {
            AppKit.NSFontAttributeName: self._font,
            AppKit.NSForegroundColorAttributeName: NSColor.whiteColor(),
        }

        # Measure full text width for centering
        full_attr = AppKit.NSAttributedString.alloc().initWithString_attributes_(
            self._text, attrs
        )
        full_size = full_attr.size()
        view_w = self.bounds().size.width
        view_h = self.bounds().size.height
        text_x_offset = (view_w - full_size.width) / 2
        if self._text_y_center is not None:
            text_y_offset = self._text_y_center - full_size.height / 2
        else:
            text_y_offset = (view_h - full_size.height) / 2

        # Get x position of each character using advancing
        char_x_positions = [0.0]
        for i in range(len(self._text)):
            sub_attr = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                self._text[:i + 1], attrs
            )
            char_x_positions.append(sub_attr.size().width)

        pt_h = int(full_size.height) + 10
        step = 2

        for ci, char in enumerate(self._text):
            if char == ' ':
                continue

            # Render single char to bitmap
            char_attr = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                char, attrs
            )
            char_size = char_attr.size()
            cw = max(1, int(char_size.width) + 4)
            ch = max(1, int(char_size.height) + 4)

            bitmap = AppKit.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
                None, cw, ch, 8, 4, True, False,
                AppKit.NSCalibratedRGBColorSpace, 0, 0
            )
            if not bitmap:
                continue

            ctx = AppKit.NSGraphicsContext.graphicsContextWithBitmapImageRep_(bitmap)
            if not ctx:
                continue

            AppKit.NSGraphicsContext.saveGraphicsState()
            AppKit.NSGraphicsContext.setCurrentContext_(ctx)
            NSColor.clearColor().setFill()
            AppKit.NSBezierPath.fillRect_(NSMakeRect(0, 0, cw, ch))
            char_attr.drawAtPoint_(AppKit.NSMakePoint(2, 2))
            AppKit.NSGraphicsContext.restoreGraphicsState()

            # Extract dots from this character
            char_dots = []
            for py in range(0, ch, step):
                for px in range(0, cw, step):
                    color = bitmap.colorAtX_y_(px, py)
                    if color:
                        try:
                            a = color.alphaComponent()
                        except Exception:
                            continue
                        if a > 0.3:
                            # Position in view coords
                            vx = text_x_offset + char_x_positions[ci] + float(px) - 2
                            vy = text_y_offset + float(ch - py) - 2
                            char_dots.append((vx, vy))

            if not char_dots:
                continue

            # Compute centroid of this letter's dots
            cx = sum(d[0] for d in char_dots) / len(char_dots)
            cy = sum(d[1] for d in char_dots) / len(char_dots)

            # Compute scatter direction for each dot (away from centroid)
            # Add per-dot randomness for organic, weightless feel
            import random
            for (dx, dy) in char_dots:
                dist = math.sqrt((dx - cx) ** 2 + (dy - cy) ** 2)
                if dist > 0.1:
                    nx = (dx - cx) / dist
                    ny = (dy - cy) / dist
                else:
                    angle = hash((dx, dy)) % 628 / 100.0
                    nx = math.cos(angle)
                    ny = math.sin(angle)
                # Add slight random angle deviation (±30°) for organic spread
                angle_offset = (hash((int(dx * 7), int(dy * 13))) % 600 - 300) / 1000.0 * math.pi
                cos_a = math.cos(angle_offset)
                sin_a = math.sin(angle_offset)
                nx2 = nx * cos_a - ny * sin_a
                ny2 = nx * sin_a + ny * cos_a
                all_dots.append((dx, dy))
                all_dirs.append((nx2, ny2))
                # Random speed: some dots fly faster than others (0.4 – 2.2)
                speed = 0.4 + (hash((int(dx * 3), int(dy * 7))) % 180) / 100.0
                all_speeds.append(speed)
                # Random drift phase for floating wobble
                drift = (hash((int(dx * 11), int(dy * 17))) % 628) / 100.0
                all_drifts.append(drift)

        self._dots = all_dots
        self._scatter_dirs = all_dirs
        self._dot_speeds = all_speeds
        self._dot_drift = all_drifts
        self._spread = 0.0
        self._target_spread = 0.0
        logging.info(f"DotTextView: extracted {len(self._dots)} dots from '{self._text}' ({len(self._text)} chars)")

    def _rebuild_visible_set(self):
        """Rebuild the set of visible dot indices from _dot_order and _visible_dots."""
        if self._visible_dots == -1 or not self._dot_order:
            self._visible_set = None
        else:
            self._visible_set = set(self._dot_order[:self._visible_dots])

    def _is_dot_visible(self, i):
        """Check if dot at index i should be drawn during dissolve/reveal."""
        if self._visible_dots == -1:
            return True  # all visible
        if self._visible_set is None:
            return i < self._visible_dots
        return i in self._visible_set

    def drawRect_(self, rect):
        if self._visible_dots == 0:
            return  # fully dissolved, draw nothing

        if self._spread > 0.001 and self._dots:
            # Animated mode: dots float away like weightless in space
            self._color.setFill()
            r = self.DOT_RADIUS
            t = self._anim_time
            for i, (x, y) in enumerate(self._dots):
                if not self._is_dot_visible(i):
                    continue
                if i < len(self._scatter_dirs):
                    sx, sy = self._scatter_dirs[i]
                else:
                    sx, sy = 0.0, 0.0
                speed = self._dot_speeds[i] if i < len(self._dot_speeds) else 1.0
                scatter = self._spread * self._max_scatter * speed
                dx = x + sx * scatter
                dy = y + sy * scatter
                if self._spread > 0.01:
                    drift_phase = self._dot_drift[i] if i < len(self._dot_drift) else 0.0
                    wobble = math.sin(t * 2.5 + drift_phase) * self._spread * 6.0 * speed
                    dx += -sy * wobble
                    dy += sx * wobble
                dot_rect = NSMakeRect(dx - r, dy - r, r * 2, r * 2)
                path = AppKit.NSBezierPath.bezierPathWithOvalInRect_(dot_rect)
                path.fill()
        elif self._visible_dots != -1 and self._dots:
            # Dissolve/reveal mode: draw individual dots
            self._color.setFill()
            r = self.DOT_RADIUS
            for i, (x, y) in enumerate(self._dots):
                if not self._is_dot_visible(i):
                    continue
                dot_rect = NSMakeRect(x - r, y - r, r * 2, r * 2)
                path = AppKit.NSBezierPath.bezierPathWithOvalInRect_(dot_rect)
                path.fill()
        elif self._attr_str:
            # Idle mode: draw text directly with Ndot font
            self._attr_str.drawAtPoint_(self._text_origin)


# ── Clickable history label ─────────────────────────────────────────────
class ClickableLabel(NSTextField):
    def initWithFrame_(self, frame):
        self = objc.super(ClickableLabel, self).initWithFrame_(frame)
        if self is None:
            return None
        self._full_text = ""
        self._click_callback = None
        return self

    def setFullText_(self, text):
        self._full_text = text

    def setClickCallback_(self, cb):
        self._click_callback = cb

    def mouseDown_(self, event):
        if self._click_callback and self._full_text:
            self._click_callback(self._full_text)

    def resetCursorRects(self):
        self.addCursorRect_cursor_(self.bounds(), AppKit.NSCursor.pointingHandCursor())


# ── All Whisper-supported languages ────────────────────────────────────
WHISPER_LANGUAGES = [
    ("auto", "Auto-detect"), ("af", "Afrikaans"), ("am", "Amharic"), ("ar", "Arabic"),
    ("as", "Assamese"), ("az", "Azerbaijani"), ("ba", "Bashkir"), ("be", "Belarusian"),
    ("bg", "Bulgarian"), ("bn", "Bengali"), ("bo", "Tibetan"), ("br", "Breton"),
    ("bs", "Bosnian"), ("ca", "Catalan"), ("cs", "Czech"), ("cy", "Welsh"),
    ("da", "Danish"), ("de", "German"), ("el", "Greek"), ("en", "English"),
    ("es", "Spanish"), ("et", "Estonian"), ("eu", "Basque"), ("fa", "Persian"),
    ("fi", "Finnish"), ("fo", "Faroese"), ("fr", "French"), ("gl", "Galician"),
    ("gu", "Gujarati"), ("ha", "Hausa"), ("haw", "Hawaiian"), ("he", "Hebrew"),
    ("hi", "Hindi"), ("hr", "Croatian"), ("ht", "Haitian Creole"), ("hu", "Hungarian"),
    ("hy", "Armenian"), ("id", "Indonesian"), ("is", "Icelandic"), ("it", "Italian"),
    ("ja", "Japanese"), ("jw", "Javanese"), ("ka", "Georgian"), ("kk", "Kazakh"),
    ("km", "Khmer"), ("kn", "Kannada"), ("ko", "Korean"), ("la", "Latin"),
    ("lb", "Luxembourgish"), ("ln", "Lingala"), ("lo", "Lao"), ("lt", "Lithuanian"),
    ("lv", "Latvian"), ("mg", "Malagasy"), ("mi", "Maori"), ("mk", "Macedonian"),
    ("ml", "Malayalam"), ("mn", "Mongolian"), ("mr", "Marathi"), ("ms", "Malay"),
    ("mt", "Maltese"), ("my", "Myanmar"), ("ne", "Nepali"), ("nl", "Dutch"),
    ("nn", "Nynorsk"), ("no", "Norwegian"), ("oc", "Occitan"), ("pa", "Punjabi"),
    ("pl", "Polish"), ("ps", "Pashto"), ("pt", "Portuguese"), ("ro", "Romanian"),
    ("ru", "Russian"), ("sa", "Sanskrit"), ("sd", "Sindhi"), ("si", "Sinhala"),
    ("sk", "Slovak"), ("sl", "Slovenian"), ("sn", "Shona"), ("so", "Somali"),
    ("sq", "Albanian"), ("sr", "Serbian"), ("su", "Sundanese"), ("sv", "Swedish"),
    ("sw", "Swahili"), ("ta", "Tamil"), ("te", "Telugu"), ("tg", "Tajik"),
    ("th", "Thai"), ("tk", "Turkmen"), ("tl", "Tagalog"), ("tr", "Turkish"),
    ("tt", "Tatar"), ("uk", "Ukrainian"), ("ur", "Urdu"), ("uz", "Uzbek"),
    ("vi", "Vietnamese"), ("yi", "Yiddish"), ("yo", "Yoruba"), ("yue", "Cantonese"),
    ("zh", "Chinese"),
]

# ── Language dropdown (clickable pill → popup menu) ────────────────────
class LangDropdownView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(LangDropdownView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._selected_code = "en"
        self._selected_name = "English"
        self._callback = None
        return self

    def setCallback_(self, cb):
        self._callback = cb

    def setLanguage_code_(self, name, code):
        self._selected_name = name
        self._selected_code = code
        self.setNeedsDisplay_(True)

    def mouseDown_(self, event):
        # Toggle popup
        if hasattr(self, '_popup_window') and self._popup_window and self._popup_window.isVisible():
            self._popup_window.orderOut_(None)
            return
        self._open_popup()

    def _open_popup(self):
        # Position below the pill
        pill_screen = self.window().convertRectToScreen_(
            self.convertRect_toView_(self.bounds(), None)
        )
        popup_w = 150
        popup_h = 280
        popup_x = pill_screen.origin.x + (pill_screen.size.width - popup_w) / 2
        popup_y = pill_screen.origin.y - popup_h - 4

        popup = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(popup_x, popup_y, popup_w, popup_h),
            AppKit.NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )
        popup.setLevel_(NSFloatingWindowLevel + 1)
        popup.setOpaque_(False)
        popup.setBackgroundColor_(NSColor.clearColor())
        popup.setHasShadow_(True)

        content = popup.contentView()
        content.setWantsLayer_(True)
        content.layer().setCornerRadius_(14)
        content.layer().setMasksToBounds_(True)

        # Dark background
        tint = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, popup_w, popup_h))
        tint.setWantsLayer_(True)
        tint.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.0, 0.0, 0.25).CGColor()
        )
        content.addSubview_(tint)

        # Blur
        blur = NSVisualEffectView.alloc().initWithFrame_(NSMakeRect(0, 0, popup_w, popup_h))
        blur.setMaterial_(13)
        blur.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        blur.setState_(1)
        blur.setAlphaValue_(0.7)
        blur.setWantsLayer_(True)
        blur.layer().setCornerRadius_(14)
        blur.layer().setMasksToBounds_(True)
        content.addSubview_(blur)

        # Scroll view
        scroll = AppKit.NSScrollView.alloc().initWithFrame_(
            NSMakeRect(0, 0, popup_w, popup_h)
        )
        scroll.setHasVerticalScroller_(True)
        scroll.setDrawsBackground_(False)
        scroll.setBorderType_(0)
        scroller = scroll.verticalScroller()
        if scroller:
            scroller.setAlphaValue_(0.3)

        row_h = 26
        total_h = len(WHISPER_LANGUAGES) * row_h
        doc = AppKit.NSFlippedView.alloc().initWithFrame_(
            NSMakeRect(0, 0, popup_w, max(total_h, popup_h))
        )

        # Keep strong refs to all rows
        self._popup_rows = []
        for idx, (code, name) in enumerate(WHISPER_LANGUAGES):
            y = idx * row_h
            btn = AppKit.NSButton.alloc().initWithFrame_(
                NSMakeRect(4, y, popup_w - 8, row_h)
            )
            btn.setBezelStyle_(0)  # NSBezelStyleInline = 0 doesn't exist, use borderless
            btn.setBordered_(False)
            btn.setWantsLayer_(True)
            btn.layer().setCornerRadius_(8)

            # Attributed title
            is_sel = (code == self._selected_code)
            weight = NSFontWeightSemibold if is_sel else NSFontWeightRegular
            color = NSColor.whiteColor() if is_sel else \
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.65)
            font = NSFont.systemFontOfSize_weight_(11, weight)
            prefix = "✓ " if is_sel else "  "
            attrs = {
                AppKit.NSFontAttributeName: font,
                AppKit.NSForegroundColorAttributeName: color,
            }
            title = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                prefix + name.upper(), attrs
            )
            btn.setAttributedTitle_(title)
            btn.setAlignment_(NSCenterTextAlignment)
            btn.setTag_(idx)
            btn.setTarget_(self)
            btn.setAction_(objc.selector(self.langBtnClicked_, signature=b'v@:@'))
            doc.addSubview_(btn)
            self._popup_rows.append(btn)

        scroll.setDocumentView_(doc)

        # Scroll to selected
        for idx, (code, _) in enumerate(WHISPER_LANGUAGES):
            if code == self._selected_code:
                target_y = idx * row_h - popup_h / 2 + row_h
                doc.scrollPoint_(AppKit.NSMakePoint(0, max(0, target_y)))
                break

        content.addSubview_(scroll)

        # Store strong refs
        self._popup_window = popup
        self._popup_tint = tint
        self._popup_blur = blur
        self._popup_scroll = scroll
        self._popup_doc = doc

        # Make parent window the popup's parent so it closes with it
        self.window().addChildWindow_ordered_(popup, AppKit.NSWindowAbove)
        popup.makeKeyAndOrderFront_(None)

    def langBtnClicked_(self, sender):
        idx = sender.tag()
        if 0 <= idx < len(WHISPER_LANGUAGES):
            code, name = WHISPER_LANGUAGES[idx]
            self._selected_code = code
            self._selected_name = name
            self.setNeedsDisplay_(True)
            if self._callback:
                self._callback(code)
        # Close popup on next run loop tick to avoid use-after-free
        def _close():
            if self._popup_window:
                self.window().removeChildWindow_(self._popup_window)
                self._popup_window.orderOut_(None)
                self._popup_window = None
                self._popup_rows = None
        AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(_close)

    langBtnClicked_ = objc.selector(langBtnClicked_, signature=b'v@:@')

    def resetCursorRects(self):
        self.addCursorRect_cursor_(self.bounds(), AppKit.NSCursor.pointingHandCursor())

    def drawRect_(self, rect):
        bounds = self.bounds()
        w = bounds.size.width
        h = bounds.size.height
        r = h / 2

        # Pill background
        track = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(bounds, r, r)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.0, 0.0, 0.25).setFill()
        track.fill()

        # Language name + arrow — centered together
        label = f"{self._selected_name.upper()} ▾"
        font = NSFont.systemFontOfSize_weight_(11, NSFontWeightSemibold)
        color = NSColor.whiteColor()
        attrs = {
            AppKit.NSFontAttributeName: font,
            AppKit.NSForegroundColorAttributeName: color,
        }
        attr_str = AppKit.NSAttributedString.alloc().initWithString_attributes_(label, attrs)
        size = attr_str.size()
        x = (w - size.width) / 2
        y = (h - size.height) / 2
        attr_str.drawAtPoint_(AppKit.NSMakePoint(x, y))



# ── Flipped view (for top-to-bottom layout in scroll views) ────────────
class _FlippedView(NSView):
    def isFlipped(self):
        return True

# Register as NSFlippedView for use in the dropdown
AppKit.NSFlippedView = _FlippedView

# ── Hands-Free toggle (single on/off button) ────────────────────────────
class HandsFreeToggleView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(HandsFreeToggleView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._active = False
        self._callback = None
        return self

    def setCallback_(self, cb):
        self._callback = cb

    def isActive(self):
        return self._active

    def setActive_(self, active):
        self._active = active
        self.setNeedsDisplay_(True)

    def mouseDown_(self, event):
        self._active = not self._active
        self.setNeedsDisplay_(True)
        if self._callback:
            self._callback(self._active)

    def resetCursorRects(self):
        self.addCursorRect_cursor_(self.bounds(), AppKit.NSCursor.pointingHandCursor())

    def drawRect_(self, rect):
        bounds = self.bounds()
        w = bounds.size.width
        h = bounds.size.height
        r = h / 2

        # Track background
        track = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(bounds, r, r)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.0, 0.0, 0.25).setFill()
        track.fill()

        if self._active:
            pill = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                bounds, r, r
            )
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.0, 0.0, 0.35).setFill()
            pill.fill()

        label = "● HANDS-FREE" if self._active else "HANDS-FREE"
        color = NSColor.whiteColor() if self._active else NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.45)
        font = NSFont.systemFontOfSize_weight_(11, NSFontWeightSemibold if self._active else NSFontWeightRegular)
        attrs = {
            AppKit.NSFontAttributeName: font,
            AppKit.NSForegroundColorAttributeName: color,
        }
        attr_str = AppKit.NSAttributedString.alloc().initWithString_attributes_(label, attrs)
        size = attr_str.size()
        x = (w - size.width) / 2
        y = (h - size.height) / 2
        attr_str.drawAtPoint_(AppKit.NSMakePoint(x, y))


# ── Mini mode toggle button ────────────────────────────────────────────
class MiniToggleView(NSView):
    """Small clickable pill to toggle between full and mini mode."""
    def initWithFrame_(self, frame):
        self = objc.super(MiniToggleView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._callback = None
        self._is_mini = False
        return self

    def setCallback_(self, cb):
        self._callback = cb

    def setMini_(self, mini):
        self._is_mini = mini
        self.setNeedsDisplay_(True)

    def mouseDown_(self, event):
        if self._callback:
            self._callback()

    def resetCursorRects(self):
        self.addCursorRect_cursor_(self.bounds(), AppKit.NSCursor.pointingHandCursor())

    def drawRect_(self, rect):
        bounds = self.bounds()
        w = bounds.size.width
        h = bounds.size.height

        # Draw a subtle pill background
        pill = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            bounds, h / 2, h / 2
        )
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.0, 0.0, 0.35).setFill()
        pill.fill()

        # Draw collapse/expand icon (two horizontal lines for collapse, expand arrows)
        icon = "▴" if not self._is_mini else "▾"
        font = NSFont.systemFontOfSize_weight_(10, NSFontWeightMedium)
        color = NSColor.whiteColor() if self._is_mini else NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.6)
        attrs = {
            AppKit.NSFontAttributeName: font,
            AppKit.NSForegroundColorAttributeName: color,
        }
        attr_str = AppKit.NSAttributedString.alloc().initWithString_attributes_(icon, attrs)
        size = attr_str.size()
        attr_str.drawAtPoint_(AppKit.NSMakePoint(
            (w - size.width) / 2, (h - size.height) / 2
        ))


class VoiceToTextApp:
    def __init__(self):
        self.recording = False
        self.audio_frames = []
        self.stream = None
        self.paused_sources = []
        self.language = "en"
        self.hands_free_mode = False
        self._stop_clap_monitor = threading.Event()
        self._is_mini = False
        self._animating_waveform = False
        self._mini_lock = threading.Lock()

        self.app = NSApplication.sharedApplication()
        self.app.setActivationPolicy_(0)  # NSApplicationActivationPolicyRegular

        self.build_window()
        self.set_dock_icon()
        self.setup_event_tap()

    def set_dock_icon(self):
        size = 256
        icon = AppKit.NSImage.alloc().initWithSize_((size, size))
        icon.lockFocus()

        inset = size * 0.1
        icon_sz = size * 0.8
        radius = icon_sz * 0.22
        rect = NSMakeRect(inset, inset, icon_sz, icon_sz)
        shape = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, radius, radius)

        # Gradient background — dark
        top_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.14, 0.14, 0.16, 1.0)
        bot_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.05, 0.05, 0.06, 1.0)
        gradient = AppKit.NSGradient.alloc().initWithStartingColor_endingColor_(bot_color, top_color)
        gradient.drawInBezierPath_angle_(shape, 90)

        # Edge highlight
        AppKit.NSGraphicsContext.currentContext().saveGraphicsState()
        shape.addClip()
        inner_rect = NSMakeRect(inset + 0.5, inset + 0.5, icon_sz - 1, icon_sz - 1)
        inner_shape = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(inner_rect, radius - 0.5, radius - 0.5)
        hl_top = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.15)
        hl_bot = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.03)
        stroke_grad = AppKit.NSGradient.alloc().initWithStartingColor_endingColor_(hl_bot, hl_top)
        stroke_grad.drawInBezierPath_angle_(inner_shape, 90)
        fill_rect = NSMakeRect(inset + 1.5, inset + 1.5, icon_sz - 3, icon_sz - 3)
        fill_shape = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(fill_rect, radius - 1.5, radius - 1.5)
        gradient2 = AppKit.NSGradient.alloc().initWithStartingColor_endingColor_(bot_color, top_color)
        gradient2.drawInBezierPath_angle_(fill_shape, 90)
        AppKit.NSGraphicsContext.currentContext().restoreGraphicsState()

        # 3 dots centered
        num = 3
        dot_r = icon_sz * 0.055
        gap = icon_sz * 0.08
        total_w = num * (dot_r * 2) + (num - 1) * gap
        start_x = inset + (icon_sz - total_w) / 2
        cy = inset + icon_sz / 2

        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.85).setFill()
        for i in range(num):
            cx = start_x + dot_r + i * (dot_r * 2 + gap)
            dot = AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx - dot_r, cy - dot_r, dot_r * 2, dot_r * 2)
            )
            dot.fill()

        icon.unlockFocus()
        self.app.setApplicationIconImage_(icon)

    def build_window(self):
        screen = AppKit.NSScreen.mainScreen().frame()
        x = (screen.size.width - WIN_WIDTH) / 2  # top-center
        y = screen.size.height - WIN_HEIGHT - 40

        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                 NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskFullSizeContentView)

        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, WIN_WIDTH, WIN_HEIGHT),
            style,
            NSBackingStoreBuffered,
            False,
        )

        self.window.setLevel_(NSFloatingWindowLevel)
        self.window.setTitle_("Say the word...")
        self.window.setTitleVisibility_(1)  # NSWindowTitleHidden
        self.window.setTitlebarAppearsTransparent_(True)
        self.window.setMovableByWindowBackground_(True)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(NSColor.clearColor())
        self.window.setHasShadow_(True)
        self.window.setAlphaValue_(0.97)

        # ── Hide window buttons (close/minimize/zoom) ──
        for btn_type in [AppKit.NSWindowCloseButton, AppKit.NSWindowMiniaturizeButton, AppKit.NSWindowZoomButton]:
            btn = self.window.standardWindowButton_(btn_type)
            if btn:
                btn.setHidden_(True)

        # ── Muted blue-grey card background ──
        content = self.window.contentView()
        content.setWantsLayer_(True)
        content.layer().setCornerRadius_(CORNER_RADIUS)
        content.layer().setCornerCurve_("continuous")  # squircle
        content.layer().setMasksToBounds_(True)

        # Base tint
        tint_view = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, WIN_WIDTH, WIN_HEIGHT))
        tint_view.setWantsLayer_(True)
        tint_view.layer().setCornerRadius_(CORNER_RADIUS)
        tint_view.layer().setCornerCurve_("continuous")
        tint_view.layer().setMasksToBounds_(True)
        tint_view.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(BG_R, BG_G, BG_B, BG_ALPHA).CGColor()
        )
        content.addSubview_(tint_view)
        self._tint_view = tint_view

        # Frosted blur
        vibrancy = NSVisualEffectView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WIN_WIDTH, WIN_HEIGHT)
        )
        vibrancy.setMaterial_(13)
        vibrancy.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        vibrancy.setState_(1)
        vibrancy.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        vibrancy.setWantsLayer_(True)
        vibrancy.layer().setCornerRadius_(CORNER_RADIUS)
        vibrancy.layer().setCornerCurve_("continuous")
        vibrancy.layer().setMasksToBounds_(True)
        vibrancy.setAlphaValue_(0.75)
        content.addSubview_(vibrancy)
        self._vibrancy = vibrancy

        # Specular highlight — subtle edge glow
        from Quartz import CAGradientLayer, CAShapeLayer
        highlight_view = PassthroughView.alloc().initWithFrame_(NSMakeRect(0, 0, WIN_WIDTH, WIN_HEIGHT))
        highlight_view.setWantsLayer_(True)
        highlight_view.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)

        specular = CAGradientLayer.alloc().init()
        specular.setFrame_(Quartz.CGRectMake(0, 0, WIN_WIDTH, WIN_HEIGHT))
        white_bright = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.30).CGColor()
        white_mid = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.06).CGColor()
        clear = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.0).CGColor()
        specular.setColors_([clear, white_mid, white_bright])
        specular.setLocations_([0.0, 0.75, 1.0])
        specular.setStartPoint_(Quartz.CGPointMake(0.5, 0.0))
        specular.setEndPoint_(Quartz.CGPointMake(0.5, 1.0))

        edge_mask = CAShapeLayer.alloc().init()
        edge_path = Quartz.CGPathCreateWithRoundedRect(
            Quartz.CGRectMake(0.5, 0.5, WIN_WIDTH - 1, WIN_HEIGHT - 1),
            CORNER_RADIUS, CORNER_RADIUS, None
        )
        edge_mask.setPath_(edge_path)
        edge_mask.setFillColor_(None)
        edge_mask.setStrokeColor_(NSColor.whiteColor().CGColor())
        edge_mask.setLineWidth_(1.0)
        specular.setMask_(edge_mask)

        highlight_view.layer().addSublayer_(specular)
        content.addSubview_(highlight_view)
        self._highlight_view = highlight_view
        self._specular = specular

        # ── Ndot headline ──
        # dot_text covers full window so scattered dots can fly behind other views
        self.dot_text = DotTextView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WIN_WIDTH, WIN_HEIGHT)
        )
        self.dot_text._text_y_center = 203  # upper portion of window
        self.dot_text.setText_("Say the word...")
        content.addSubview_(self.dot_text)

        # ── Status label (secondary info below the dots) ──
        self.status_label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(20, 142, WIN_WIDTH - 40, 22)
        )
        self.status_label.setStringValue_("")
        self.status_label.setBezeled_(False)
        self.status_label.setDrawsBackground_(False)
        self.status_label.setEditable_(False)
        self.status_label.setSelectable_(False)
        self.status_label.setFont_(NSFont.systemFontOfSize_weight_(12, NSFontWeightRegular))
        self.status_label.setAlignment_(NSCenterTextAlignment)
        self.status_label.setTextColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(*TEXT_SECONDARY)
        )
        content.addSubview_(self.status_label)

        # ── Language dropdown ──
        tog_w = 112
        tog_h = 26
        self.lang_toggle = LangDropdownView.alloc().initWithFrame_(
            NSMakeRect((WIN_WIDTH - tog_w) / 2, 126, tog_w, tog_h)
        )
        self.lang_toggle.setCallback_(self._on_lang_select)
        content.addSubview_(self.lang_toggle)

        # ── Hands-Free toggle ──
        hf_w = 112
        hf_h = 24
        self.hf_toggle = HandsFreeToggleView.alloc().initWithFrame_(
            NSMakeRect((WIN_WIDTH - hf_w) / 2, 94, hf_w, hf_h)
        )
        self.hf_toggle.setCallback_(self._on_hands_free_toggle)
        content.addSubview_(self.hf_toggle)

        # ── Mini mode toggle (top-right corner) ──
        mini_btn_sz = 22
        self.mini_toggle = MiniToggleView.alloc().initWithFrame_(
            NSMakeRect(WIN_WIDTH - mini_btn_sz - 12, WIN_HEIGHT - mini_btn_sz - 12,
                       mini_btn_sz, mini_btn_sz)
        )
        self.mini_toggle.setCallback_(self._toggle_mini_mode)
        content.addSubview_(self.mini_toggle)

        # ── History (last 3 transcriptions, clickable to copy) ──
        self.history = []
        self.history_labels = []
        for i in range(3):
            y_pos = 48 - (i * 18)
            label = ClickableLabel.alloc().initWithFrame_(
                NSMakeRect(16, y_pos, WIN_WIDTH - 32, 18)
            )
            label.setStringValue_("—")
            label.setBezeled_(False)
            label.setDrawsBackground_(False)
            label.setEditable_(False)
            label.setSelectable_(False)
            label.setFont_(NSFont.systemFontOfSize_weight_(11, NSFontWeightRegular))
            label.setAlignment_(NSCenterTextAlignment)
            label.setTextColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(*TEXT_SECONDARY))
            label.setLineBreakMode_(NSLineBreakByTruncatingTail)
            label.setClickCallback_(self.copy_to_clipboard)
            content.addSubview_(label)
            self.history_labels.append(label)

        # Store refs to full-mode-only views (dot_text NOT included — it stays visible always)
        self._full_mode_views = [
            self.status_label,
            self.lang_toggle, self.hf_toggle,
        ] + self.history_labels

        # Store full-mode and mini-mode frames for dot_text
        self._dot_text_full_frame = (0, 0, WIN_WIDTH, WIN_HEIGHT)
        self._dot_text_mini_frame = (0, 0, MINI_WIDTH, MINI_HEIGHT)

        self.window.makeKeyAndOrderFront_(None)

    def _update_highlight_edge(self, w, h, radius):
        """Rebuild the specular highlight mask for a given size and corner radius."""
        from Quartz import CAShapeLayer
        self._specular.setFrame_(Quartz.CGRectMake(0, 0, w, h))
        mask = CAShapeLayer.alloc().init()
        path = Quartz.CGPathCreateWithRoundedRect(
            Quartz.CGRectMake(0.5, 0.5, w - 1, h - 1), radius, radius, None
        )
        mask.setPath_(path)
        mask.setFillColor_(None)
        mask.setStrokeColor_(NSColor.whiteColor().CGColor())
        mask.setLineWidth_(1.0)
        self._specular.setMask_(mask)

    def _toggle_mini_mode(self):
        if not self._mini_lock.acquire(blocking=False):
            return  # already animating, ignore
        self._is_mini = not self._is_mini
        self.mini_toggle.setMini_(self._is_mini)

        frame = self.window.frame()
        ANIM_DURATION = 0.18

        if self._is_mini:
            # ── Collapse to mini pill — always top-center ──
            screen = AppKit.NSScreen.mainScreen().frame()
            new_w = MINI_WIDTH
            new_h = MINI_HEIGHT
            new_x = (screen.size.width - new_w) / 2
            new_y = screen.size.height - new_h - 40

            def _do_collapse():
              try:
                # Dissolve dots + fade out other views simultaneously
                def _start():
                    AppKit.NSAnimationContext.beginGrouping()
                    AppKit.NSAnimationContext.currentContext().setDuration_(0.25)
                    for v in self._full_mode_views:
                        v.animator().setAlphaValue_(0.0)
                    AppKit.NSAnimationContext.endGrouping()
                self.on_main(_start)
                self.dot_text.dissolve()
                deadline = time.time() + 1.0
                while self.dot_text._visible_dots > 0 and time.time() < deadline:
                    time.sleep(0.01)

                # Start resize — hide highlight during transition
                def _resize():
                    self._highlight_view.setHidden_(True)
                    for v in self._full_mode_views:
                        v.setHidden_(True)
                        v.setAlphaValue_(1.0)

                    self.window.contentView().layer().setCornerRadius_(new_h / 2)
                    self._tint_view.layer().setCornerRadius_(new_h / 2)
                    self._vibrancy.layer().setCornerRadius_(new_h / 2)

                    AppKit.NSAnimationContext.beginGrouping()
                    AppKit.NSAnimationContext.currentContext().setDuration_(ANIM_DURATION)
                    self.window.animator().setFrame_display_(
                        NSMakeRect(new_x, new_y, new_w, new_h), True
                    )
                    self.mini_toggle.animator().setFrame_(NSMakeRect(
                        new_w - 34, (new_h - 22) / 2, 22, 22
                    ))
                    AppKit.NSAnimationContext.endGrouping()
                self.on_main(_resize)
                time.sleep(ANIM_DURATION + 0.05)

                # Reveal in mini — show highlight with new mask
                def _reveal():
                    self._update_highlight_edge(new_w, new_h, new_h / 2)
                    self._highlight_view.setHidden_(False)
                    self.dot_text.setFrame_(NSMakeRect(*self._dot_text_mini_frame))
                    self.dot_text._font_size = 18
                    self.dot_text._font = NSFont.fontWithName_size_("Ndot 55", 18) or \
                                           NSFont.fontWithName_size_("Ndot 57 Aligned", 18) or \
                                           NSFont.systemFontOfSize_(18)
                    self.dot_text._text_y_center = None
                    self.dot_text.setText_(self._mini_text_for_state())
                    self.dot_text.reveal()
                self.on_main(_reveal)
              finally:
                self._mini_lock.release()
            threading.Thread(target=_do_collapse, daemon=True).start()

        else:
            # ── Expand to full — always top-center ──
            screen = AppKit.NSScreen.mainScreen().frame()
            new_w = WIN_WIDTH
            new_h = WIN_HEIGHT
            new_x = (screen.size.width - new_w) / 2
            new_y = screen.size.height - new_h - 40

            def _do_expand():
              try:
                # Dissolve dots
                self.dot_text.dissolve()
                deadline = time.time() + 1.0
                while self.dot_text._visible_dots > 0 and time.time() < deadline:
                    time.sleep(0.01)

                # Resize — hide highlight during transition
                def _resize():
                    self._highlight_view.setHidden_(True)
                    self.window.contentView().layer().setCornerRadius_(CORNER_RADIUS)
                    self.window.contentView().layer().setCornerCurve_("continuous")
                    self._tint_view.layer().setCornerRadius_(CORNER_RADIUS)
                    self._tint_view.layer().setCornerCurve_("continuous")
                    self._vibrancy.layer().setCornerRadius_(CORNER_RADIUS)
                    self._vibrancy.layer().setCornerCurve_("continuous")

                    for v in self._full_mode_views:
                        v.setAlphaValue_(0.0)
                        v.setHidden_(False)

                    AppKit.NSAnimationContext.beginGrouping()
                    AppKit.NSAnimationContext.currentContext().setDuration_(ANIM_DURATION)
                    self.window.animator().setFrame_display_(
                        NSMakeRect(new_x, new_y, new_w, new_h), True
                    )
                    self.mini_toggle.animator().setFrame_(NSMakeRect(
                        WIN_WIDTH - 34, WIN_HEIGHT - 34, 22, 22
                    ))
                    AppKit.NSAnimationContext.endGrouping()
                self.on_main(_resize)
                time.sleep(ANIM_DURATION + 0.05)

                # Reveal in full — show highlight with new mask
                def _reveal():
                    self._update_highlight_edge(new_w, new_h, CORNER_RADIUS)
                    self._highlight_view.setHidden_(False)
                    self.dot_text.setFrame_(NSMakeRect(*self._dot_text_full_frame))
                    self.dot_text._font_size = 24
                    self.dot_text._font = NSFont.fontWithName_size_("Ndot 55", 24) or \
                                           NSFont.fontWithName_size_("Ndot 57 Aligned", 24) or \
                                           NSFont.systemFontOfSize_(24)
                    self.dot_text._text_y_center = 203
                    self.dot_text.setText_(self._full_text_for_state())
                    self.dot_text.reveal()

                    AppKit.NSAnimationContext.beginGrouping()
                    AppKit.NSAnimationContext.currentContext().setDuration_(0.2)
                    for v in self._full_mode_views:
                        v.animator().setAlphaValue_(1.0)
                    AppKit.NSAnimationContext.endGrouping()
                self.on_main(_reveal)
              finally:
                self._mini_lock.release()
            threading.Thread(target=_do_expand, daemon=True).start()

    def _mini_text_for_state(self):
        """Get the appropriate short text for mini mode based on current state."""
        if self.recording:
            return "Recording..."
        if self.hands_free_mode:
            return "Clap twice..."
        return "Say the word..."

    def _full_text_for_state(self):
        """Get the appropriate full text based on current state."""
        if self.recording:
            return "Recording..."
        if self.hands_free_mode:
            return "Clap twice..."
        return "Say the word..."

    def make_label(self, text, frame, size=13, weight=NSFontWeightRegular,
                   alignment=NSCenterTextAlignment, color=None):
        label = NSTextField.alloc().initWithFrame_(frame)
        label.setStringValue_(text)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setFont_(NSFont.systemFontOfSize_weight_(size, weight))
        label.setAlignment_(alignment)
        if color:
            label.setTextColor_(color)
        return label

    def _on_lang_select(self, code):
        self.language = code
        logging.info(f"Language switched to: {self.language}")

    def _on_hands_free_toggle(self, active):
        self.hands_free_mode = active
        logging.info(f"Hands-free mode: {active}")
        # Write voice mode for Claude Code integration
        try:
            voice_dir = os.path.expanduser("~/.config/claude-voice")
            os.makedirs(voice_dir, exist_ok=True)
            with open(os.path.join(voice_dir, "voice_mode"), "w") as f:
                f.write("on" if active else "off")
            # Copy bundled speak.sh if not already present
            speak_dst = os.path.join(voice_dir, "speak.sh")
            if not os.path.exists(speak_dst):
                speak_src = os.path.join(SCRIPT_DIR, "speak.sh")
                if os.path.exists(speak_src):
                    import shutil
                    shutil.copy2(speak_src, speak_dst)
                    os.chmod(speak_dst, 0o755)
        except Exception:
            logging.exception("Could not update voice_mode file")
        if active:
            def _():
                self.dot_text.setText_("Clap twice...")
            self.on_main(_)
            self._stop_clap_monitor.clear()
            threading.Thread(target=self._clap_monitor, daemon=True).start()
        else:
            self._stop_clap_monitor.set()
            def _():
                self.dot_text.setText_("Say the word...")
                self.status_label.setStringValue_("")
            self.on_main(_)

    def _clap_monitor(self):
        """Single persistent stream for clap detection — handles both start and stop."""
        CHUNK = 512
        THRESHOLD = 0.15
        MIN_CLAP_GAP = 0.2
        DOUBLE_CLAP_WINDOW = 0.9
        MIN_RECORD_BEFORE_STOP = 1.0
        POST_TRANSCRIBE_COOLDOWN = 1.5  # ignore claps after transcription finishes

        clap_times = []
        last_clap_time = 0.0
        record_start_time = [0.0]
        last_transcribe_end = [0.0]  # track when transcription finishes

        def audio_cb(indata, frames, time_info, status):
            nonlocal last_clap_time, clap_times
            if self._stop_clap_monitor.is_set():
                return

            peak = float(np.max(np.abs(indata)))
            now = time.time()

            if self.recording and (now - record_start_time[0]) < MIN_RECORD_BEFORE_STOP:
                return

            # Cooldown after transcription finishes to avoid phantom re-triggers
            if not self.recording and (now - last_transcribe_end[0]) < POST_TRANSCRIBE_COOLDOWN:
                return

            if peak < THRESHOLD or (now - last_clap_time) < MIN_CLAP_GAP:
                return

            last_clap_time = now
            clap_times.append(now)
            clap_times[:] = [t for t in clap_times if now - t <= DOUBLE_CLAP_WINDOW]
            logging.info(f"Clap detected — peak={peak:.3f}, count={len(clap_times)}, recording={self.recording}")

            if len(clap_times) == 1:
                if not self.recording:
                    if subprocess.run(["pgrep", "-x", "afplay"], capture_output=True).returncode == 0:
                        subprocess.run(["pkill", "-x", "afplay"], capture_output=True)
                        clap_times.clear()
                        logging.info("Single clap — interrupted Matilda")
                        return
                    def _(): self.dot_text.setText_("Clap again...")
                    self.on_main(_)
                    def _reset(w=DOUBLE_CLAP_WINDOW):
                        time.sleep(w + 0.1)
                        if self.hands_free_mode and not self.recording:
                            def __(): self.dot_text.setText_("Clap twice...")
                            self.on_main(__)
                    threading.Thread(target=_reset, daemon=True).start()
                else:
                    def _(): self.dot_text.setText_("Clap again...")
                    self.on_main(_)
                    def _reset(w=DOUBLE_CLAP_WINDOW):
                        time.sleep(w + 0.1)
                        if self.recording and self.hands_free_mode:
                            def __(): self.dot_text.setText_("Clap twice...")
                            self.on_main(__)
                    threading.Thread(target=_reset, daemon=True).start()

            elif len(clap_times) >= 2:
                clap_times.clear()
                if not self.recording:
                    record_start_time[0] = time.time()
                    def _(): self.dot_text.setText_("Clap twice...")
                    self.on_main(_)
                    threading.Thread(target=self.start_recording, daemon=True).start()
                else:
                    def _stop_and_mark():
                        self.stop_and_transcribe()
                        last_transcribe_end[0] = time.time()
                    threading.Thread(target=_stop_and_mark, daemon=True).start()

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                                blocksize=CHUNK, callback=audio_cb):
                logging.info("Clap monitor stream open")
                while not self._stop_clap_monitor.is_set():
                    time.sleep(0.05)
        except Exception:
            logging.exception("Clap monitor error")

    # ── UI updates (main thread) ────────────────────────────────────────

    def copy_to_clipboard(self, text):
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
        def _():
            self.status_label.setStringValue_("Copied!")
            self.perform_selector_delayed("_reset_status")
        self.on_main(_)

    def _reset_status(self):
        def _():
            if not self.recording:
                self.status_label.setStringValue_("")
                if self.hands_free_mode:
                    self.dot_text.setText_("Clap twice...")
                else:
                    self.dot_text.setText_("Say the word...")
        self.on_main(_)

    def perform_selector_delayed(self, sel_name):
        def _delayed():
            time.sleep(1.5)
            self._reset_status()
        threading.Thread(target=_delayed, daemon=True).start()

    def set_state_idle(self):
        self._animating_waveform = False
        def _():
            self.dot_text.setAnimating_(False)
            self.dot_text._spread = 0.0
            self.dot_text._target_spread = 0.0
            self.dot_text.setColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(*TEXT_PRIMARY)
            )
            if self.hands_free_mode:
                self.dot_text.setText_("Clap twice...")
            else:
                self.dot_text.setText_("Say the word...")
            self.status_label.setStringValue_("")
        self.on_main(_)

    def set_state_recording(self):
        red = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.25, 0.25, 1.0)
        def _():
            self.dot_text.setColor_(red)
            self.dot_text.setText_("Recording...")
            self.dot_text.setAnimating_(True)
            self.status_label.setStringValue_("")
        self.on_main(_)
        self._animating_waveform = True
        threading.Thread(target=self._animate_waveform, daemon=True).start()

    def set_state_transcribing(self):
        self._animating_waveform = False
        yellow = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.95, 0.80, 0.20, 1.0)
        def _():
            self.dot_text.setAnimating_(False)
            self.dot_text.setColor_(yellow)
            self.dot_text.setText_("Transcribing...")
            self.status_label.setStringValue_("")
        self.on_main(_)

    def _animate_waveform(self):
        """Animate mini waveform bars and drive dot text scatter from audio level."""
        while self._animating_waveform:
            if self.audio_frames:
                latest = self.audio_frames[-1].flatten().astype(np.float32)
                chunk_size = max(1, len(latest) // WaveformView.NUM_BARS)
                levels = []
                for i in range(WaveformView.NUM_BARS):
                    start = i * chunk_size
                    end = min(start + chunk_size, len(latest))
                    if start < len(latest):
                        rms = np.sqrt(np.mean(latest[start:end] ** 2)) / 32768.0
                        level = min(1.0, rms * 50)
                    else:
                        level = 0.0
                    levels.append(level)
            else:
                levels = [0.05] * WaveformView.NUM_BARS

            # Average audio energy drives scatter spread
            avg_level = sum(levels) / len(levels)

            def _update(lvls=levels, spread=avg_level):
                # Set target spread based on volume (louder = more scatter)
                self.dot_text.setSpread_(spread)
                self.dot_text.updateAnimation()
            self.on_main(_update)
            time.sleep(0.05)  # ~20fps

        # After recording stops, keep updating for smooth settle
        def _settle():
            while self.dot_text._spread > 0.005:
                def _():
                    self.dot_text.updateAnimation()
                self.on_main(_)
                time.sleep(0.05)
            def _final():
                self.dot_text._spread = 0.0
                self.dot_text._animating = False
                self.dot_text.setNeedsDisplay_(True)
            self.on_main(_final)
        threading.Thread(target=_settle, daemon=True).start()

    def set_last_text(self, text):
        self.history.insert(0, text)
        self.history = self.history[:3]
        def _():
            for i, label in enumerate(self.history_labels):
                if i < len(self.history):
                    t = self.history[i]
                    display = t[:35] + "…" if len(t) > 35 else t
                    label.setStringValue_(display)
                    label.setFullText_(t)
                else:
                    label.setStringValue_("—")
                    label.setFullText_("")
        self.on_main(_)

    def on_main(self, func):
        AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(func)

    # ── Audio pause/resume ──────────────────────────────────────────────

    def get_playing_sources(self):
        sources = []
        for app_name in ["Spotify", "Music"]:
            try:
                running = subprocess.run(
                    ["osascript", "-e", f'tell application "System Events" to get (name of processes) contains "{app_name}"'],
                    capture_output=True, text=True, timeout=2,
                )
                if running.stdout.strip() != "true":
                    continue
                result = subprocess.run(
                    ["osascript", "-e", f'tell application "{app_name}" to player state as string'],
                    capture_output=True, text=True, timeout=2,
                )
                if result.stdout.strip() == "playing":
                    sources.append(("app", app_name))
            except Exception:
                pass

        for browser in ["Google Chrome", "Google Chrome Canary", "Brave Browser", "Arc"]:
            try:
                result = subprocess.run(
                    ["osascript", "-e", f'tell application "System Events" to get (name of processes) contains "{browser}"'],
                    capture_output=True, text=True, timeout=2,
                )
                if result.stdout.strip() != "true":
                    continue
                js_check = f'''
                    tell application "{browser}"
                        repeat with w in windows
                            repeat with t in tabs of w
                                set isPlaying to execute t javascript "
                                    (function() {{
                                        var v = document.querySelectorAll('video, audio');
                                        var playing = false;
                                        v.forEach(function(el) {{
                                            if (!el.paused) {{ playing = true; }}
                                        }});
                                        return playing;
                                    }})()"
                                if isPlaying is "true" then return "playing"
                            end repeat
                        end repeat
                    end tell'''
                r = subprocess.run(["osascript", "-e", js_check], capture_output=True, text=True, timeout=3)
                if r.stdout.strip() == "playing":
                    sources.append(("browser", browser))
            except Exception:
                pass

        try:
            result = subprocess.run(
                ["osascript", "-e", 'tell application "System Events" to get (name of processes) contains "Safari"'],
                capture_output=True, text=True, timeout=2,
            )
            if result.stdout.strip() == "true":
                js_check = '''
                    tell application "Safari"
                        repeat with w in windows
                            repeat with t in tabs of w
                                set isPlaying to do JavaScript "
                                    (function() {
                                        var v = document.querySelectorAll('video, audio');
                                        var playing = false;
                                        v.forEach(function(el) {
                                            if (!el.paused) { playing = true; }
                                        });
                                        return playing;
                                    })()" in t
                                if isPlaying is "true" then return "playing"
                            end repeat
                        end repeat
                    end tell'''
                r = subprocess.run(["osascript", "-e", js_check], capture_output=True, text=True, timeout=3)
                if r.stdout.strip() == "playing":
                    sources.append(("browser", "Safari"))
        except Exception:
            pass
        return sources

    def _pause_browser(self, browser):
        """Pause audio/video in a specific browser."""
        if browser == "Safari":
            js = 'tell application "Safari" to repeat with w in windows\nrepeat with t in tabs of w\ndo JavaScript "document.querySelectorAll(\'video, audio\').forEach(function(el) { if (!el.paused) el.pause(); })" in t\nend repeat\nend repeat'
        else:
            js = f'tell application "{browser}" to repeat with w in windows\nrepeat with t in tabs of w\nexecute t javascript "document.querySelectorAll(\'video, audio\').forEach(function(el) {{ if (!el.paused) el.pause(); }})"\nend repeat\nend repeat'
        subprocess.Popen(["osascript", "-e", js], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def pause_all_audio(self):
        sources = self.get_playing_sources()
        for kind, name in sources:
            if kind == "app":
                subprocess.Popen(["osascript", "-e", f'tell application "{name}" to pause'],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif kind == "browser":
                self._pause_browser(name)
        return sources

    def resume_all_audio(self):
        if not self.paused_sources:
            return
        for kind, name in self.paused_sources:
            if kind == "app":
                subprocess.Popen(["osascript", "-e", f'tell application "{name}" to play'],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif kind == "browser":
                if name == "Safari":
                    js = 'tell application "Safari" to repeat with w in windows\nrepeat with t in tabs of w\ndo JavaScript "document.querySelectorAll(\'video, audio\').forEach(function(el) { el.play(); })" in t\nend repeat\nend repeat'
                else:
                    js = f'tell application "{name}" to repeat with w in windows\nrepeat with t in tabs of w\nexecute t javascript "document.querySelectorAll(\'video, audio\').forEach(function(el) {{ el.play(); }})"\nend repeat\nend repeat'
                subprocess.Popen(["osascript", "-e", js], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.paused_sources = []

    # ── Recording ───────────────────────────────────────────────────────

    def audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            self.audio_frames.append(indata.copy())

    def start_recording(self):
        try:
            logging.info("Fn pressed — recording")
            # Stop clap monitor stream before opening recording stream
            # to avoid two concurrent streams on the same mic
            if self.hands_free_mode:
                self._stop_clap_monitor.set()
                time.sleep(0.05)  # let the clap stream close
            self.audio_frames = []
            self.recording = True
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS,
                dtype="int16", callback=self.audio_callback,
            )
            self.stream.start()
            self.set_state_recording()
            threading.Thread(target=self._pause_bg, daemon=True).start()
        except sd.PortAudioError:
            logging.exception("Microphone access denied or unavailable")
            self.recording = False
            def _warn():
                self.dot_text.setText_("No mic")
                self.status_label.setStringValue_(
                    "Grant Microphone access in System Settings"
                )
            self.on_main(_warn)
        except Exception:
            logging.exception("Error starting recording")
            self.recording = False

    def _pause_bg(self):
        self.paused_sources = self.pause_all_audio()

    def stop_and_transcribe(self):
        try:
            logging.info("Fn released — transcribing")
            self.recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            self.set_state_transcribing()

            if not self.audio_frames:
                logging.info("No audio")
                self.set_state_idle()
                self.resume_all_audio()
                return

            audio_data = np.concatenate(self.audio_frames, axis=0)
            duration = len(audio_data) / SAMPLE_RATE
            logging.info(f"Audio: {duration:.1f}s")

            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            logging.info(f"Audio RMS: {rms:.1f}")
            if rms < 100:
                logging.info("Audio too quiet, skipping transcription")
                self.set_state_idle()
                self.resume_all_audio()
                return

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                with wave.open(tmp.name, "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio_data.tobytes())

                whisper_args = [WHISPER_CMD, "-m", MODEL_PATH, "-f", tmp.name, "--no-timestamps",
                     "-l", self.language]
                if HINT_WORDS:
                    whisper_args += ["--prompt", ", ".join(HINT_WORDS)]
                result = subprocess.run(
                    whisper_args,
                    capture_output=True, stdin=subprocess.DEVNULL, text=True, timeout=30,
                )

                logging.info(f"whisper stdout: {result.stdout!r}")

                text = result.stdout.strip().strip("[] \n")

                PROMPT_WORDS = {w.lower() for w in HINT_WORDS} if HINT_WORDS else set()
                text_words = set(text.lower().replace(",", "").replace(".", "").split())
                if text_words.issubset(PROMPT_WORDS):
                    logging.info(f"Filtered prompt hallucination: {text}")
                    self.set_state_idle()
                    self.resume_all_audio()
                    return

                if not text or text.lower() in ("(blank audio)", "[blank_audio]", ""):
                    logging.info("No speech detected")
                    self.set_state_idle()
                    self.resume_all_audio()
                    return

                logging.info(f"Transcribed: {text}")
                self.set_last_text(text)

                saved = self._clipboard_save()
                subprocess.run(["pbcopy"], input=text.encode(), check=True)
                self.simulate_paste()
                if self.hands_free_mode:
                    self.simulate_return()
                    time.sleep(0.8)  # let paste + Return land before restoring
                else:
                    time.sleep(0.5)  # give target app time to process paste
                self._clipboard_restore(saved)

            finally:
                os.unlink(tmp.name)
                self.resume_all_audio()
                self.set_state_idle()

        except Exception:
            logging.exception("Error in transcription")
            self.set_state_idle()
        finally:
            # Restart clap monitor if hands-free is still active
            if self.hands_free_mode:
                self._stop_clap_monitor.clear()
                threading.Thread(target=self._clap_monitor, daemon=True).start()

    def _clipboard_save(self):
        pb = AppKit.NSPasteboard.generalPasteboard()
        saved = {}
        for t in (pb.types() or []):
            data = pb.dataForType_(t)
            if data:
                saved[t] = data
        return saved

    def _clipboard_restore(self, saved):
        if not saved:
            return
        pb = AppKit.NSPasteboard.generalPasteboard()
        pb.clearContents()
        for t, data in saved.items():
            pb.setData_forType_(data, t)

    def simulate_paste(self):
        source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
        key_down = CGEventCreateKeyboardEvent(source, 9, True)
        key_up = CGEventCreateKeyboardEvent(source, 9, False)
        CGEventSetFlags(key_down, kCGEventFlagMaskCommand)
        CGEventSetFlags(key_up, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, key_down)
        CGEventPost(kCGHIDEventTap, key_up)

    def simulate_return(self):
        time.sleep(0.12)
        source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
        key_down = CGEventCreateKeyboardEvent(source, 36, True)
        key_up = CGEventCreateKeyboardEvent(source, 36, False)
        CGEventPost(kCGHIDEventTap, key_down)
        CGEventPost(kCGHIDEventTap, key_up)

    # ── Fn key event tap ────────────────────────────────────────────────

    def setup_event_tap(self):
        fn_held = False
        app_ref = self

        def callback(proxy, event_type, event, refcon):
            nonlocal fn_held
            flags = CGEventGetFlags(event)
            fn_now = bool(flags & FN_FLAG)

            if fn_now and not fn_held:
                fn_held = True
                app_ref.start_recording()
            elif not fn_now and fn_held:
                fn_held = False
                threading.Thread(target=app_ref.stop_and_transcribe, daemon=True).start()

            return event

        event_mask = CGEventMaskBit(kCGEventFlagsChanged)
        self._event_tap_callback = callback  # prevent GC
        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            event_mask,
            callback,
            None,
        )

        if tap is None:
            logging.error("Could not create event tap — Accessibility permission missing")
            def _warn():
                self.dot_text.setText_("No access")
                self.status_label.setStringValue_(
                    "Grant Accessibility in System Settings"
                )
            self.on_main(_warn)
            return

        self._event_tap = tap  # prevent GC
        run_loop_source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        self._event_tap_source = run_loop_source  # prevent GC
        Quartz.CFRunLoopAddSource(
            Quartz.CFRunLoopGetMain(), run_loop_source, Quartz.kCFRunLoopCommonModes
        )
        Quartz.CGEventTapEnable(tap, True)
        logging.info("Event tap active")

    def run(self):
        self.app.run()


if __name__ == "__main__":
    # Override the bundle info so macOS shows "Say the word..." in menu bar
    bundle = AppKit.NSBundle.mainBundle()
    info = bundle.infoDictionary()
    info["CFBundleName"] = "Say the word..."
    info["CFBundleDisplayName"] = "Say the word..."

    AppKit.NSProcessInfo.processInfo().setProcessName_("Say the word...")

    # Set menu bar
    menubar = AppKit.NSMenu.alloc().init()
    app_menu_item = AppKit.NSMenuItem.alloc().init()
    menubar.addItem_(app_menu_item)
    NSApplication.sharedApplication().setMainMenu_(menubar)
    app_menu = AppKit.NSMenu.alloc().initWithTitle_("Say the word...")
    quit_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "Quit Voice to Text", "terminate:", "q"
    )
    app_menu.addItem_(quit_item)
    app_menu_item.setSubmenu_(app_menu)

    # First-run setup — install whisper-cli and download model if needed
    _first_run_setup()
    # Re-discover whisper-cli in case it was just installed
    if not WHISPER_CMD:
        WHISPER_CMD = _find_whisper_cli()

    # Kill any existing instances — verify it's actually voice_app before killing
    import signal
    my_pid = os.getpid()
    pid_file = os.path.join(SCRIPT_DIR, ".voice_app.pid")
    if os.path.exists(pid_file):
        try:
            old_pid = int(open(pid_file).read().strip())
            if old_pid != my_pid:
                # Verify the PID is actually a voice_app process before killing
                check = subprocess.run(
                    ["ps", "-p", str(old_pid), "-o", "command="],
                    capture_output=True, text=True,
                )
                if "voice_app" in check.stdout:
                    os.kill(old_pid, signal.SIGTERM)
                    time.sleep(0.5)
        except (ProcessLookupError, ValueError):
            pass
    with open(pid_file, "w") as f:
        f.write(str(my_pid))

    try:
        voice_app = VoiceToTextApp()
        voice_app.run()
    except Exception as e:
        logging.exception("Fatal error")
        print(f"Fatal error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
