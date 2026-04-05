#!/usr/bin/env python3
"""
Voice to Text — minimal floating macOS window.
Hold Fn to record, release to transcribe and paste.
Frosted glass UI inspired by iOS/visionOS.
"""

import os
import sys
import json
import wave
import tempfile
import subprocess
import threading
import logging

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

# ── Config ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

LANG_LABELS = CONFIG["languages"]                # e.g. ["EN", "DA"] or ["EN", "SE"]
WHISPER_CODES = CONFIG.get("whisper_codes", {})   # e.g. {"EN": "en", "DA": "da", "SE": "sv"}
HINT_WORDS = CONFIG.get("hint_words", [])

MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "ggml-small.bin")
WHISPER_CMD = (
    subprocess.run(["which", "whisper-cli"], capture_output=True, text=True).stdout.strip()
    or "/opt/homebrew/bin/whisper-cli"
)
SAMPLE_RATE = 16000
CHANNELS = 1
FN_FLAG = 1 << 23

WIN_WIDTH = 280
WIN_HEIGHT = 276

# ── Logging ─────────────────────────────────────────────────────────────
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


# ── Waveform view ───────────────────────────────────────────────────────
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
        self._color = NSColor.systemRedColor()
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


# ── Language toggle (single click area, sliding indicator) ─────────────
class LangToggleView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(LangToggleView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._selected = 0
        self._labels = LANG_LABELS
        self._callback = None
        return self

    def setCallback_(self, cb):
        self._callback = cb

    def selectedIndex(self):
        return self._selected

    def mouseDown_(self, event):
        # Toggle on any click anywhere in the view
        self._selected = 1 - self._selected
        self.setNeedsDisplay_(True)
        if self._callback:
            self._callback(self._selected)

    def resetCursorRects(self):
        self.addCursorRect_cursor_(self.bounds(), AppKit.NSCursor.pointingHandCursor())

    def drawRect_(self, rect):
        bounds = self.bounds()
        w = bounds.size.width
        h = bounds.size.height
        half_w = w / 2
        r = h / 2

        # Track background
        track = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(bounds, r, r)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.08).setFill()
        track.fill()

        # Sliding indicator pill
        pad = 2
        pill_w = half_w - pad
        pill_h = h - pad * 2
        pill_x = pad + self._selected * (half_w)
        pill_r = pill_h / 2
        pill_rect = NSMakeRect(pill_x, pad, pill_w, pill_h)
        pill = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(pill_rect, pill_r, pill_r)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.22).setFill()
        pill.fill()

        # Labels
        for i, label in enumerate(self._labels):
            is_active = (i == self._selected)
            color = NSColor.whiteColor() if is_active else NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.45)
            font = NSFont.systemFontOfSize_weight_(11, NSFontWeightSemibold if is_active else NSFontWeightRegular)
            attrs = {
                AppKit.NSFontAttributeName: font,
                AppKit.NSForegroundColorAttributeName: color,
            }
            attr_str = AppKit.NSAttributedString.alloc().initWithString_attributes_(label, attrs)
            size = attr_str.size()
            x = i * half_w + (half_w - size.width) / 2
            y = (h - size.height) / 2
            attr_str.drawAtPoint_(AppKit.NSMakePoint(x, y))


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
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.08).setFill()
        track.fill()

        if self._active:
            # Full highlight pill when on
            pad = 2
            pill_rect = NSMakeRect(pad, pad, w - pad * 2, h - pad * 2)
            pill_r = (h - pad * 2) / 2
            pill = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                pill_rect, pill_r, pill_r
            )
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.22).setFill()
            pill.fill()

        # Label
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


class VoiceToTextApp:
    def __init__(self):
        self.recording = False
        self.audio_frames = []
        self.stream = None
        self.paused_sources = []
        self.language = "en"
        self.hands_free_mode = False
        self._stop_clap_monitor = threading.Event()

        self.app = NSApplication.sharedApplication()
        self.app.setActivationPolicy_(0)  # NSApplicationActivationPolicyRegular — shows dock dot

        self.build_window()
        self.set_dock_icon()
        self.setup_event_tap()

    def set_dock_icon(self):
        size = 256
        icon = AppKit.NSImage.alloc().initWithSize_((size, size))
        icon.lockFocus()

        # macOS icon sizing: 80% of canvas
        inset = size * 0.1
        icon_sz = size * 0.8
        radius = icon_sz * 0.22
        rect = NSMakeRect(inset, inset, icon_sz, icon_sz)
        shape = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, radius, radius)

        # Gradient background
        top_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.18, 0.17, 0.22, 1.0)
        bot_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.08, 0.08, 0.10, 1.0)
        gradient = AppKit.NSGradient.alloc().initWithStartingColor_endingColor_(bot_color, top_color)
        gradient.drawInBezierPath_angle_(shape, 90)

        # Edge highlight
        AppKit.NSGraphicsContext.currentContext().saveGraphicsState()
        shape.addClip()
        inner_rect = NSMakeRect(inset + 0.5, inset + 0.5, icon_sz - 1, icon_sz - 1)
        inner_shape = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(inner_rect, radius - 0.5, radius - 0.5)
        hl_top = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.25)
        hl_bot = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.05)
        stroke_grad = AppKit.NSGradient.alloc().initWithStartingColor_endingColor_(hl_bot, hl_top)
        stroke_grad.drawInBezierPath_angle_(inner_shape, 90)
        fill_rect = NSMakeRect(inset + 1.5, inset + 1.5, icon_sz - 3, icon_sz - 3)
        fill_shape = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(fill_rect, radius - 1.5, radius - 1.5)
        gradient2 = AppKit.NSGradient.alloc().initWithStartingColor_endingColor_(bot_color, top_color)
        gradient2.drawInBezierPath_angle_(fill_shape, 90)
        AppKit.NSGraphicsContext.currentContext().restoreGraphicsState()

        # 7 dots in a line
        num = 7
        dot_r = icon_sz * 0.035
        gap = icon_sz * 0.045
        total_w = num * (dot_r * 2) + (num - 1) * gap
        start_x = inset + (icon_sz - total_w) / 2
        cy = inset + icon_sz / 2

        NSColor.whiteColor().setFill()
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
        x = screen.size.width - WIN_WIDTH - 20
        y = screen.size.height - WIN_HEIGHT - 60

        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                 NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskFullSizeContentView)

        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, WIN_WIDTH, WIN_HEIGHT),
            style,
            NSBackingStoreBuffered,
            False,
        )

        self.window.setLevel_(NSFloatingWindowLevel)
        self.window.setTitle_("Voice to Text")
        self.window.setTitleVisibility_(1)  # NSWindowTitleHidden
        self.window.setTitlebarAppearsTransparent_(True)
        self.window.setMovableByWindowBackground_(True)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(NSColor.clearColor())
        self.window.setHasShadow_(True)
        self.window.setAlphaValue_(0.95)

        # ── Liquid Glass background ──
        content = self.window.contentView()
        content.setWantsLayer_(True)
        content.layer().setCornerRadius_(22)
        content.layer().setMasksToBounds_(True)

        # Base tint — very subtle warm-neutral fill behind the blur
        # so the glass has body even over dark backgrounds
        tint_view = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, WIN_WIDTH, WIN_HEIGHT))
        tint_view.setWantsLayer_(True)
        tint_view.layer().setCornerRadius_(22)
        tint_view.layer().setMasksToBounds_(True)
        tint_view.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.45, 0.45, 0.48, 0.15).CGColor()
        )
        content.addSubview_(tint_view)

        # Primary blur — light, translucent material that refracts behind-window content
        vibrancy = NSVisualEffectView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WIN_WIDTH, WIN_HEIGHT)
        )
        vibrancy.setMaterial_(13)  # NSVisualEffectMaterialHUDWindow — true glass
        vibrancy.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        vibrancy.setState_(1)  # Active
        vibrancy.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        vibrancy.setWantsLayer_(True)
        vibrancy.layer().setCornerRadius_(22)
        vibrancy.layer().setMasksToBounds_(True)
        vibrancy.setAlphaValue_(0.75)  # let more of the background bleed through
        content.addSubview_(vibrancy)

        # Specular highlight — bright arc across the top edge, like light catching glass
        from Quartz import CAGradientLayer, CAShapeLayer
        highlight_view = PassthroughView.alloc().initWithFrame_(NSMakeRect(0, 0, WIN_WIDTH, WIN_HEIGHT))
        highlight_view.setWantsLayer_(True)
        highlight_view.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)

        specular = CAGradientLayer.alloc().init()
        specular.setFrame_(Quartz.CGRectMake(0, 0, WIN_WIDTH, WIN_HEIGHT))
        # In layer coords: bottom = visual top of window
        white_bright = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.30).CGColor()
        white_mid = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.06).CGColor()
        clear = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.0).CGColor()
        specular.setColors_([clear, white_mid, white_bright])
        specular.setLocations_([0.0, 0.75, 1.0])
        specular.setStartPoint_(Quartz.CGPointMake(0.5, 0.0))
        specular.setEndPoint_(Quartz.CGPointMake(0.5, 1.0))

        # Mask so it only shows as a thin edge, not a full fill
        edge_mask = CAShapeLayer.alloc().init()
        edge_path = Quartz.CGPathCreateWithRoundedRect(
            Quartz.CGRectMake(0.5, 0.5, WIN_WIDTH - 1, WIN_HEIGHT - 1), 22, 22, None
        )
        edge_mask.setPath_(edge_path)
        edge_mask.setFillColor_(None)
        edge_mask.setStrokeColor_(NSColor.whiteColor().CGColor())
        edge_mask.setLineWidth_(1.5)
        specular.setMask_(edge_mask)

        highlight_view.layer().addSublayer_(specular)
        content.addSubview_(highlight_view)

        # ── Waveform (always visible — white when idle, red when recording) ──
        waveform_w = WaveformView.NUM_BARS * WaveformView.BAR_WIDTH + (WaveformView.NUM_BARS - 1) * WaveformView.BAR_GAP + 20
        self.waveform = WaveformView.alloc().initWithFrame_(
            NSMakeRect((WIN_WIDTH - waveform_w) / 2, 186, waveform_w, 40)
        )
        self.waveform._color = NSColor.whiteColor()
        self.waveform.setLevels_([0.0] * WaveformView.NUM_BARS)
        content.addSubview_(self.waveform)

        # ── Status label (New York — Apple's modern serif) ──
        self.status_label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(20, 142, WIN_WIDTH - 40, 30)
        )
        self.status_label.setStringValue_("Hold fn to record")
        self.status_label.setBezeled_(False)
        self.status_label.setDrawsBackground_(False)
        self.status_label.setEditable_(False)
        self.status_label.setSelectable_(False)
        self.status_label.setFont_(NSFont.fontWithName_size_("Charter-Roman", 19)
                                   or NSFont.fontWithName_size_("Georgia", 19))
        self.status_label.setAlignment_(NSCenterTextAlignment)
        content.addSubview_(self.status_label)

        # ── Language toggle ──
        tog_w = 112
        tog_h = 26
        self.lang_toggle = LangToggleView.alloc().initWithFrame_(
            NSMakeRect((WIN_WIDTH - tog_w) / 2, 106, tog_w, tog_h)
        )
        self.lang_toggle.setCallback_(self._on_lang_toggle)
        content.addSubview_(self.lang_toggle)

        # ── Hands-Free toggle ──
        hf_w = 112
        hf_h = 24
        self.hf_toggle = HandsFreeToggleView.alloc().initWithFrame_(
            NSMakeRect((WIN_WIDTH - hf_w) / 2, 74, hf_w, hf_h)
        )
        self.hf_toggle.setCallback_(self._on_hands_free_toggle)
        content.addSubview_(self.hf_toggle)

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
            label.setTextColor_(NSColor.secondaryLabelColor())
            label.setLineBreakMode_(NSLineBreakByTruncatingTail)
            label.setClickCallback_(self.copy_to_clipboard)
            content.addSubview_(label)
            self.history_labels.append(label)
        self.window.makeKeyAndOrderFront_(None)

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

    def _on_lang_toggle(self, index):
        label = LANG_LABELS[index]
        self.language = WHISPER_CODES.get(label, label.lower())
        logging.info(f"Language switched to: {self.language}")

    def _on_hands_free_toggle(self, active):
        self.hands_free_mode = active
        logging.info(f"Hands-free mode: {active}")
        # Sync Claude Code voice mode
        try:
            voice_mode_path = os.path.expanduser("~/.config/claude-voice/voice_mode")
            with open(voice_mode_path, "w") as f:
                f.write("on" if active else "off")
        except Exception:
            logging.exception("Could not update voice_mode file")
        if active:
            def _():
                self.status_label.setStringValue_("Clap twice to start")
            self.on_main(_)
            self._stop_clap_monitor.clear()
            threading.Thread(target=self._clap_monitor, daemon=True).start()
        else:
            self._stop_clap_monitor.set()
            def _():
                self.status_label.setStringValue_("Hold fn to record")
            self.on_main(_)

    def _clap_monitor(self):
        """Single persistent stream for clap detection — handles both start and stop."""
        import time
        CHUNK = 512
        THRESHOLD = 0.15        # float32 peak (0.0–1.0)
        MIN_CLAP_GAP = 0.2      # min seconds between detected claps
        DOUBLE_CLAP_WINDOW = 0.9  # max seconds between two claps
        MIN_RECORD_BEFORE_STOP = 1.0  # ignore claps in first second of recording

        clap_times = []
        last_clap_time = 0.0
        record_start_time = [0.0]  # mutable for closure

        def audio_cb(indata, frames, time_info, status):
            nonlocal last_clap_time, clap_times
            if self._stop_clap_monitor.is_set():
                return

            peak = float(np.max(np.abs(indata)))
            now = time.time()

            # Ignore claps immediately after recording starts (mic pop noise)
            if self.recording and (now - record_start_time[0]) < MIN_RECORD_BEFORE_STOP:
                return

            if peak < THRESHOLD or (now - last_clap_time) < MIN_CLAP_GAP:
                return

            last_clap_time = now
            clap_times.append(now)
            clap_times[:] = [t for t in clap_times if now - t <= DOUBLE_CLAP_WINDOW]
            logging.info(f"Clap detected — peak={peak:.3f}, count={len(clap_times)}, recording={self.recording}")

            if len(clap_times) == 1:
                if not self.recording:
                    # Single clap while idle — interrupt Matilda if she's speaking
                    if subprocess.run(["pgrep", "-x", "afplay"], capture_output=True).returncode == 0:
                        subprocess.run(["pkill", "-x", "afplay"], capture_output=True)
                        clap_times.clear()
                        logging.info("Single clap — interrupted Matilda")
                        return
                    def _(): self.status_label.setStringValue_("Clap again…")
                    self.on_main(_)
                    def _reset(w=DOUBLE_CLAP_WINDOW):
                        time.sleep(w + 0.1)
                        if self.hands_free_mode and not self.recording:
                            def __(): self.status_label.setStringValue_("Clap twice to start")
                            self.on_main(__)
                    threading.Thread(target=_reset, daemon=True).start()
                else:
                    def _(): self.status_label.setStringValue_("Clap again to stop…")
                    self.on_main(_)
                    def _reset(w=DOUBLE_CLAP_WINDOW):
                        time.sleep(w + 0.1)
                        if self.recording and self.hands_free_mode:
                            def __(): self.status_label.setStringValue_("Clap twice to stop")
                            self.on_main(__)
                    threading.Thread(target=_reset, daemon=True).start()

            elif len(clap_times) >= 2:
                clap_times.clear()
                if not self.recording:
                    record_start_time[0] = time.time()
                    def _(): self.status_label.setStringValue_("Clap twice to stop")
                    self.on_main(_)
                    threading.Thread(target=self.start_recording, daemon=True).start()
                else:
                    threading.Thread(target=self.stop_and_transcribe, daemon=True).start()

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
            # Reset after 1.5s
            self.perform_selector_delayed("_reset_status")
        self.on_main(_)

    def _reset_status(self):
        def _():
            if not self.recording:
                msg = "Clap twice to start" if self.hands_free_mode else "Hold fn to record"
                self.status_label.setStringValue_(msg)
        self.on_main(_)

    def perform_selector_delayed(self, sel_name):
        import time
        def _delayed():
            time.sleep(1.5)
            self._reset_status()
        threading.Thread(target=_delayed, daemon=True).start()

    def set_state_idle(self):
        self._animating_waveform = False
        def _():
            self.waveform._color = NSColor.whiteColor()
            self.waveform.setLevels_([0.0] * WaveformView.NUM_BARS)
            msg = "Clap twice to start" if self.hands_free_mode else "Hold fn to record"
            self.status_label.setStringValue_(msg)
        self.on_main(_)

    def set_state_recording(self):
        def _():
            self.waveform._color = NSColor.systemRedColor()
            self.status_label.setStringValue_("Recording…")
        self.on_main(_)
        self._animating_waveform = True
        threading.Thread(target=self._animate_waveform, daemon=True).start()

    def set_state_transcribing(self):
        self._animating_waveform = False
        def _():
            self.waveform._color = NSColor.systemOrangeColor()
            self.waveform.setLevels_([0.3] * WaveformView.NUM_BARS)
            self.status_label.setStringValue_("Transcribing…")
        self.on_main(_)

    def _animate_waveform(self):
        import time
        while self._animating_waveform:
            if self.audio_frames:
                # Get the latest audio chunk and compute RMS levels
                latest = self.audio_frames[-1].flatten().astype(np.float32)
                chunk_size = max(1, len(latest) // WaveformView.NUM_BARS)
                levels = []
                for i in range(WaveformView.NUM_BARS):
                    start = i * chunk_size
                    end = min(start + chunk_size, len(latest))
                    if start < len(latest):
                        rms = np.sqrt(np.mean(latest[start:end] ** 2)) / 32768.0
                        level = min(1.0, rms * 25)  # High sensitivity for visible movement
                    else:
                        level = 0.0
                    levels.append(level)
            else:
                levels = [0.05] * WaveformView.NUM_BARS

            def _update(lvls=levels):
                self.waveform.setLevels_(lvls)
            self.on_main(_update)
            time.sleep(0.05)  # ~20fps

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
                                set wasPlaying to execute t javascript "
                                    (function() {{
                                        var v = document.querySelectorAll('video, audio');
                                        var playing = false;
                                        v.forEach(function(el) {{
                                            if (!el.paused) {{ playing = true; el.pause(); }}
                                        }});
                                        return playing;
                                    }})()"
                                if wasPlaying is "true" then return "paused"
                            end repeat
                        end repeat
                    end tell'''
                r = subprocess.run(["osascript", "-e", js_check], capture_output=True, text=True, timeout=3)
                if r.stdout.strip() == "paused":
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
                                set wasPlaying to do JavaScript "
                                    (function() {
                                        var v = document.querySelectorAll('video, audio');
                                        var playing = false;
                                        v.forEach(function(el) {
                                            if (!el.paused) { playing = true; el.pause(); }
                                        });
                                        return playing;
                                    })()" in t
                                if wasPlaying is "true" then return "paused"
                            end repeat
                        end repeat
                    end tell'''
                r = subprocess.run(["osascript", "-e", js_check], capture_output=True, text=True, timeout=3)
                if r.stdout.strip() == "paused":
                    sources.append(("browser", "Safari"))
        except Exception:
            pass
        return sources

    def pause_all_audio(self):
        sources = self.get_playing_sources()
        for kind, name in sources:
            if kind == "app":
                subprocess.Popen(["osascript", "-e", f'tell application "{name}" to pause'],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
            self.audio_frames = []
            self.recording = True
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS,
                dtype="int16", callback=self.audio_callback,
            )
            self.stream.start()
            self.set_state_recording()
            threading.Thread(target=self._pause_bg, daemon=True).start()
        except Exception:
            logging.exception("Error starting recording")

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

            # Check if audio is just silence (RMS below threshold)
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            logging.info(f"Audio RMS: {rms:.1f}")
            if rms < 200:  # Very quiet — likely no speech
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

                # Filter out prompt hallucinations and blank audio
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

                # Save existing clipboard, paste text, then restore
                saved = self._clipboard_save()
                subprocess.run(["pbcopy"], input=text.encode(), check=True)
                self.simulate_paste()
                if self.hands_free_mode:
                    self.simulate_return()
                import time; time.sleep(0.15)  # let paste land before restoring
                self._clipboard_restore(saved)

            finally:
                os.unlink(tmp.name)
                self.resume_all_audio()
                self.set_state_idle()

        except Exception:
            logging.exception("Error in transcription")
            self.set_state_idle()

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
        import time
        time.sleep(0.12)  # let paste land first
        source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
        key_down = CGEventCreateKeyboardEvent(source, 36, True)   # 36 = Return
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
        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            event_mask,
            callback,
            None,
        )

        if tap is None:
            logging.error("Could not create event tap")
            return

        run_loop_source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(
            Quartz.CFRunLoopGetMain(), run_loop_source, Quartz.kCFRunLoopCommonModes
        )
        Quartz.CGEventTapEnable(tap, True)
        logging.info("Event tap active")

    def run(self):
        self.app.run()


if __name__ == "__main__":
    # Set process name so dock shows "Voice to Text" not "Python"
    AppKit.NSProcessInfo.processInfo().setProcessName_("Voice to Text")
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        sys.exit(1)

    # Kill any existing instances
    import signal
    my_pid = os.getpid()
    pid_file = os.path.join(SCRIPT_DIR, ".voice_app.pid")
    if os.path.exists(pid_file):
        try:
            old_pid = int(open(pid_file).read().strip())
            if old_pid != my_pid:
                os.kill(old_pid, signal.SIGTERM)
                import time; time.sleep(0.5)
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
