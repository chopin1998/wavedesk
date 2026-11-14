"""WaveDesk: Next-Gen GPU-Accelerated AD3 Oscilloscope, Spectrum Analyzer & AWG Workbench."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import wave
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets as W

from .driver import (
    ACQ_MODES,
    AWG_IDLE_MODES,
    AWG_TRIGGER_SOURCES,
    AwgChannelConfig,
    ChannelConfig,
    HW_FILTERS,
    SSB_MODES,
    ScopeConfig,
    TRIGGER_LENGTH_CONDS,
    TRIGGER_TYPES,
    TriggerConfig,
    WAVEFORM_FUNCTIONS,
    select_optimal_sample_rate,
)
from .gpu_fft import WINDOWS, GpuFftEngine
from .gpu_view import GpuWaveformCurve
from .measurements import format_si
from .worker import InstrumentWorker


DARK_THEME_QSS = """
QMainWindow, QWidget#root, QWidget#rack, QScrollArea, QScrollArea > QWidget > QWidget {
    background-color: #0e131a;
    color: #dbe4ee;
}
QWidget {
    color: #dbe4ee;
    font-size: 11pt;
}
QSplitter::handle {
    background-color: #1a2330;
}
QFrame#card {
    background-color: #151d28;
    border: 1px solid #243244;
    border-radius: 8px;
}
QFrame#subCard {
    background-color: #101722;
    border: 1px solid #223044;
    border-radius: 6px;
}
QPushButton#collapseHeader {
    background-color: transparent;
    border: none;
    text-align: left;
    padding: 2px 4px;
    font-size: 11.5pt;
    font-weight: 700;
    color: #8ab4f8;
}
QPushButton#collapseHeader:hover {
    color: #b3d1ff;
}
QPushButton#miniBtn {
    background-color: #1a2738;
    border: 1px solid #2e425c;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 9.8pt;
    color: #9ecbff;
}
QPushButton#miniBtn:hover {
    background-color: #24374f;
    border-color: #5294e2;
}
QPushButton#removeBtn {
    background-color: #2a181c;
    border: 1px solid #592b33;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 9.5pt;
    color: #ff99a8;
}
QPushButton#removeBtn:hover {
    background-color: #452229;
    border-color: #a84454;
}
QFrame#metricBar {
    background-color: #131a24;
    border: 1px solid #223042;
    border-radius: 7px;
}
QLabel#brandTitle {
    font-size: 16pt;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: 0.5px;
}
QLabel#secTitle {
    font-size: 11.5pt;
    font-weight: 700;
    color: #8ab4f8;
}
QLabel#ch1Badge {
    color: #00f2fe;
    font-weight: 700;
    font-size: 11.2pt;
}
QLabel#ch2Badge {
    color: #ffb800;
    font-weight: 700;
    font-size: 11.2pt;
}
QLabel#muted {
    color: #8b9bb0;
    font-size: 10pt;
}
QLabel#metricValue {
    font-family: "JetBrains Mono", "DejaVu Sans Mono", "Consolas", monospace;
    font-size: 11pt;
    font-weight: 600;
    color: #eef4fa;
}
QStatusBar QLabel {
    font-family: "JetBrains Mono", "DejaVu Sans Mono", "Consolas", monospace;
}
QLabel#gpuBadge {
    background-color: #132f2c;
    color: #38ef7d;
    border: 1px solid #235c53;
    border-radius: 5px;
    padding: 4px 8px;
    font-weight: 600;
    font-size: 10pt;
}
QLabel#devBadge {
    background-color: #182638;
    color: #79b8ff;
    border: 1px solid #294466;
    border-radius: 5px;
    padding: 4px 8px;
    font-weight: 600;
    font-size: 10pt;
}
QPushButton {
    background-color: #1d2837;
    border: 1px solid #30425a;
    border-radius: 5px;
    padding: 5px 10px;
    color: #dbe4ee;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #263549;
    border-color: #4b6a90;
}
QPushButton:checked {
    background-color: #1f4068;
    border-color: #5294e2;
    color: #ffffff;
    font-weight: 700;
}
QPushButton#runBtn {
    background-color: #194d38;
    border-color: #2b8a62;
    color: #ffffff;
    font-weight: 700;
}
QPushButton#runBtn:hover {
    background-color: #21664a;
}
QPushButton#stopBtn {
    background-color: #5c2424;
    border-color: #9e3e3e;
    color: #ffffff;
    font-weight: 700;
}
QPushButton#stopBtn:hover {
    background-color: #752e2e;
}
QComboBox, QDoubleSpinBox, QSpinBox {
    background-color: #0f1620;
    border: 1px solid #2b3b4e;
    border-radius: 4px;
    padding: 4px 6px;
    color: #eef4fa;
}
QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {
    border-color: #49688d;
}
QComboBox QAbstractItemView {
    background-color: #121a24;
    color: #eef4fa;
    border: 1px solid #3b526d;
    selection-background-color: #1f4c7a;
    selection-color: #ffffff;
    outline: 0;
}
QComboBox QAbstractItemView::item {
    background-color: #121a24;
    color: #eef4fa;
    padding: 5px 10px;
    min-height: 24px;
}
QComboBox QAbstractItemView::item:hover,
QComboBox QAbstractItemView::item:selected {
    background-color: #1f4c7a;
    color: #ffffff;
}
QMenu {
    background-color: #121a24;
    color: #eef4fa;
    border: 1px solid #3b526d;
    padding: 4px;
}
QMenu::item {
    padding: 5px 16px;
    background-color: transparent;
    color: #eef4fa;
}
QMenu::item:selected {
    background-color: #1f4c7a;
    color: #ffffff;
}
QToolTip {
    background-color: #151f2b;
    color: #eef4fa;
    border: 1px solid #3b526d;
    padding: 4px 8px;
}
QTabWidget::pane {
    border: 1px solid #243244;
    border-radius: 6px;
    background-color: #151d28;
}
QTabBar::tab {
    background-color: #101720;
    border: 1px solid #243244;
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    padding: 6px 14px;
    color: #8b9bb0;
}
QTabBar::tab:selected {
    background-color: #151d28;
    color: #00f2fe;
    font-weight: 700;
}
QScrollArea {
    border: none;
}
QScrollBar:vertical {
    background: #0e131a;
    width: 10px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #2b3b4e;
    border-radius: 5px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QStatusBar {
    background-color: #0b0f15;
    color: #8b9bb0;
    border-top: 1px solid #1e2938;
    font-size: 10.2pt;
}
"""


# 10 horizontal divisions across the oscilloscope view (Span = 10 * time_per_div)
TIMEBASES = [
    ("50 ns/div (500 ns)", 50e-9),
    ("100 ns/div (1 µs)", 100e-9),
    ("200 ns/div (2 µs)", 200e-9),
    ("500 ns/div (5 µs)", 500e-9),
    ("1 µs/div (10 µs)", 1e-6),
    ("2 µs/div (20 µs)", 2e-6),
    ("5 µs/div (50 µs)", 5e-6),
    ("10 µs/div (100 µs)", 10e-6),
    ("20 µs/div (200 µs)", 20e-6),
    ("50 µs/div (500 µs)", 50e-6),
    ("100 µs/div (1 ms)", 100e-6),
    ("200 µs/div (2 ms)", 200e-6),
    ("500 µs/div (5 ms)", 500e-6),
    ("1 ms/div (10 ms)", 1e-3),
    ("2 ms/div (20 ms)", 2e-3),
    ("5 ms/div (50 ms)", 5e-3),
    ("10 ms/div (100 ms)", 10e-3),
]

SAMPLE_RATES = [
    ("Auto (随动高采样防混叠)", 0.0),
    ("100 MS/s", 100_000_000.0),
    ("50 MS/s", 50_000_000.0),
    ("20 MS/s", 20_000_000.0),
    ("10 MS/s", 10_000_000.0),
    ("5 MS/s", 5_000_000.0),
    ("2 MS/s", 2_000_000.0),
    ("1 MS/s", 1_000_000.0),
    ("500 kS/s", 500_000.0),
    ("200 kS/s", 200_000.0),
    ("100 kS/s", 100_000.0),
]

BUFFER_SIZES = [
    ("32768 pts (AD3 满深度)", 32768),
    ("16384 pts", 16384),
    ("8192 pts", 8192),
    ("4096 pts", 4096),
]

FFT_SIZES = [
    ("跟随采样深度 (1:1)", 0),
    ("65,536 pts (64K 超清频谱)", 65536),
    ("131,072 pts (128K 超高分辨)", 131072),
    ("262,144 pts (256K 极限分辨)", 262144),
]

FFT_SPANS = [
    ("5 MHz (差频+本振+和频+互调)", 5_000_000.0),
    ("2.5 MHz (差频+本振+和频)", 2_500_000.0),
    ("500 kHz (混频差频 IF 特写)", 500_000.0),
    ("10 MHz", 10_000_000.0),
    ("25 MHz", 25_000_000.0),
    ("自动谐波视野 (Auto)", -1.0),
    ("全奈奎斯特带宽 (Full)", 0.0),
]

VOLTAGE_RANGES = [
    ("±5 mV (1 mV/div · 软件100x)", (1.0, 0.01)),
    ("±10 mV (2 mV/div · 软件50x)", (1.0, 0.02)),
    ("±25 mV (5 mV/div · 软件20x)", (1.0, 0.05)),
    ("±50 mV (10 mV/div · 软件10x)", (1.0, 0.1)),
    ("±0.1 V (20 mV/div · 软件5x)", (1.0, 0.2)),
    ("±0.25 V (50 mV/div · 软件2x)", (1.0, 0.5)),
    ("±0.5 V (100 mV/div)", (1.0, 1.0)),
    ("±1.0 V (200 mV/div)", (2.0, 2.0)),
    ("±2.5 V (500 mV/div)", (5.0, 5.0)),
    ("±5.0 V (1 V/div)", (10.0, 10.0)),
    ("±10 V (2 V/div)", (20.0, 20.0)),
    ("±25 V (5 V/div)", (50.0, 50.0)),
]

CHANNEL_FILTERS = [
    ("全带宽 + IF包络虚线", (0.0, True)),
    ("全带宽 (Wideband)", (0.0, False)),
    ("LPF 200 kHz (仅提取差频 IF)", (200_000.0, False)),
    ("LPF 500 kHz (仅提取差频 IF)", (500_000.0, False)),
    ("LPF 1.0 MHz", (1_000_000.0, False)),
    ("LPF 2.5 MHz", (2_500_000.0, False)),
    ("LPF 5.0 MHz", (5_000_000.0, False)),
]

TRIGGER_COUPLINGS = [
    ("智能稳相 (Auto 包络+载波锁相)", "Auto"),
    ("高频抑制 (HF Reject 包络触发)", "HF Reject"),
    ("原始边沿 (DC 宽带触发)", "DC"),
]


def _create_app_icon() -> QtGui.QIcon:
    """Render a crisp multi-resolution oscilloscope waveform icon."""
    icon = QtGui.QIcon()
    for sz in (32, 64, 128, 256):
        pm = QtGui.QPixmap(sz, sz)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        pad = sz * 0.06
        rect = QtCore.QRectF(pad, pad, sz - 2 * pad, sz - 2 * pad)
        grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0.0, QtGui.QColor("#111b29"))
        grad.setColorAt(1.0, QtGui.QColor("#090f17"))
        p.setBrush(grad)
        p.setPen(QtGui.QPen(QtGui.QColor("#234263"), max(1.5, sz * 0.03)))
        p.drawRoundedRect(rect, sz * 0.22, sz * 0.22)

        # Subtle scope graticule crosshairs
        p.setPen(QtGui.QPen(QtGui.QColor(45, 76, 110, 110), max(1.0, sz * 0.015)))
        cx, cy = sz * 0.5, sz * 0.5
        p.drawLine(QtCore.QPointF(rect.left() + sz * 0.1, cy), QtCore.QPointF(rect.right() - sz * 0.1, cy))
        p.drawLine(QtCore.QPointF(cx, rect.top() + sz * 0.12), QtCore.QPointF(cx, rect.bottom() - sz * 0.12))

        # Glowing cyan modulated waveform trace
        path = QtGui.QPainterPath()
        n_pts = 80
        x0 = rect.left() + sz * 0.12
        w_span = rect.width() - sz * 0.24
        for i in range(n_pts + 1):
            u = i / n_pts
            px = x0 + u * w_span
            env = np.sin(np.pi * u) ** 0.7
            py = cy - env * (sz * 0.28) * np.sin(2.0 * np.pi * (u * 3.0 - 0.25))
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)

        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(0, 242, 254, 70), max(3.0, sz * 0.09), QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
        p.drawPath(path)
        p.setPen(QtGui.QPen(QtGui.QColor("#00f2fe"), max(1.8, sz * 0.045), QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
        p.drawPath(path)
        p.end()
        icon.addPixmap(pm)
    return icon


def _compact_combo(cmb: W.QComboBox, min_chars: int = 16) -> W.QComboBox:
    cmb.setSizeAdjustPolicy(W.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    cmb.setMinimumContentsLength(min_chars)
    cmb.setSizePolicy(W.QSizePolicy.Policy.Expanding, W.QSizePolicy.Policy.Fixed)
    return cmb


def _format_si_fixed(value: float, unit: str, decimals: int = 2, width: int = 11) -> str:
    return f"{format_si(value, unit, decimals):>{width}}"


class FreqUnitInput(W.QWidget):
    """Editable numeric frequency input paired with a selectable unit dropdown (Hz / kHz / MHz)."""

    valueChanged = QtCore.Signal(float)

    _UNITS: list[tuple[str, float]] = [
        ("Hz", 1.0),
        ("kHz", 1e3),
        ("MHz", 1e6),
    ]

    def __init__(
        self,
        min_hz: float = 0.1,
        max_hz: float = 25_000_000.0,
        default_hz: float = 1_000_000.0,
        allow_full: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._min_hz = float(min_hz)
        self._max_hz = float(max_hz)
        self._allow_full = bool(allow_full)

        row = W.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self.spn = W.QDoubleSpinBox()
        self.spn.setKeyboardTracking(False)
        self.spn.setDecimals(3)
        self.spn.setRange(0.001, 100_000_000.0)
        self.spn.setSingleStep(1.0)
        self.spn.setMinimumWidth(80)
        self.spn.valueChanged.connect(self._on_changed)

        self.cmb_unit = W.QComboBox()
        for lbl, scale in self._UNITS:
            self.cmb_unit.addItem(lbl, scale)
        if self._allow_full:
            self.cmb_unit.addItem("全频 (fs/2)", 0.0)
        self.cmb_unit.setMinimumWidth(72)
        self.cmb_unit.currentIndexChanged.connect(self._on_unit_changed)

        row.addWidget(self.spn, 1)
        row.addWidget(self.cmb_unit)
        self.setValue(default_hz)

    def value(self) -> float:
        scale = float(self.cmb_unit.currentData() or 0.0)
        if self._allow_full and scale <= 0.0:
            return 0.0
        return float(np.clip(self.spn.value() * scale, self._min_hz, self._max_hz))

    def setValue(self, hz: float):
        if self._allow_full and hz <= 0.0:
            with QtCore.QSignalBlocker(self.spn), QtCore.QSignalBlocker(self.cmb_unit):
                self.cmb_unit.setCurrentIndex(self.cmb_unit.count() - 1)
                self.spn.setEnabled(False)
            return

        hz = float(np.clip(hz, self._min_hz, self._max_hz))
        if hz >= 1e6:
            unit_idx, scale = 2, 1e6
        elif hz >= 1e3:
            unit_idx, scale = 1, 1e3
        else:
            unit_idx, scale = 0, 1.0

        with QtCore.QSignalBlocker(self.spn), QtCore.QSignalBlocker(self.cmb_unit):
            self.spn.setEnabled(True)
            self.cmb_unit.setCurrentIndex(unit_idx)
            self.spn.setValue(hz / scale)

    def _on_unit_changed(self):
        scale = float(self.cmb_unit.currentData() or 0.0)
        is_full = self._allow_full and scale <= 0.0
        self.spn.setEnabled(not is_full)
        if not is_full:
            clamped = float(np.clip(self.spn.value() * scale, self._min_hz, self._max_hz))
            if abs(clamped - self.spn.value() * scale) > 1e-6:
                with QtCore.QSignalBlocker(self.spn):
                    self.spn.setValue(clamped / scale)
        self.valueChanged.emit(self.value())

    def _on_changed(self):
        self.valueChanged.emit(self.value())


class CollapsibleCard(W.QFrame):
    """Collapsible dark card container with persistent expand/collapse state."""

    toggled = QtCore.Signal(bool)

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._title = title
        self._expanded = expanded

        outer = W.QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)

        self.header_layout = W.QHBoxLayout()
        self.header_layout.setSpacing(6)
        self.btn_toggle = W.QPushButton()
        self.btn_toggle.setObjectName("collapseHeader")
        self.btn_toggle.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_toggle.clicked.connect(self.toggle)
        self.header_layout.addWidget(self.btn_toggle, 1)
        outer.addLayout(self.header_layout)

        self.setSizePolicy(W.QSizePolicy.Policy.Expanding, W.QSizePolicy.Policy.Maximum)
        self.content = W.QWidget()
        self.content.setSizePolicy(W.QSizePolicy.Policy.Expanding, W.QSizePolicy.Policy.Maximum)
        self.body_layout = W.QVBoxLayout(self.content)
        self.body_layout.setContentsMargins(0, 2, 0, 0)
        self.body_layout.setSpacing(6)
        outer.addWidget(self.content)

        self.set_expanded(expanded, emit=False)

    def add_header_widget(self, widget: W.QWidget):
        self.header_layout.addWidget(widget)

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool, emit: bool = True):
        self._expanded = bool(expanded)
        arrow = "▾" if self._expanded else "▸"
        self.btn_toggle.setText(f"{arrow}  {self._title}")
        self.content.setVisible(self._expanded)
        if emit:
            self.toggled.emit(self._expanded)

    def toggle(self):
        self.set_expanded(not self._expanded, emit=True)


class FftViewBox(pg.ViewBox):
    """Spectrum ViewBox matching ScopeViewBox mouse ergonomics:
    - Left-drag pans horizontally (Frequency) only; vertical dBV scale is locked.
    - Wheel on main plot zooms horizontally (Frequency).
    - Wheel on/near the left vertical axis bar (or Ctrl+Wheel) zooms vertically (dBV).
    - Right-drag zoom is disabled.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(x=True, y=False)
        self.setMenuEnabled(False)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            return
        super().mouseDragEvent(ev, axis=axis)

    def wheelEvent(self, ev, axis=None):
        delta = ev.delta() if hasattr(ev, "delta") else ev.angleDelta().y()
        if delta == 0:
            ev.accept()
            return
        scale = 0.85 if delta > 0 else (1.0 / 0.85)
        mods = ev.modifiers() if hasattr(ev, "modifiers") else QtCore.Qt.KeyboardModifier.NoModifier
        near_left_axis = False
        if axis is None and hasattr(ev, "pos"):
            try:
                near_left_axis = float(ev.pos().x()) <= 28.0
            except Exception:
                near_left_axis = False
        center = self.mapSceneToView(ev.scenePos()) if hasattr(ev, "scenePos") else self.viewRect().center()
        if (mods & QtCore.Qt.KeyboardModifier.ControlModifier) or axis == 1 or near_left_axis:
            self.scaleBy((1.0, scale), center=center)
        else:
            self.scaleBy((scale, 1.0), center=center)
        ev.accept()


class ScopeViewBox(pg.ViewBox):
    """Oscilloscope ViewBox:
    - Left-drag pans horizontally only (vertical scale is locked).
    - Wheel on main plot zooms horizontally via discrete Time/div steps.
    - Wheel on/near the left vertical axis bar steps Voltage/div.
    - Right-drag zoom is disabled so vertical/horizontal never jump unexpectedly.
    """

    wheel_timebase_step = QtCore.Signal(int)
    wheel_voltage_step = QtCore.Signal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(x=True, y=False)
        self.setMenuEnabled(False)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            return
        super().mouseDragEvent(ev, axis=axis)

    def wheelEvent(self, ev, axis=None):
        delta = ev.delta() if hasattr(ev, "delta") else ev.angleDelta().y()
        if delta == 0:
            ev.accept()
            return
        step = -1 if delta > 0 else 1  # Scroll up = zoom in (smaller Time/div or smaller V/div)
        mods = ev.modifiers() if hasattr(ev, "modifiers") else QtCore.Qt.KeyboardModifier.NoModifier
        near_left_axis = False
        if axis is None and hasattr(ev, "pos"):
            try:
                near_left_axis = float(ev.pos().x()) <= 28.0
            except Exception:
                near_left_axis = False
        if (mods & QtCore.Qt.KeyboardModifier.ControlModifier) or axis == 1 or near_left_axis:
            self.wheel_voltage_step.emit(step)
        else:
            self.wheel_timebase_step.emit(step)
        ev.accept()


class FastPlotWidget(pg.PlotWidget):
    """OpenGL-accelerated oscilloscope/spectrum plot with dark oscilloscope grid."""

    def __init__(
        self,
        title: str,
        x_label: str,
        x_units: str,
        y_label: str,
        y_units: str,
        use_opengl: bool = True,
        view_box: pg.ViewBox | None = None,
    ):
        super().__init__(viewBox=view_box)
        self.setBackground("#0b0f15")
        self.useOpenGL(use_opengl)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setTitle(title, color="#9fb3c8", size="11pt")
        self.setLabel("bottom", x_label, units=x_units, color="#8b9bb0")
        self.setLabel("left", y_label, units=y_units, color="#8b9bb0")
        for side in ("left", "bottom"):
            ax = self.getAxis(side)
            ax.setPen(pg.mkPen("#29384b", width=1))
            ax.setTextPen(pg.mkPen("#8b9bb0"))
        self.getViewBox().setBorder(pg.mkPen("#243244", width=1))
        self.setMenuEnabled(False)
        self.getPlotItem().setMenuEnabled(False)


class OverviewPositionBar(W.QWidget):
    """Lightweight horizontal position/zoom indicator bar (zero per-frame CPU overhead)."""

    region_changed = QtCore.Signal(float, float)
    wheel_timebase_step = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.setToolTip("当前窗口在全缓冲中的位置与大小（拖动滑块平移，点击轨道跳转，滚轮缩放时基）")
        self._min_x = -163.84e-6
        self._max_x = 163.84e-6
        self._win_x0 = -25e-6
        self._win_x1 = 25e-6
        self._dragging = False
        self._drag_offset_x = 0.0

    def set_bounds(self, min_x: float, max_x: float):
        if max_x <= min_x:
            return
        if abs(min_x - self._min_x) < 1e-12 and abs(max_x - self._max_x) < 1e-12:
            return
        self._min_x = float(min_x)
        self._max_x = float(max_x)
        self.update()

    def set_region(self, x0: float, x1: float):
        if abs(x0 - self._win_x0) < 1e-12 and abs(x1 - self._win_x1) < 1e-12:
            return
        self._win_x0 = float(x0)
        self._win_x1 = float(x1)
        self.update()

    def get_region(self) -> tuple[float, float]:
        return self._win_x0, self._win_x1

    def _x_to_px(self, x: float, w: float) -> float:
        span = max(self._max_x - self._min_x, 1e-12)
        return ((x - self._min_x) / span) * w

    def _px_to_x(self, px: float, w: float) -> float:
        span = max(self._max_x - self._min_x, 1e-12)
        return self._min_x + (np.clip(px, 0.0, w) / max(w, 1.0)) * span

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        r = self.rect().adjusted(1, 1, -1, -1)
        w = float(r.width())

        # Dark track background
        p.setPen(QtGui.QPen(QtGui.QColor("#243244"), 1.0))
        p.setBrush(QtGui.QColor("#0a0f16"))
        p.drawRoundedRect(r, 4.0, 4.0)

        # Subtle quarter ticks and center 0-mark
        for frac, col in ((0.25, "#1b2736"), (0.5, "#526c8a"), (0.75, "#1b2736")):
            tx = r.left() + frac * w
            p.setPen(QtGui.QPen(QtGui.QColor(col), 1.0))
            p.drawLine(QtCore.QPointF(tx, r.top() + 3), QtCore.QPointF(tx, r.bottom() - 3))

        # Highlighted active window thumb
        x0_px = r.left() + self._x_to_px(self._win_x0, w)
        x1_px = r.left() + self._x_to_px(self._win_x1, w)
        thumb_w = max(10.0, x1_px - x0_px)
        cx = 0.5 * (x0_px + x1_px)
        left_px = float(np.clip(cx - 0.5 * thumb_w, r.left() + 1.0, r.right() - thumb_w - 1.0))
        thumb = QtCore.QRectF(left_px, r.top() + 2.0, thumb_w, float(r.height()) - 4.0)

        p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.4))
        p.setBrush(QtGui.QColor(200, 236, 255, 220))
        p.drawRoundedRect(thumb, 3.0, 3.0)

        # Dark center grip mark inside thumb
        mid_x = thumb.center().x()
        p.setPen(QtGui.QPen(QtGui.QColor("#0b1a2a"), 1.6))
        p.drawLine(QtCore.QPointF(mid_x, thumb.top() + 4.0), QtCore.QPointF(mid_x, thumb.bottom() - 4.0))
        p.end()

    def _emit_centered_at(self, center_x: float):
        span = max(self._win_x1 - self._win_x0, 500e-9)
        buf_span = max(self._max_x - self._min_x, 500e-9)
        span = min(span, buf_span)
        x0 = float(np.clip(center_x - 0.5 * span, self._min_x, self._max_x - span))
        x1 = x0 + span
        self._win_x0, self._win_x1 = x0, x1
        self.update()
        self.region_changed.emit(x0, x1)

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        w = max(float(self.width() - 2), 1.0)
        click_x = self._px_to_x(float(ev.position().x()) - 1.0, w)
        cur_mid = 0.5 * (self._win_x0 + self._win_x1)
        self._dragging = True
        self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        if self._win_x0 <= click_x <= self._win_x1:
            self._drag_offset_x = click_x - cur_mid
        else:
            self._drag_offset_x = 0.0
            self._emit_centered_at(click_x)
        ev.accept()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if not self._dragging:
            return
        w = max(float(self.width() - 2), 1.0)
        cur_x = self._px_to_x(float(ev.position().x()) - 1.0, w)
        self._emit_centered_at(cur_x - self._drag_offset_x)
        ev.accept()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            ev.accept()

    def wheelEvent(self, ev: QtGui.QWheelEvent):
        delta = ev.angleDelta().y()
        if delta != 0:
            self.wheel_timebase_step.emit(-1 if delta > 0 else 1)
        ev.accept()


class WaterfallOverlay(W.QFrame):
    """In-app draggable & resizable floating spectrogram waterfall (no OS window)."""

    closed = QtCore.Signal()

    def __init__(self, parent: W.QWidget, use_opengl: bool = False, rows: int = 120, cols: int = 512):
        super().__init__(parent)
        self._rows = rows
        self._cols = cols
        self._buf = np.full((rows, cols), -95.0, dtype=np.float32)
        self._drag_origin: QtCore.QPoint | None = None
        self.setObjectName("card")
        self.setStyleSheet(
            "WaterfallOverlay { background-color: rgba(13, 19, 28, 240); border: 1px solid #355070; border-radius: 8px; }"
        )
        self.resize(620, 250)
        self.setMinimumSize(360, 170)

        vbox = W.QVBoxLayout(self)
        vbox.setContentsMargins(6, 4, 6, 4)
        vbox.setSpacing(4)

        bar = W.QHBoxLayout()
        bar.setSpacing(4)
        self._title = W.QLabel("🌊 瀑布图")
        self._title.setStyleSheet("color: #7cc4ff; font-weight: 700; font-size: 9pt;")
        self._title.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
        btn_clr = W.QPushButton("清除")
        btn_clr.setObjectName("miniBtn")
        btn_clr.setStyleSheet("font-size: 8.5pt; padding: 1px 6px;")
        btn_clr.clicked.connect(self.clear)
        btn_cls = W.QPushButton("✕")
        btn_cls.setObjectName("removeBtn")
        btn_cls.setStyleSheet("font-size: 8.5pt; padding: 1px 5px;")
        btn_cls.clicked.connect(self._on_close)
        bar.addWidget(self._title, 1)
        bar.addWidget(btn_clr)
        bar.addWidget(btn_cls)
        vbox.addLayout(bar)

        self.plot = FastPlotWidget("", "", "Hz", "", "", use_opengl=False)
        self.plot.showGrid(x=True, y=False, alpha=0.18)
        tick_font = QtGui.QFont()
        tick_font.setPointSize(8)
        for ax_name in ("bottom", "left"):
            ax = self.plot.getPlotItem().getAxis(ax_name)
            ax.setStyle(tickFont=tick_font)
        self.plot.getPlotItem().setLabel("left", None)
        self.plot.getPlotItem().setLabel("bottom", "", units="Hz", color="#8b9bb0")
        self.plot.getPlotItem().getAxis("bottom").enableAutoSIPrefix(True)
        self.plot.getViewBox().setMouseEnabled(x=False, y=False)
        self.img = pg.ImageItem(axisOrder="row-major")
        # Dark -> Blue -> Cyan -> Yellow -> Red colormap
        pos = np.array([0.0, 0.25, 0.55, 0.80, 1.0])
        cols_rgba = np.array(
            [
                [8, 12, 20, 255],
                [18, 65, 130, 255],
                [0, 215, 255, 255],
                [255, 225, 30, 255],
                [255, 55, 65, 255],
            ],
            dtype=np.ubyte,
        )
        cmap = pg.ColorMap(pos, cols_rgba)
        self.img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        self.plot.addItem(self.img)
        vbox.addWidget(self.plot, 1)

        foot = W.QHBoxLayout()
        foot.setContentsMargins(0, 0, 0, 0)
        grip = W.QSizeGrip(self)
        foot.addStretch(1)
        foot.addWidget(grip, 0, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignBottom)
        vbox.addLayout(foot)

    def _on_close(self):
        self.hide()
        self.closed.emit()

    def clear(self):
        self._buf.fill(-95.0)
        self.img.setImage(self._buf, autoLevels=False, levels=(-95.0, -5.0))

    def sync_x_range(self, fx0: float, fx1: float):
        if fx1 > fx0:
            self.plot.setXRange(fx0, fx1, padding=0.0)

    def push_spectrum(self, freqs: np.ndarray, dbv: np.ndarray, fx0: float, fx1: float):
        if not self.isVisible() or freqs.size < 2 or fx1 <= fx0:
            return
        grid_f = np.linspace(fx0, fx1, self._cols, dtype=np.float64)
        row = np.interp(grid_f, freqs, dbv).astype(np.float32)
        self._buf = np.roll(self._buf, -1, axis=0)
        self._buf[-1, :] = row
        lo = float(np.percentile(self._buf, 5))
        hi = max(lo + 20.0, float(np.max(self._buf)))
        self.img.setImage(self._buf, autoLevels=False, levels=(lo, hi))
        self.img.setRect(QtCore.QRectF(fx0, -self._rows, fx1 - fx0, self._rows))
        self.plot.setXRange(fx0, fx1, padding=0.0)
        self.plot.setYRange(-self._rows, 0.0, padding=0.0)

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and ev.position().y() <= 34:
            self._drag_origin = ev.globalPosition().toPoint() - self.pos()
            self.raise_()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if self._drag_origin is not None:
            new_pos = ev.globalPosition().toPoint() - self._drag_origin
            if self.parentWidget() is not None:
                pw, ph = self.parentWidget().width(), self.parentWidget().height()
                new_pos.setX(max(0, min(pw - self.width(), new_pos.x())))
                new_pos.setY(max(0, min(ph - self.height(), new_pos.y())))
            self.move(new_pos)
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        self._drag_origin = None
        super().mouseReleaseEvent(ev)


class InputInteractionFilter(QtCore.QObject):
    """Global UX filter:
    1) Disable mouse-wheel changes on all QComboBox widgets (forward scroll to parent QScrollArea).
    2) On QSpinBox / QDoubleSpinBox, ignore wheel unless focused; when focused, step only the digit
       immediately before the text cursor while preserving that digit's decimal-place cursor position.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._forwarding = False

    @staticmethod
    def _forward_to_scroll_area(obj: QtCore.QObject, ev: QtGui.QWheelEvent) -> bool:
        p = obj.parent() if isinstance(obj, QtCore.QObject) else None
        while isinstance(p, W.QWidget):
            if isinstance(p, W.QAbstractScrollArea):
                vp = p.viewport()
                if vp is not None:
                    QtCore.QCoreApplication.sendEvent(vp, ev)
                    return True
            p = p.parentWidget()
        return False

    @staticmethod
    def step_spinbox_digit_at_cursor(spin: W.QAbstractSpinBox, steps: int) -> bool:
        le = spin.lineEdit()
        if le is None:
            return False
        text = le.text()
        cur = le.cursorPosition()
        prefix = spin.prefix()
        suffix = spin.suffix()
        num_start = len(prefix) if prefix and text.startswith(prefix) else 0
        num_end = len(text) - len(suffix) if suffix and text.endswith(suffix) else len(text)
        idx = cur - 1
        if not (num_start <= idx < num_end) or not text[idx].isdigit():
            return False

        num_str = text[num_start:num_end]
        rel_idx = idx - num_start
        dot_pos = num_str.find(".")
        if dot_pos >= 0:
            power = (dot_pos - 1 - rel_idx) if rel_idx < dot_pos else (dot_pos - rel_idx)
        else:
            power = (len(num_str) - 1) - rel_idx

        dist_from_end = num_end - cur
        if isinstance(spin, W.QDoubleSpinBox):
            step_val = 10.0 ** power
            new_val = round(spin.value() + steps * step_val, spin.decimals())
            new_val = max(spin.minimum(), min(spin.maximum(), new_val))
            spin.setValue(new_val)
        elif isinstance(spin, W.QSpinBox):
            step_val = int(round(10 ** max(0, power)))
            new_val = int(spin.value() + steps * step_val)
            new_val = max(spin.minimum(), min(spin.maximum(), new_val))
            spin.setValue(new_val)
        else:
            return False

        new_text = le.text()
        new_num_start = len(prefix) if prefix and new_text.startswith(prefix) else 0
        new_num_end = len(new_text) - len(suffix) if suffix and new_text.endswith(suffix) else len(new_text)
        new_cur = max(new_num_start, min(new_num_end, new_num_end - dist_from_end))
        le.setCursorPosition(new_cur)
        return True

    def eventFilter(self, obj: QtCore.QObject, ev: QtCore.QEvent) -> bool:
        if ev.type() != QtCore.QEvent.Type.Wheel or self._forwarding:
            return False

        if isinstance(obj, W.QComboBox):
            self._forwarding = True
            try:
                self._forward_to_scroll_area(obj, ev)
            finally:
                self._forwarding = False
            return True

        spin: W.QAbstractSpinBox | None = None
        if isinstance(obj, (W.QDoubleSpinBox, W.QSpinBox)):
            spin = obj
        elif isinstance(obj, W.QLineEdit) and isinstance(obj.parentWidget(), (W.QDoubleSpinBox, W.QSpinBox)):
            spin = obj.parentWidget()

        if spin is not None:
            le = spin.lineEdit()
            focused = spin.hasFocus() or (le is not None and le.hasFocus())
            if not focused:
                self._forwarding = True
                try:
                    self._forward_to_scroll_area(spin, ev)
                finally:
                    self._forwarding = False
                return True

            dy = ev.angleDelta().y()
            if dy == 0:
                dy = ev.pixelDelta().y()
            if dy != 0:
                self.step_spinbox_digit_at_cursor(spin, 1 if dy > 0 else -1)
            return True

        return False


class PolynomialWaveformDialog(W.QDialog):
    """Editor for multi-sine polynomial waveform: y(t) = sum_i A_i * sin(2*pi*f_i*t + phi_i)."""

    def __init__(self, ch_idx: int, terms: list[tuple[float, float, float]], base_freq_hz: float, parent=None):
        super().__init__(parent)
        self._base_freq_hz = max(100.0, float(base_freq_hz))
        self.setWindowTitle(f"W{ch_idx + 1} 多项式波形 (多正弦叠加编辑器)")
        self.resize(620, 380)

        vbox = W.QVBoxLayout(self)
        vbox.setSpacing(8)

        lbl = W.QLabel("y(t) = Σ Aᵢ · sin(2π fᵢ t + φᵢ)   (输出自动归一化到设定 Vpk 幅值)")
        lbl.setObjectName("secTitle")
        vbox.addWidget(lbl)

        self.table = W.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["频率 (fᵢ)", "幅值权重 (Aᵢ)", "相位 (φᵢ °)", "删除"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, W.QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, W.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, W.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, W.QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(3, 52)
        self.table.verticalHeader().setDefaultSectionSize(32)
        vbox.addWidget(self.table, 1)

        btn_bar = W.QHBoxLayout()
        btn_add = W.QPushButton("+ 添加正弦项")
        btn_add.clicked.connect(lambda: self._add_row(self._next_default_freq(), 1.0, 0.0))
        btn_two = W.QPushButton("双音示例")
        btn_two.clicked.connect(self._preset_two_tone)
        btn_odd = W.QPushButton("奇次谐波 (方波逼近)")
        btn_odd.clicked.connect(self._preset_odd_harmonics)
        btn_bar.addWidget(btn_add)
        btn_bar.addWidget(btn_two)
        btn_bar.addWidget(btn_odd)
        btn_bar.addStretch(1)
        vbox.addLayout(btn_bar)

        dlg_btns = W.QDialogButtonBox(
            W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel
        )
        dlg_btns.button(W.QDialogButtonBox.StandardButton.Ok).setText("应用")
        dlg_btns.button(W.QDialogButtonBox.StandardButton.Cancel).setText("取消")
        dlg_btns.accepted.connect(self.accept)
        dlg_btns.rejected.connect(self.reject)
        vbox.addWidget(dlg_btns)

        init_terms = terms if terms else [(self._base_freq_hz, 1.0, 0.0), (self._base_freq_hz * 1.2, 0.5, 0.0)]
        for f, a, ph in init_terms:
            self._add_row(f, a, ph)

    def _next_default_freq(self) -> float:
        cur = self.get_terms()
        if not cur:
            return self._base_freq_hz
        return min(25_000_000.0, cur[-1][0] + self._base_freq_hz)

    def _add_row(self, freq_hz: float, amp_w: float, phase_deg: float):
        row = self.table.rowCount()
        self.table.insertRow(row)

        f_in = FreqUnitInput(min_hz=1.0, max_hz=25_000_000.0, default_hz=float(freq_hz), allow_full=False)
        spn_a = W.QDoubleSpinBox()
        spn_a.setKeyboardTracking(False)
        spn_a.setRange(-100.0, 100.0)
        spn_a.setDecimals(3)
        spn_a.setSingleStep(0.1)
        spn_a.setValue(float(amp_w))

        spn_ph = W.QDoubleSpinBox()
        spn_ph.setKeyboardTracking(False)
        spn_ph.setRange(-360.0, 360.0)
        spn_ph.setDecimals(1)
        spn_ph.setSingleStep(15.0)
        spn_ph.setSuffix(" °")
        spn_ph.setValue(float(phase_deg))

        btn_del = W.QPushButton("✕")
        btn_del.setObjectName("removeBtn")
        btn_del.clicked.connect(lambda _, b=btn_del: self._remove_row_for_button(b))

        self.table.setCellWidget(row, 0, f_in)
        self.table.setCellWidget(row, 1, spn_a)
        self.table.setCellWidget(row, 2, spn_ph)
        self.table.setCellWidget(row, 3, btn_del)

    def _remove_row_for_button(self, btn: W.QPushButton):
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 3) is btn:
                if self.table.rowCount() > 1:
                    self.table.removeRow(r)
                break

    def _preset_two_tone(self):
        self.table.setRowCount(0)
        self._add_row(self._base_freq_hz, 1.0, 0.0)
        self._add_row(min(25_000_000.0, self._base_freq_hz * 1.1), 1.0, 0.0)

    def _preset_odd_harmonics(self):
        self.table.setRowCount(0)
        for k in (1, 3, 5, 7):
            f_k = self._base_freq_hz * k
            if f_k <= 25_000_000.0:
                self._add_row(f_k, round(1.0 / k, 4), 0.0)

    def get_terms(self) -> list[tuple[float, float, float]]:
        terms: list[tuple[float, float, float]] = []
        for r in range(self.table.rowCount()):
            f_in = self.table.cellWidget(r, 0)
            spn_a = self.table.cellWidget(r, 1)
            spn_ph = self.table.cellWidget(r, 2)
            if isinstance(f_in, FreqUnitInput) and isinstance(spn_a, W.QDoubleSpinBox) and isinstance(spn_ph, W.QDoubleSpinBox):
                terms.append((float(f_in.value()), float(spn_a.value()), float(spn_ph.value())))
        return terms


class MainWindow(W.QMainWindow):
    req_connect = QtCore.Signal()
    req_disconnect = QtCore.Signal()
    req_demo = QtCore.Signal(bool)
    req_start = QtCore.Signal()
    req_stop = QtCore.Signal()
    req_single = QtCore.Signal()
    req_scope_cfg = QtCore.Signal(object)
    req_awg_cfg = QtCore.Signal(int, object)
    req_pc_trig = QtCore.Signal()
    req_force_scope_trig = QtCore.Signal()
    req_fft_cfg = QtCore.Signal(dict)
    req_power_cfg = QtCore.Signal(bool, float, float)

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        plat = QtGui.QGuiApplication.platformName().lower()
        want_gl = getattr(args, "renderer", "opengl") == "opengl"
        self.use_opengl = want_gl and plat not in ("offscreen", "minimal")

        self.setWindowTitle("WaveDesk · AD3 GPU 超算示波器与频谱分析工作台")
        self.setWindowIcon(_create_app_icon())
        self.resize(1660, 980)
        self.setMinimumSize(1240, 760)

        self.scope_cfg = ScopeConfig(
            sample_rate_hz=50_000_000.0,
            buffer_size=32768,
            timebase_s_per_div=5e-6,
            auto_sample_rate=True,
            channels=[
                ChannelConfig(True, 2.0, lpf_cutoff_hz=0.0, show_envelope=True),
                ChannelConfig(False, 5.0, lpf_cutoff_hz=0.0, show_envelope=False),
            ],
            trigger=TriggerConfig(mode="Auto", source="CH1", slope="Rising", level_v=0.0, coupling="Auto"),
        )
        self.awg_cfgs = [
            AwgChannelConfig(enabled=False, function="Sine", frequency_hz=1_000_000.0, amplitude_v=1.0),
            AwgChannelConfig(enabled=False, function="Sine", frequency_hz=1_100_000.0, amplitude_v=1.0),
        ]
        self._active_scope_channels: set[int] = {0}
        self._active_awg_channels: set[int] = set()
        self._last_frame_data: dict | None = None
        self._peak_labels: list[pg.TextItem] = []

        self._input_filter = InputInteractionFilter(self)
        if (qapp := W.QApplication.instance()) is not None:
            qapp.installEventFilter(self._input_filter)

        self._build_ui()
        self._load_settings()
        self._apply_timebase_view()
        self._apply_vertical_view()
        self._init_worker()

        if getattr(args, "smoke", False):
            QtCore.QTimer.singleShot(2500, self.close)

    def _build_ui(self):
        root = W.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        main_layout = W.QVBoxLayout(root)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(8)

        # Top Toolbar
        top_bar = W.QFrame()
        top_bar.setObjectName("card")
        top_layout = W.QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 6, 12, 6)
        top_layout.setSpacing(8)

        brand = W.QLabel("WaveDesk")
        brand.setObjectName("brandTitle")
        top_layout.addWidget(brand)

        sub = W.QLabel("AD3 GPU Super-Workbench")
        sub.setObjectName("muted")
        top_layout.addWidget(sub)
        top_layout.addSpacing(8)

        self.btn_run = W.QPushButton("▶ RUN")
        self.btn_run.setObjectName("runBtn")
        self.btn_run.clicked.connect(self.req_start.emit)
        top_layout.addWidget(self.btn_run)

        self.btn_single = W.QPushButton("⏭ SINGLE")
        self.btn_single.clicked.connect(self.req_single.emit)
        top_layout.addWidget(self.btn_single)

        self.btn_stop = W.QPushButton("⏹ STOP")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.clicked.connect(self.req_stop.emit)
        top_layout.addWidget(self.btn_stop)

        self.btn_top_waterfall = W.QPushButton("🌊 瀑布图")
        self.btn_top_waterfall.setCheckable(True)
        self.btn_top_waterfall.setToolTip("打开/关闭应用内悬浮瀑布图 (非系统窗口)")
        self.btn_top_waterfall.toggled.connect(self._toggle_waterfall)
        top_layout.addWidget(self.btn_top_waterfall)

        top_layout.addStretch(1)

        self.lbl_status = W.QLabel("就绪")
        self.lbl_status.setObjectName("muted")
        top_layout.addWidget(self.lbl_status)

        self.lbl_fps = W.QLabel("FPS: -- | FFT: -- ms")
        self.lbl_fps.setObjectName("muted")
        top_layout.addWidget(self.lbl_fps)

        self.lbl_gpu_badge = W.QLabel("GPU FFT: 检测中…")
        self.lbl_gpu_badge.setObjectName("gpuBadge")
        top_layout.addWidget(self.lbl_gpu_badge)

        self.lbl_dev_badge = W.QLabel("设备: 未连接")
        self.lbl_dev_badge.setObjectName("devBadge")
        top_layout.addWidget(self.lbl_dev_badge)

        main_layout.addWidget(top_bar)

        # Body Splitter: Left plots + Right control rack
        self.body_split = W.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(self.body_split, 1)

        # Left side: Dual Vertical Splitter (Scope on top, GPU FFT Spectrum on bottom) + Measurement Dock
        left_container = W.QWidget()
        left_layout = W.QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.plot_splitter = W.QSplitter(QtCore.Qt.Orientation.Vertical)

        # 1. Time-Domain Oscilloscope Container (Top: Full-Buffer Overview Minimap + Bottom: Main Scope Plot)
        scope_container = W.QWidget()
        scope_vbox = W.QVBoxLayout(scope_container)
        scope_vbox.setContentsMargins(0, 0, 0, 0)
        scope_vbox.setSpacing(4)

        # 1a. Lightweight Horizontal Window Position/Size Bar (zero per-frame CPU)
        self.overview_bar = OverviewPositionBar()
        self.overview_bar.region_changed.connect(self._on_overview_region_changed)
        self.overview_bar.wheel_timebase_step.connect(self._step_timebase)
        self._last_overview_bounds: tuple[float, float] | None = None
        scope_vbox.addWidget(self.overview_bar)

        # 1b. Main Time-Domain Oscilloscope Plot (Left-drag horizontal pan, wheel=Time/div, wheel near Y-axis=V/div)
        self.scope_vb = ScopeViewBox()
        self.scope_vb.wheel_timebase_step.connect(self._step_timebase)
        self.scope_vb.wheel_voltage_step.connect(self._step_voltage_range)
        self.scope_vb.sigXRangeChanged.connect(self._on_scope_xrange_changed)
        self.scope_vb.setLimits(
            xMin=-0.1,
            xMax=0.1,
            minXRange=500e-9,
            maxXRange=0.1,
            yMin=-25.0,
            yMax=25.0,
            minYRange=0.002,
            maxYRange=50.0,
        )
        self.scope_plot = FastPlotWidget(
            "Time Domain · 实时稳相示波器 (主图滚轮=水平缩放 · 左键=水平拖动 · 左侧纵轴滚轮=垂直量程)",
            "Time",
            "s",
            "Voltage",
            "V",
            use_opengl=self.use_opengl,
            view_box=self.scope_vb,
        )
        self.scope_plot.setYRange(-1.0, 1.0)
        self.scope_curve_ch1 = GpuWaveformCurve(color="#00f2fe", width=2.1, dpo_enabled=False)
        self.scope_curve_ch2 = GpuWaveformCurve(color="#ffb800", width=2.1, dpo_enabled=False)
        self.scope_curve_ch2.setVisible(False)
        self.scope_plot.addItem(self.scope_curve_ch1)
        self.scope_plot.addItem(self.scope_curve_ch2)

        # IF Low-Pass Envelope Overlay Curve (dashed bright green-yellow so user sees 100 kHz IF wave clearly over composite signal)
        self.scope_env_ch1 = pg.PlotCurveItem(
            pen=pg.mkPen("#38ef7d", width=2.0, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.scope_env_ch1.setZValue(25)
        self.scope_plot.addItem(self.scope_env_ch1)

        # Peak-Detect (MinMax) hardware noise envelope curves
        self.scope_noise_ch1_min = pg.PlotCurveItem(pen=pg.mkPen("#00f2fe55", width=1.0, style=QtCore.Qt.PenStyle.DotLine))
        self.scope_noise_ch1_max = pg.PlotCurveItem(pen=pg.mkPen("#00f2fe55", width=1.0, style=QtCore.Qt.PenStyle.DotLine))
        self.scope_noise_ch2_min = pg.PlotCurveItem(pen=pg.mkPen("#ffb80055", width=1.0, style=QtCore.Qt.PenStyle.DotLine))
        self.scope_noise_ch2_max = pg.PlotCurveItem(pen=pg.mkPen("#ffb80055", width=1.0, style=QtCore.Qt.PenStyle.DotLine))
        for nc in (self.scope_noise_ch1_min, self.scope_noise_ch1_max, self.scope_noise_ch2_min, self.scope_noise_ch2_max):
            nc.setVisible(False)
            nc.setZValue(15)
            self.scope_plot.addItem(nc)

        # Trigger Horizontal Position (T=0) vertical marker line
        self.trig_pos_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=pg.mkPen("#5294e2", width=1.2, style=QtCore.Qt.PenStyle.DotLine),
            label="T0={value:0.2g}s",
            labelOpts={"position": 0.92, "color": "#8ab4f8", "fill": (15, 22, 32, 190)},
        )
        self.trig_pos_line.sigPositionChangeFinished.connect(self._on_trig_pos_dragged)
        self.scope_plot.addItem(self.trig_pos_line)

        # Trigger Level horizontal line
        self.trig_line = pg.InfiniteLine(
            pos=0.0,
            angle=0,
            movable=True,
            pen=pg.mkPen("#ff5252", width=1.4, style=QtCore.Qt.PenStyle.DashLine),
            label="Trig={value:0.2f}V",
            labelOpts={"position": 0.94, "color": "#ff7b7b", "fill": (25, 15, 20, 190)},
        )
        self.trig_line.sigPositionChangeFinished.connect(self._on_trig_line_dragged)
        self.scope_plot.addItem(self.trig_line)
        scope_vbox.addWidget(self.scope_plot, 1)
        self.plot_splitter.addWidget(scope_container)

        # 2. Frequency-Domain GPU FFT Spectrum Plot
        self.fft_vb = FftViewBox()
        self.fft_vb.sigXRangeChanged.connect(self._on_fft_xrange_changed)
        self.fft_plot = FastPlotWidget(
            "Frequency Domain · GPU 实时频谱分析仪 (Cooley-Tukey Radix-2 OpenCL/CUDA)",
            "Frequency",
            "Hz",
            "Magnitude",
            "dBV",
            use_opengl=self.use_opengl,
            view_box=self.fft_vb,
        )
        self.fft_plot.getViewBox().setLimits(
            xMin=0.0,
            xMax=50_000_000.0,
            minXRange=5_000.0,
            maxXRange=50_000_000.0,
            yMin=-140.0,
            yMax=30.0,
            minYRange=10.0,
            maxYRange=170.0,
        )
        self.fft_plot.setYRange(-105.0, 10.0)
        self.fft_plot.setXRange(0.0, 5_000_000.0, padding=0.0)
        self.fft_curve_ch1 = GpuWaveformCurve(color="#00f2fe", width=1.9, dpo_enabled=False)
        self.fft_curve_ch2 = GpuWaveformCurve(color="#ffb800", width=1.9, dpo_enabled=False)
        self.fft_curve_ch2.setVisible(False)
        self.fft_plot.addItem(self.fft_curve_ch1)
        self.fft_plot.addItem(self.fft_curve_ch2)

        # Peak markers & text callouts on FFT
        self.fft_peaks_scatter = pg.ScatterPlotItem(
            size=8, brush=pg.mkBrush("#ff4e79"), pen=pg.mkPen("#ffffff", width=1.2)
        )
        self.fft_peaks_scatter.setZValue(50)
        self.fft_plot.addItem(self.fft_peaks_scatter)
        for _ in range(5):
            lbl = pg.TextItem("", color="#ffd1dc", anchor=(0.5, 1.15), fill=(18, 24, 34, 205))
            lbl.setZValue(60)
            lbl.setVisible(False)
            self.fft_plot.addItem(lbl)
            self._peak_labels.append(lbl)

        self.plot_splitter.addWidget(self.fft_plot)
        self.plot_splitter.setSizes([520, 360])
        left_layout.addWidget(self.plot_splitter, 1)

        # Bottom Live Telemetry Cards (Two clean rows so nothing is ever truncated)
        metric_bar = W.QFrame()
        metric_bar.setObjectName("metricBar")
        m_vbox = W.QVBoxLayout(metric_bar)
        m_vbox.setContentsMargins(12, 6, 12, 6)
        m_vbox.setSpacing(4)

        row1 = W.QHBoxLayout()
        row1.setSpacing(14)
        self.lbl_meas_ch1 = W.QLabel("CH1: 等待采集…")
        self.lbl_meas_ch1.setObjectName("metricValue")
        self.lbl_meas_ch2 = W.QLabel("CH2: 已禁用")
        self.lbl_meas_ch2.setObjectName("metricValue")

        c1_badge = W.QLabel("● CH1")
        c1_badge.setObjectName("ch1Badge")
        row1.addWidget(c1_badge)
        row1.addWidget(self.lbl_meas_ch1, 2)

        c2_badge = W.QLabel("● CH2")
        c2_badge.setObjectName("ch2Badge")
        row1.addWidget(c2_badge)
        row1.addWidget(self.lbl_meas_ch2, 1)
        m_vbox.addLayout(row1)

        row2 = W.QHBoxLayout()
        row2.setSpacing(14)
        fft_badge = W.QLabel("◆ 频谱峰值与指标")
        fft_badge.setObjectName("secTitle")
        self.lbl_meas_fft = W.QLabel("FFT: 等待数据…")
        self.lbl_meas_fft.setObjectName("metricValue")
        row2.addWidget(fft_badge)
        row2.addWidget(self.lbl_meas_fft, 1)
        m_vbox.addLayout(row2)

        left_layout.addWidget(metric_bar)
        self.body_split.addWidget(left_container)

        # Right side: Instrument Control Rack (auto-sized to fit all controls cleanly on dark background)
        right_scroll = W.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setSizePolicy(W.QSizePolicy.Policy.Preferred, W.QSizePolicy.Policy.Expanding)
        rack = W.QWidget()
        rack.setObjectName("rack")
        rack_layout = W.QVBoxLayout(rack)
        rack_layout.setContentsMargins(4, 0, 4, 0)
        rack_layout.setSpacing(8)
        right_scroll.setWidget(rack)

        self.card_tb = self._build_timebase_card()
        self.card_trig = self._build_trigger_card()
        self.card_ch = self._build_channels_card()
        self.card_fft = self._build_fft_card()
        self.card_awg = self._build_awg_card()
        self.card_misc = self._build_misc_card()

        self.waterfall_overlay = WaterfallOverlay(root, use_opengl=self.use_opengl)
        self.waterfall_overlay.move(28, 86)
        self.waterfall_overlay.hide()
        self.waterfall_overlay.closed.connect(self._on_waterfall_closed)

        rack_layout.addWidget(self.card_tb)
        rack_layout.addWidget(self.card_trig)
        rack_layout.addWidget(self.card_ch)
        rack_layout.addWidget(self.card_fft)
        rack_layout.addWidget(self.card_awg)
        rack_layout.addWidget(self.card_misc)
        rack_layout.addStretch(1)

        # Let the right panel automatically expand to fit all controls cleanly without truncation
        rack.adjustSize()
        rack_w = max(610, rack.sizeHint().width() + 32)
        right_scroll.setMinimumWidth(rack_w)

        self.body_split.addWidget(right_scroll)
        self.body_split.setStretchFactor(0, 1)
        self.body_split.setStretchFactor(1, 0)
        self.body_split.setSizes([max(760, self.width() - rack_w - 24), rack_w])

    def _build_timebase_card(self) -> CollapsibleCard:
        card = CollapsibleCard("⏱ 水平时基与采集 (Timebase)", expanded=True)

        grid = W.QGridLayout()
        grid.setSpacing(5)
        grid.setColumnStretch(1, 1)

        # Row 0: Timebase (Time/div) with +/- buttons
        grid.addWidget(W.QLabel("水平时基 (Time/div):"), 0, 0)
        tb_row = W.QHBoxLayout()
        tb_row.setSpacing(4)
        self.btn_tb_faster = W.QPushButton("◀ 展开")
        self.btn_tb_faster.setToolTip("缩短时基 (放大波形细节)")
        self.btn_tb_faster.clicked.connect(lambda: self._step_timebase(-1))
        self.cmb_timebase = _compact_combo(W.QComboBox(), 11)
        for label, val in TIMEBASES:
            self.cmb_timebase.addItem(label, val)
        self.cmb_timebase.setCurrentIndex(6)  # 5 us/div (50 us total span = 5 cycles of 100 kHz IF)
        self.cmb_timebase.currentIndexChanged.connect(self._on_timebase_changed)
        self.btn_tb_slower = W.QPushButton("收缩 ▶")
        self.btn_tb_slower.setToolTip("拉长时基 (查看更多周期)")
        self.btn_tb_slower.clicked.connect(lambda: self._step_timebase(1))
        tb_row.addWidget(self.btn_tb_faster)
        tb_row.addWidget(self.cmb_timebase, 1)
        tb_row.addWidget(self.btn_tb_slower)
        grid.addLayout(tb_row, 0, 1)

        # Row 1: Horizontal Position + Sample Rate
        grid.addWidget(W.QLabel("水平位移 / 采样率:"), 1, 0)
        pos_rate_row = W.QHBoxLayout()
        pos_rate_row.setSpacing(4)
        self.spn_trig_pos = W.QDoubleSpinBox()
        self.spn_trig_pos.setKeyboardTracking(False)
        self.spn_trig_pos.setRange(-50000.0, 50000.0)
        self.spn_trig_pos.setDecimals(1)
        self.spn_trig_pos.setSingleStep(5.0)
        self.spn_trig_pos.setValue(0.0)
        self.spn_trig_pos.setSuffix(" µs")
        self.spn_trig_pos.valueChanged.connect(self._on_trig_pos_spin_changed)
        self.cmb_rate = _compact_combo(W.QComboBox(), 10)
        for label, val in SAMPLE_RATES:
            self.cmb_rate.addItem(label, val)
        self.cmb_rate.setCurrentIndex(0)  # Auto high-sample-rate anti-aliasing
        self.cmb_rate.currentIndexChanged.connect(self._emit_scope_config)
        pos_rate_row.addWidget(self.spn_trig_pos)
        pos_rate_row.addWidget(self.cmb_rate, 1)
        grid.addLayout(pos_rate_row, 1, 1)

        # Row 2: Buffer Size + Quick Timebase Actions
        grid.addWidget(W.QLabel("存储深度 / 快捷操作:"), 2, 0)
        buf_act_row = W.QHBoxLayout()
        buf_act_row.setSpacing(4)
        self.cmb_buf = _compact_combo(W.QComboBox(), 8)
        for label, val in BUFFER_SIZES:
            self.cmb_buf.addItem(label, val)
        self.cmb_buf.setCurrentIndex(0)  # 32768 pts default
        self.cmb_buf.currentIndexChanged.connect(self._emit_scope_config)
        self.btn_autoset = W.QPushButton("⚡ 自动时基")
        self.btn_autoset.setToolTip("智能自动时基与量程 (Auto-Set)")
        self.btn_autoset.clicked.connect(self.auto_set_scope)
        self.btn_autoscale = W.QPushButton("⤢ 视图复位")
        self.btn_autoscale.clicked.connect(self.autoscale_views)
        buf_act_row.addWidget(self.cmb_buf, 1)
        buf_act_row.addWidget(self.btn_autoset)
        buf_act_row.addWidget(self.btn_autoscale)
        grid.addLayout(buf_act_row, 2, 1)

        # Row 3: Acquisition Mode / Record Length / Hardware Counter
        grid.addWidget(W.QLabel("采集模式 / 计频:"), 3, 0)
        acq_row = W.QHBoxLayout()
        acq_row.setSpacing(4)
        self.cmb_acq_mode = _compact_combo(W.QComboBox(), 10)
        self.cmb_acq_mode.addItems(list(ACQ_MODES.keys()))
        self.cmb_acq_mode.currentTextChanged.connect(self._on_acq_mode_changed)
        self.spn_rec_len = W.QDoubleSpinBox()
        self.spn_rec_len.setKeyboardTracking(False)
        self.spn_rec_len.setRange(0.005, 5.0)
        self.spn_rec_len.setSingleStep(0.05)
        self.spn_rec_len.setDecimals(3)
        self.spn_rec_len.setValue(0.05)
        self.spn_rec_len.setSuffix(" s")
        self.spn_rec_len.setToolTip("流盘采集时长 (仅 Record 模式)")
        self.spn_rec_len.setVisible(False)
        self.spn_rec_len.valueChanged.connect(self._emit_scope_config)
        self.chk_hw_counter = W.QCheckBox("硬件计频")
        self.chk_hw_counter.setToolTip("启用 AD3 示波器硬件频率计 (FDwfAnalogInCounterSet)")
        self.chk_hw_counter.toggled.connect(self._emit_scope_config)
        acq_row.addWidget(self.cmb_acq_mode, 1)
        acq_row.addWidget(self.spn_rec_len)
        acq_row.addWidget(self.chk_hw_counter)
        grid.addLayout(acq_row, 3, 1)

        card.body_layout.addLayout(grid)
        return card

    def _build_trigger_card(self) -> CollapsibleCard:
        card = CollapsibleCard("🎯 稳相触发 (Trigger)", expanded=True)

        grid = W.QGridLayout()
        grid.setSpacing(5)
        grid.setColumnStretch(1, 1)

        # Row 0: Trigger Mode / Source / Force Trigger
        grid.addWidget(W.QLabel("触发模式 / 触发源:"), 0, 0)
        trig_row = W.QHBoxLayout()
        trig_row.setSpacing(4)
        self.cmb_trig_mode = _compact_combo(W.QComboBox(), 6)
        self.cmb_trig_mode.addItems(["Auto", "Normal", "Single", "None"])
        self.cmb_trig_mode.currentTextChanged.connect(self._emit_scope_config)
        self.cmb_trig_src = _compact_combo(W.QComboBox(), 6)
        self.cmb_trig_src.addItems(["CH1", "CH2", "W1", "W2", "Ext T1", "Ext T2", "PC"])
        self.cmb_trig_src.currentTextChanged.connect(self._emit_scope_config)
        self.btn_force_trig = W.QPushButton("⚡强制")
        self.btn_force_trig.setToolTip("强制示波器立即触发一帧 (FDwfAnalogInTriggerForce)")
        self.btn_force_trig.clicked.connect(self.req_force_scope_trig.emit)
        trig_row.addWidget(self.cmb_trig_mode, 1)
        trig_row.addWidget(self.cmb_trig_src, 1)
        trig_row.addWidget(self.btn_force_trig)
        grid.addLayout(trig_row, 0, 1)

        # Row 1: Trigger Slope / Level
        grid.addWidget(W.QLabel("触发边沿 / 电平:"), 1, 0)
        edge_row = W.QHBoxLayout()
        edge_row.setSpacing(4)
        self.cmb_trig_slope = _compact_combo(W.QComboBox(), 6)
        self.cmb_trig_slope.addItems(["Rising", "Falling", "Either"])
        self.cmb_trig_slope.currentTextChanged.connect(self._emit_scope_config)
        self.spn_trig_lvl = W.QDoubleSpinBox()
        self.spn_trig_lvl.setKeyboardTracking(False)
        self.spn_trig_lvl.setRange(-250.0, 250.0)
        self.spn_trig_lvl.setSingleStep(0.05)
        self.spn_trig_lvl.setDecimals(2)
        self.spn_trig_lvl.setSuffix(" V")
        self.spn_trig_lvl.valueChanged.connect(self._on_trig_spin_changed)
        btn_zero_trig = W.QPushButton("0V")
        btn_zero_trig.setMinimumWidth(44)
        btn_zero_trig.setToolTip("触发电平归零")
        btn_zero_trig.clicked.connect(lambda: self.spn_trig_lvl.setValue(0.0))
        edge_row.addWidget(self.cmb_trig_slope, 1)
        edge_row.addWidget(self.spn_trig_lvl, 1)
        edge_row.addWidget(btn_zero_trig)
        grid.addLayout(edge_row, 1, 1)

        # Row 2: Trigger Type / Hysteresis
        grid.addWidget(W.QLabel("触发类型 / 迟滞:"), 2, 0)
        ttype_row = W.QHBoxLayout()
        ttype_row.setSpacing(4)
        self.cmb_trig_type = _compact_combo(W.QComboBox(), 8)
        self.cmb_trig_type.addItems(list(TRIGGER_TYPES.keys()))
        self.cmb_trig_type.currentTextChanged.connect(self._on_trig_type_changed)
        self.spn_trig_hyst = W.QDoubleSpinBox()
        self.spn_trig_hyst.setKeyboardTracking(False)
        self.spn_trig_hyst.setRange(0.0, 10.0)
        self.spn_trig_hyst.setSingleStep(0.01)
        self.spn_trig_hyst.setDecimals(3)
        self.spn_trig_hyst.setValue(0.025)
        self.spn_trig_hyst.setSuffix(" V迟滞")
        self.spn_trig_hyst.valueChanged.connect(self._emit_scope_config)
        ttype_row.addWidget(self.cmb_trig_type, 1)
        ttype_row.addWidget(self.spn_trig_hyst, 1)
        grid.addLayout(ttype_row, 2, 1)

        # Row 3: Pulse/Transition Length Condition & Width (hidden for Edge/Window)
        self.lbl_trig_len = W.QLabel("脉宽条件 / 时长:")
        grid.addWidget(self.lbl_trig_len, 3, 0)
        tlen_row = W.QHBoxLayout()
        tlen_row.setSpacing(4)
        self.cmb_trig_len_cond = _compact_combo(W.QComboBox(), 8)
        self.cmb_trig_len_cond.addItems(list(TRIGGER_LENGTH_CONDS.keys()))
        self.cmb_trig_len_cond.currentTextChanged.connect(self._emit_scope_config)
        self.spn_trig_len_us = W.QDoubleSpinBox()
        self.spn_trig_len_us.setKeyboardTracking(False)
        self.spn_trig_len_us.setRange(0.01, 1_000_000.0)
        self.spn_trig_len_us.setSingleStep(1.0)
        self.spn_trig_len_us.setDecimals(2)
        self.spn_trig_len_us.setValue(1.0)
        self.spn_trig_len_us.setSuffix(" µs")
        self.spn_trig_len_us.valueChanged.connect(self._emit_scope_config)
        tlen_row.addWidget(self.cmb_trig_len_cond, 1)
        tlen_row.addWidget(self.spn_trig_len_us, 1)
        grid.addLayout(tlen_row, 3, 1)
        self.lbl_trig_len.setVisible(False)
        self.cmb_trig_len_cond.setVisible(False)
        self.spn_trig_len_us.setVisible(False)

        # Row 4: Holdoff / Hardware Trigger Filter
        grid.addWidget(W.QLabel("释抑 / 触发滤波:"), 4, 0)
        hold_row = W.QHBoxLayout()
        hold_row.setSpacing(4)
        self.spn_trig_holdoff = W.QDoubleSpinBox()
        self.spn_trig_holdoff.setKeyboardTracking(False)
        self.spn_trig_holdoff.setRange(0.0, 100_000.0)
        self.spn_trig_holdoff.setSingleStep(10.0)
        self.spn_trig_holdoff.setDecimals(1)
        self.spn_trig_holdoff.setValue(0.0)
        self.spn_trig_holdoff.setSuffix(" µs释抑")
        self.spn_trig_holdoff.valueChanged.connect(self._emit_scope_config)
        self.cmb_trig_hw_flt = _compact_combo(W.QComboBox(), 8)
        self.cmb_trig_hw_flt.addItems(["Average", "Decimate"])
        self.cmb_trig_hw_flt.currentTextChanged.connect(self._emit_scope_config)
        hold_row.addWidget(self.spn_trig_holdoff, 1)
        hold_row.addWidget(self.cmb_trig_hw_flt, 1)
        grid.addLayout(hold_row, 4, 1)

        # Row 5: Trigger Coupling / Stabilization
        grid.addWidget(W.QLabel("稳相模式:"), 5, 0)
        self.cmb_trig_coupling = _compact_combo(W.QComboBox(), 10)
        for label, code in TRIGGER_COUPLINGS:
            self.cmb_trig_coupling.addItem(label, code)
        self.cmb_trig_coupling.setCurrentIndex(0)  # Auto Smart Lock
        self.cmb_trig_coupling.currentIndexChanged.connect(self._emit_scope_config)
        grid.addWidget(self.cmb_trig_coupling, 5, 1)

        card.body_layout.addLayout(grid)
        return card

    def _build_misc_card(self) -> CollapsibleCard:
        card = CollapsibleCard("🛠 显示与系统杂项 (System / Misc)", expanded=False)

        pwr_row = W.QHBoxLayout()
        pwr_row.setSpacing(6)
        self.chk_pwr = W.QCheckBox("⚡ V+/V- 供电")
        self.chk_pwr.setChecked(False)
        self.chk_pwr.setToolTip("开启/关闭 AD3 硬件 V+ / V- 直流电源引脚 (为外部混频器/有源模块供电)")
        self.chk_pwr.toggled.connect(self._emit_power_config)
        self.spn_vpos = W.QDoubleSpinBox()
        self.spn_vpos.setKeyboardTracking(False)
        self.spn_vpos.setRange(0.5, 5.0)
        self.spn_vpos.setDecimals(2)
        self.spn_vpos.setSingleStep(0.1)
        self.spn_vpos.setValue(5.0)
        self.spn_vpos.setPrefix("V+ ")
        self.spn_vpos.setSuffix(" V")
        self.spn_vpos.valueChanged.connect(self._emit_power_config)
        self.spn_vneg = W.QDoubleSpinBox()
        self.spn_vneg.setKeyboardTracking(False)
        self.spn_vneg.setRange(-5.0, -0.5)
        self.spn_vneg.setDecimals(2)
        self.spn_vneg.setSingleStep(0.1)
        self.spn_vneg.setValue(-5.0)
        self.spn_vneg.setPrefix("V- ")
        self.spn_vneg.setSuffix(" V")
        self.spn_vneg.valueChanged.connect(self._emit_power_config)
        pwr_row.addWidget(self.chk_pwr)
        pwr_row.addWidget(self.spn_vpos, 1)
        pwr_row.addWidget(self.spn_vneg, 1)
        card.body_layout.addLayout(pwr_row)

        row = W.QHBoxLayout()
        row.setSpacing(8)
        self.chk_dpo = W.QCheckBox("DPO 余晖")
        self.chk_dpo.setChecked(False)
        self.chk_dpo.toggled.connect(self._on_dpo_toggled)
        self.chk_demo = W.QCheckBox("Demo 模拟")
        self.chk_demo.setChecked(bool(self.args.demo))
        self.chk_demo.toggled.connect(self.req_demo.emit)
        self.btn_reconnect = W.QPushButton("重连硬件")
        self.btn_reconnect.clicked.connect(self.req_connect.emit)

        row.addWidget(self.chk_dpo)
        row.addWidget(self.chk_demo)
        row.addStretch(1)
        row.addWidget(self.btn_reconnect)
        card.body_layout.addLayout(row)
        return card

    def _emit_power_config(self, *_):
        self.req_power_cfg.emit(
            self.chk_pwr.isChecked(),
            float(self.spn_vpos.value()),
            float(self.spn_vneg.value()),
        )

    def _build_channels_card(self) -> CollapsibleCard:
        card = CollapsibleCard("📈 示波器通道 (CH1 / CH2)", expanded=True)

        self.btn_add_ch_btns: list[W.QPushButton] = []
        for idx in range(2):
            b = W.QPushButton(f"+ CH{idx + 1}")
            b.setObjectName("miniBtn")
            b.clicked.connect(lambda _, i=idx: self._set_scope_channel_open(i, True))
            card.add_header_widget(b)
            self.btn_add_ch_btns.append(b)

        self.lbl_scope_empty = W.QLabel("未开启示波器通道，点击右上角 + CH1 / + CH2 添加")
        self.lbl_scope_empty.setObjectName("muted")
        self.lbl_scope_empty.setVisible(False)
        card.body_layout.addWidget(self.lbl_scope_empty)

        self.ch_controls = []
        self.ch_subcards: list[W.QFrame] = []
        for idx, name in enumerate(["● CH1 通道 1", "● CH2 通道 2"]):
            sub = W.QFrame()
            sub.setObjectName("subCard")
            sub.setSizePolicy(W.QSizePolicy.Policy.Expanding, W.QSizePolicy.Policy.Maximum)
            row_box = W.QVBoxLayout(sub)
            row_box.setContentsMargins(8, 6, 8, 6)
            row_box.setSpacing(5)

            hdr = W.QHBoxLayout()
            hdr.setSpacing(6)
            lbl_ch = W.QLabel(name)
            lbl_ch.setStyleSheet("color: #00f2fe; font-weight: 700;" if idx == 0 else "color: #ffb800; font-weight: 700;")
            hdr.addWidget(lbl_ch)

            cmb_range = _compact_combo(W.QComboBox(), 10)
            for lbl, val in VOLTAGE_RANGES:
                cmb_range.addItem(lbl, val)
            cmb_range.setCurrentIndex(7 if idx == 0 else 8)
            cmb_range.currentIndexChanged.connect(self._on_channel_range_changed)
            hdr.addWidget(cmb_range, 1)

            chk_ac = W.QCheckBox("AC耦合")
            chk_ac.toggled.connect(self._emit_scope_config)
            hdr.addWidget(chk_ac)

            btn_rem = W.QPushButton("✕")
            btn_rem.setObjectName("removeBtn")
            btn_rem.setToolTip(f"移除 CH{idx + 1} 通道")
            btn_rem.clicked.connect(lambda _, i=idx: self._set_scope_channel_open(i, False))
            hdr.addWidget(btn_rem)
            row_box.addLayout(hdr)

            # Hardware Offset + Probe Attenuation + Hardware Filter row
            hw_row = W.QHBoxLayout()
            hw_row.setSpacing(4)
            spn_offset = W.QDoubleSpinBox()
            spn_offset.setKeyboardTracking(False)
            spn_offset.setRange(-25.0, 25.0)
            spn_offset.setSingleStep(0.1)
            spn_offset.setDecimals(2)
            spn_offset.setValue(0.0)
            spn_offset.setSuffix(" V偏置")
            spn_offset.setToolTip("硬件垂直偏置 (FDwfAnalogInChannelOffsetSet)")
            spn_offset.valueChanged.connect(self._on_channel_range_changed)
            cmb_atten = _compact_combo(W.QComboBox(), 4)
            for lbl, aval in (("1x", 1.0), ("10x", 10.0), ("100x", 100.0)):
                cmb_atten.addItem(lbl, aval)
            cmb_atten.setToolTip("探头衰减倍率 (FDwfAnalogInChannelAttenuationSet)")
            cmb_atten.currentIndexChanged.connect(self._on_channel_range_changed)
            cmb_hw_flt = _compact_combo(W.QComboBox(), 10)
            cmb_hw_flt.addItems(list(HW_FILTERS.keys()))
            cmb_hw_flt.setToolTip("硬件采集滤波模式 (FDwfAnalogInChannelFilterSet)")
            cmb_hw_flt.currentIndexChanged.connect(self._emit_scope_config)
            hw_row.addWidget(spn_offset, 1)
            hw_row.addWidget(cmb_atten)
            hw_row.addWidget(cmb_hw_flt, 1)
            row_box.addLayout(hw_row)

            flt_row = W.QHBoxLayout()
            flt_row.setSpacing(5)
            flt_lbl = W.QLabel("信号/IF滤波:")
            flt_lbl.setObjectName("muted")
            cmb_flt = _compact_combo(W.QComboBox(), 12)
            for lbl, val in CHANNEL_FILTERS:
                cmb_flt.addItem(lbl, val)
            cmb_flt.setCurrentIndex(0 if idx == 0 else 1)
            cmb_flt.currentIndexChanged.connect(self._emit_scope_config)
            flt_row.addWidget(flt_lbl)
            flt_row.addWidget(cmb_flt, 1)
            row_box.addLayout(flt_row)

            sub.setVisible(idx in self._active_scope_channels)
            card.body_layout.addWidget(sub)
            self.ch_subcards.append(sub)
            self.ch_controls.append((cmb_range, chk_ac, cmb_flt, spn_offset, cmb_atten, cmb_hw_flt))

        self._sync_scope_channel_buttons()
        return card

    def _build_fft_card(self) -> CollapsibleCard:
        card = CollapsibleCard("🚀 GPU 频谱加速引擎 (FFT)", expanded=True)
        self.chk_fft_en = W.QCheckBox("启用")
        self.chk_fft_en.setChecked(True)
        self.chk_fft_en.toggled.connect(self._emit_fft_config)
        card.add_header_widget(self.chk_fft_en)

        grid = W.QGridLayout()
        grid.setSpacing(5)
        grid.setColumnStretch(1, 1)

        grid.addWidget(W.QLabel("频谱视野 (Span):"), 0, 0)
        self.fft_span_input = FreqUnitInput(
            min_hz=5_000.0,
            max_hz=50_000_000.0,
            default_hz=5_000_000.0,
            allow_full=True,
        )
        self.fft_span_input.valueChanged.connect(self._on_fft_span_changed)
        grid.addWidget(self.fft_span_input, 0, 1)

        grid.addWidget(W.QLabel("计算后端 (Backend):"), 1, 0)
        self.cmb_fft_backend = _compact_combo(W.QComboBox(), 12)
        self.cmb_fft_backend.addItem("OpenCL GPU (自动检测)", "opencl")
        self.cmb_fft_backend.addItem("CPU (SciPy PocketFFT)", "cpu")
        self.cmb_fft_backend.currentIndexChanged.connect(self._emit_fft_config)
        grid.addWidget(self.cmb_fft_backend, 1, 1)

        grid.addWidget(W.QLabel("窗函数 (Window):"), 2, 0)
        self.cmb_fft_win = _compact_combo(W.QComboBox(), 12)
        for wname, (_, desc) in WINDOWS.items():
            self.cmb_fft_win.addItem(f"{wname} · {desc}", wname)
        self.cmb_fft_win.currentIndexChanged.connect(self._emit_fft_config)
        grid.addWidget(self.cmb_fft_win, 2, 1)

        grid.addWidget(W.QLabel("平滑平均 / 点数:"), 3, 0)
        avg_res_row = W.QHBoxLayout()
        avg_res_row.setSpacing(4)
        self.spn_fft_avg = W.QSpinBox()
        self.spn_fft_avg.setRange(1, 64)
        self.spn_fft_avg.setValue(4)
        self.spn_fft_avg.setSuffix(" 帧")
        self.spn_fft_avg.valueChanged.connect(self._emit_fft_config)
        self.cmb_fft_size = _compact_combo(W.QComboBox(), 10)
        for lbl, val in FFT_SIZES:
            self.cmb_fft_size.addItem(lbl, val)
        self.cmb_fft_size.setCurrentIndex(0)
        self.cmb_fft_size.currentIndexChanged.connect(self._emit_fft_config)
        avg_res_row.addWidget(self.spn_fft_avg)
        avg_res_row.addWidget(self.cmb_fft_size, 1)
        grid.addLayout(avg_res_row, 3, 1)

        grid.addWidget(W.QLabel("单位 / 轨迹模式:"), 4, 0)
        unit_tr_row = W.QHBoxLayout()
        unit_tr_row.setSpacing(4)
        self.cmb_fft_unit = _compact_combo(W.QComboBox(), 6)
        self.cmb_fft_unit.addItem("dBV", "dbv")
        self.cmb_fft_unit.addItem("dBm (50Ω)", "dbm")
        self.cmb_fft_unit.addItem("Vrms (线性)", "vrms")
        self.cmb_fft_unit.currentIndexChanged.connect(self._on_fft_unit_changed)

        self.cmb_fft_trace = _compact_combo(W.QComboBox(), 8)
        self.cmb_fft_trace.addItem("实时 (Live)", "live")
        self.cmb_fft_trace.addItem("峰值保持 (Max)", "max_hold")
        self.cmb_fft_trace.addItem("谷值保持 (Min)", "min_hold")
        self.cmb_fft_trace.currentIndexChanged.connect(self._emit_fft_config)

        self.btn_fft_clear_hold = W.QPushButton("清除")
        self.btn_fft_clear_hold.setToolTip("清除峰值/谷值保持曲线")
        self.btn_fft_clear_hold.clicked.connect(self._clear_fft_hold)
        self.btn_waterfall = W.QPushButton("🌊 瀑布图")
        self.btn_waterfall.setCheckable(True)
        self.btn_waterfall.setToolTip("打开/关闭应用内悬浮瀑布图 (非系统窗口)")
        self.btn_waterfall.toggled.connect(self._toggle_waterfall)
        unit_tr_row.addWidget(self.cmb_fft_unit)
        unit_tr_row.addWidget(self.cmb_fft_trace, 1)
        unit_tr_row.addWidget(self.btn_fft_clear_hold)
        unit_tr_row.addWidget(self.btn_waterfall)
        grid.addLayout(unit_tr_row, 4, 1)

        self.chk_fft_zoom = W.QCheckBox("🔍 Zoom-FFT 窄带细化 (CZT 2048pt @ 当前视野)")
        self.chk_fft_zoom.setToolTip("对当前视野 [0, Span] 使用 Chirp-Z 变换做窄带高分辨率频谱细化")
        self.chk_fft_zoom.toggled.connect(self._emit_fft_config)
        grid.addWidget(self.chk_fft_zoom, 5, 0, 1, 2)

        self.chk_goertzel = W.QCheckBox("🎯 单频测量")
        self.chk_goertzel.setToolTip("Goertzel 算法精确测量指定频率的幅度与相位")
        self.chk_goertzel.toggled.connect(self._emit_fft_config)
        grid.addWidget(self.chk_goertzel, 6, 0)

        gz_row = W.QHBoxLayout()
        gz_row.setSpacing(4)
        self.goertzel_freq = FreqUnitInput(
            min_hz=10.0,
            max_hz=50_000_000.0,
            default_hz=200_000.0,
            allow_full=False,
        )
        self.goertzel_freq.valueChanged.connect(self._emit_fft_config)
        self.lbl_goertzel = W.QLabel("-- dBV · -- °")
        self.lbl_goertzel.setObjectName("metricValue")
        gz_row.addWidget(self.goertzel_freq, 1)
        gz_row.addWidget(self.lbl_goertzel)
        grid.addLayout(gz_row, 6, 1)

        card.body_layout.addLayout(grid)
        return card

    def _build_awg_card(self) -> CollapsibleCard:
        card = CollapsibleCard("🌊 信号发生器 (Wavegen W1 / W2)", expanded=False)

        self.btn_add_awg_btns: list[W.QPushButton] = []
        for idx in range(2):
            b = W.QPushButton(f"+ W{idx + 1}")
            b.setObjectName("miniBtn")
            b.clicked.connect(lambda _, i=idx: self._set_awg_channel_open(i, True))
            card.add_header_widget(b)
            self.btn_add_awg_btns.append(b)

        self.lbl_awg_empty = W.QLabel("波形发生器默认关闭，点击右上角 + W1 / + W2 添加通道")
        self.lbl_awg_empty.setObjectName("muted")
        self.lbl_awg_empty.setWordWrap(True)
        card.body_layout.addWidget(self.lbl_awg_empty)

        self.awg_widgets = []
        self.awg_adv_widgets = []
        self.awg_custom_samples: list[list[float] | None] = [None, None]
        self.awg_poly_terms: list[list[tuple[float, float, float]] | None] = [None, None]
        self.awg_am_wav: list[tuple[list[float], float, str] | None] = [None, None]
        self.awg_fm_wav: list[tuple[list[float], float, str] | None] = [None, None]
        self.awg_subcards: list[W.QFrame] = []

        for idx in range(2):
            sub = W.QFrame()
            sub.setObjectName("subCard")
            sub.setSizePolicy(W.QSizePolicy.Policy.Expanding, W.QSizePolicy.Policy.Maximum)
            p_layout = W.QVBoxLayout(sub)
            p_layout.setContentsMargins(8, 6, 8, 6)
            p_layout.setSpacing(5)

            hdr = W.QHBoxLayout()
            hdr.setSpacing(6)
            chk_en = W.QPushButton(f"⚡ 启用 W{idx + 1} 输出")
            chk_en.setCheckable(True)
            chk_en.setChecked(False)  # Always default OFF on startup
            chk_en.toggled.connect(lambda _, i=idx: self._emit_awg_config(i))
            hdr.addWidget(chk_en, 1)

            btn_rem = W.QPushButton("✕")
            btn_rem.setObjectName("removeBtn")
            btn_rem.setToolTip(f"移除并关闭 W{idx + 1}")
            btn_rem.clicked.connect(lambda _, i=idx: self._set_awg_channel_open(i, False))
            hdr.addWidget(btn_rem)
            p_layout.addLayout(hdr)

            grid = W.QGridLayout()
            grid.setSpacing(5)
            grid.setColumnStretch(1, 1)

            grid.addWidget(W.QLabel("波形类型:"), 0, 0)
            fn_row = W.QHBoxLayout()
            fn_row.setSpacing(4)
            cmb_fn = _compact_combo(W.QComboBox(), 10)
            cmb_fn.addItems(list(WAVEFORM_FUNCTIONS.keys()))
            cmb_fn.setCurrentText("Sine")
            btn_load_custom = W.QPushButton("📂 导入")
            btn_load_custom.setToolTip("从 CSV/TXT/NPY 加载自定义任意波形 (归一化到 [-1, 1])")
            btn_load_custom.setVisible(False)
            btn_load_custom.clicked.connect(lambda _, i=idx: self._load_custom_awg_waveform(i))
            btn_edit_poly = W.QPushButton("∑ 编辑多项式")
            btn_edit_poly.setToolTip("输入多个正弦项的频率、幅值和相位进行叠加")
            btn_edit_poly.setVisible(False)
            btn_edit_poly.clicked.connect(lambda _, i=idx: self._edit_poly_awg_waveform(i))
            cmb_fn.currentTextChanged.connect(
                lambda txt, i=idx: self._on_awg_fn_changed(i, txt)
            )
            fn_row.addWidget(cmb_fn, 1)
            fn_row.addWidget(btn_load_custom)
            fn_row.addWidget(btn_edit_poly)
            grid.addLayout(fn_row, 0, 1)

            grid.addWidget(W.QLabel("频率:"), 1, 0)
            spn_freq = FreqUnitInput(
                min_hz=0.1,
                max_hz=25_000_000.0,
                default_hz=1_000_000.0 if idx == 0 else 1_100_000.0,
                allow_full=False,
            )
            spn_freq.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            grid.addWidget(spn_freq, 1, 1)

            grid.addWidget(W.QLabel("幅值 / 偏置:"), 2, 0)
            amp_off_row = W.QHBoxLayout()
            amp_off_row.setSpacing(4)
            spn_amp = W.QDoubleSpinBox()
            spn_amp.setKeyboardTracking(False)
            spn_amp.setRange(0.01, 5.0)
            spn_amp.setDecimals(2)
            spn_amp.setSingleStep(0.1)
            spn_amp.setValue(1.0)
            spn_amp.setSuffix(" Vpk")
            spn_amp.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))

            spn_off = W.QDoubleSpinBox()
            spn_off.setKeyboardTracking(False)
            spn_off.setRange(-5.0, 5.0)
            spn_off.setDecimals(2)
            spn_off.setSingleStep(0.1)
            spn_off.setValue(0.0)
            spn_off.setSuffix(" V偏置")
            spn_off.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            amp_off_row.addWidget(spn_amp, 1)
            amp_off_row.addWidget(spn_off, 1)
            grid.addLayout(amp_off_row, 2, 1)

            grid.addWidget(W.QLabel("占空比 / 相位:"), 3, 0)
            sym_ph_row = W.QHBoxLayout()
            sym_ph_row.setSpacing(4)
            spn_sym = W.QDoubleSpinBox()
            spn_sym.setKeyboardTracking(False)
            spn_sym.setRange(0.0, 100.0)
            spn_sym.setDecimals(1)
            spn_sym.setValue(50.0)
            spn_sym.setSuffix(" %")
            spn_sym.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))

            spn_phase = W.QDoubleSpinBox()
            spn_phase.setKeyboardTracking(False)
            spn_phase.setRange(0.0, 360.0)
            spn_phase.setDecimals(1)
            spn_phase.setSingleStep(15.0)
            spn_phase.setValue(0.0)
            spn_phase.setSuffix(" °")
            spn_phase.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            sym_ph_row.addWidget(spn_sym, 1)
            sym_ph_row.addWidget(spn_phase, 1)
            grid.addLayout(sym_ph_row, 3, 1)

            p_layout.addLayout(grid)

            # Collapsible SDK Advanced Drawer (AM/FM Modulation, Burst Run/Wait/Repeat, Trigger, Idle, Sync)
            btn_adv = W.QPushButton("▸ 调制 / 突发 / 触发 / 同步 (AM/FM/Burst)")
            btn_adv.setObjectName("miniBtn")
            btn_adv.setCheckable(True)
            btn_adv.setChecked(False)
            p_layout.addWidget(btn_adv)

            adv_box = W.QWidget()
            adv_box.setVisible(False)
            btn_adv.toggled.connect(
                lambda checked, b=btn_adv, w=adv_box: (
                    w.setVisible(checked),
                    b.setText(("▾ " if checked else "▸ ") + "调制 / 突发 / 触发 / 同步 (AM/FM/Burst)"),
                )
            )
            adv_grid = W.QGridLayout(adv_box)
            adv_grid.setContentsMargins(0, 4, 0, 0)
            adv_grid.setSpacing(4)
            adv_grid.setColumnStretch(1, 1)

            mod_fns = [k for k in WAVEFORM_FUNCTIONS if k != "Custom (自定义)"]

            # AM Modulation
            chk_am = W.QCheckBox("AM 调幅")
            chk_am.toggled.connect(lambda _, i=idx: self._emit_awg_config(i))
            adv_grid.addWidget(chk_am, 0, 0)
            am_row = W.QHBoxLayout()
            am_row.setSpacing(4)
            cmb_am_fn = _compact_combo(W.QComboBox(), 8)
            cmb_am_fn.addItems(mod_fns + ["WAV (音频文件)"])
            freq_am = FreqUnitInput(min_hz=0.1, max_hz=1_000_000.0, default_hz=1_000.0)
            freq_am.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            cmb_am_wav = self._populate_wav_combo(_compact_combo(W.QComboBox(), 16))
            cmb_am_wav.setVisible(False)
            cmb_am_wav.currentIndexChanged.connect(lambda _, i=idx: self._on_mod_wav_selected(i, "am"))
            cmb_am_fn.currentTextChanged.connect(lambda txt, i=idx: self._on_mod_fn_changed(i, "am", txt))
            spn_am_depth = W.QDoubleSpinBox()
            spn_am_depth.setKeyboardTracking(False)
            spn_am_depth.setRange(0.0, 100.0)
            spn_am_depth.setDecimals(1)
            spn_am_depth.setValue(50.0)
            spn_am_depth.setSuffix(" %")
            spn_am_depth.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            am_row.addWidget(cmb_am_fn)
            am_row.addWidget(freq_am, 1)
            am_row.addWidget(cmb_am_wav, 1)
            am_row.addWidget(spn_am_depth)
            adv_grid.addLayout(am_row, 0, 1)

            # FM Modulation
            chk_fm = W.QCheckBox("FM 调频")
            chk_fm.toggled.connect(lambda _, i=idx: self._emit_awg_config(i))
            adv_grid.addWidget(chk_fm, 1, 0)
            fm_row = W.QHBoxLayout()
            fm_row.setSpacing(4)
            cmb_fm_fn = _compact_combo(W.QComboBox(), 8)
            cmb_fm_fn.addItems(mod_fns + ["WAV (音频文件)"])
            freq_fm = FreqUnitInput(min_hz=0.1, max_hz=1_000_000.0, default_hz=1_000.0)
            freq_fm.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            cmb_fm_wav = self._populate_wav_combo(_compact_combo(W.QComboBox(), 16))
            cmb_fm_wav.setVisible(False)
            cmb_fm_wav.currentIndexChanged.connect(lambda _, i=idx: self._on_mod_wav_selected(i, "fm"))
            cmb_fm_fn.currentTextChanged.connect(lambda txt, i=idx: self._on_mod_fn_changed(i, "fm", txt))
            spn_fm_depth = W.QDoubleSpinBox()
            spn_fm_depth.setKeyboardTracking(False)
            spn_fm_depth.setRange(0.0, 100.0)
            spn_fm_depth.setDecimals(1)
            spn_fm_depth.setValue(10.0)
            spn_fm_depth.setSuffix(" %")
            spn_fm_depth.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            fm_row.addWidget(cmb_fm_fn)
            fm_row.addWidget(freq_fm, 1)
            fm_row.addWidget(cmb_fm_wav, 1)
            fm_row.addWidget(spn_fm_depth)
            adv_grid.addLayout(fm_row, 1, 1)

            # Run / Wait
            adv_grid.addWidget(W.QLabel("运行 / 等待:"), 2, 0)
            rw_row = W.QHBoxLayout()
            rw_row.setSpacing(4)
            spn_run_ms = W.QDoubleSpinBox()
            spn_run_ms.setKeyboardTracking(False)
            spn_run_ms.setRange(0.0, 100_000.0)
            spn_run_ms.setDecimals(3)
            spn_run_ms.setValue(0.0)
            spn_run_ms.setSuffix(" ms运行")
            spn_run_ms.setToolTip("0 = 连续运行; >0 = 单次突发持续时间")
            spn_run_ms.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))

            spn_wait_ms = W.QDoubleSpinBox()
            spn_wait_ms.setKeyboardTracking(False)
            spn_wait_ms.setRange(0.0, 100_000.0)
            spn_wait_ms.setDecimals(3)
            spn_wait_ms.setValue(0.0)
            spn_wait_ms.setSuffix(" ms等待")
            spn_wait_ms.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            rw_row.addWidget(spn_run_ms, 1)
            rw_row.addWidget(spn_wait_ms, 1)
            adv_grid.addLayout(rw_row, 2, 1)

            # Repeat / RepeatTrigger
            adv_grid.addWidget(W.QLabel("重复次数:"), 3, 0)
            rep_row = W.QHBoxLayout()
            rep_row.setSpacing(6)
            spn_repeat = W.QSpinBox()
            spn_repeat.setKeyboardTracking(False)
            spn_repeat.setRange(0, 1_000_000)
            spn_repeat.setValue(0)
            spn_repeat.setSpecialValueText("无限 (0)")
            spn_repeat.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            chk_rep_trig = W.QCheckBox("重触发")
            chk_rep_trig.setToolTip("每次循环等待新触发")
            chk_rep_trig.toggled.connect(lambda _, i=idx: self._emit_awg_config(i))
            rep_row.addWidget(spn_repeat, 1)
            rep_row.addWidget(chk_rep_trig)
            adv_grid.addLayout(rep_row, 3, 1)

            # Trigger Source / Slope / PC Trigger
            adv_grid.addWidget(W.QLabel("触发源 / 边沿:"), 4, 0)
            atrig_row = W.QHBoxLayout()
            atrig_row.setSpacing(4)
            cmb_awg_trig = _compact_combo(W.QComboBox(), 7)
            cmb_awg_trig.addItems(list(AWG_TRIGGER_SOURCES.keys()))
            cmb_awg_trig.currentTextChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            cmb_awg_slope = _compact_combo(W.QComboBox(), 5)
            cmb_awg_slope.addItems(["Rising", "Falling", "Either"])
            cmb_awg_slope.currentTextChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            btn_pc_trig = W.QPushButton("⚡PC")
            btn_pc_trig.setToolTip("发送 PC 手动触发脉冲")
            btn_pc_trig.clicked.connect(self.req_pc_trig.emit)
            atrig_row.addWidget(cmb_awg_trig, 1)
            atrig_row.addWidget(cmb_awg_slope)
            atrig_row.addWidget(btn_pc_trig)
            adv_grid.addLayout(atrig_row, 4, 1)

            # Idle Mode / Limitation / Master Sync
            adv_grid.addWidget(W.QLabel("空闲 / 限幅:"), 5, 0)
            idle_row = W.QHBoxLayout()
            idle_row.setSpacing(4)
            cmb_idle = _compact_combo(W.QComboBox(), 8)
            cmb_idle.addItems(list(AWG_IDLE_MODES.keys()))
            cmb_idle.currentTextChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            spn_limit_v = W.QDoubleSpinBox()
            spn_limit_v.setKeyboardTracking(False)
            spn_limit_v.setRange(0.0, 5.0)
            spn_limit_v.setDecimals(2)
            spn_limit_v.setValue(0.0)
            spn_limit_v.setSpecialValueText("默认限幅")
            spn_limit_v.setSuffix(" V限幅")
            spn_limit_v.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            idle_row.addWidget(cmb_idle, 1)
            idle_row.addWidget(spn_limit_v)
            adv_grid.addLayout(idle_row, 5, 1)

            chk_sync_w1 = W.QCheckBox("同步锁相至 W1 (Master)")
            chk_sync_w1.setVisible(idx == 1)
            chk_sync_w1.toggled.connect(lambda _, i=idx: self._emit_awg_config(i))
            if idx == 1:
                adv_grid.addWidget(chk_sync_w1, 6, 0, 1, 2)

            # SSB / VSB hardware carrier synthesis
            adv_grid.addWidget(W.QLabel("SSB / VSB:"), 7, 0)
            cmb_ssb = _compact_combo(W.QComboBox(), 14)
            cmb_ssb.addItems(list(SSB_MODES))
            cmb_ssb.setToolTip("单边带 (USB/LSB) 与残留边带 (VSB) 合成，使用 AM 频率与调制度参数")
            cmb_ssb.currentTextChanged.connect(lambda txt, i=idx: self._on_ssb_changed(i, txt))
            adv_grid.addWidget(cmb_ssb, 7, 1)

            # Linear Frequency Sweep (Chirp)
            chk_sweep = W.QCheckBox("线性扫频 (Chirp)")
            chk_sweep.setToolTip("硬件线性调频扫频：从起始频率线性扫至结束频率，可设扫频周期、间隔和重复次数")
            chk_sweep.toggled.connect(lambda _, i=idx: self._emit_awg_config(i))
            adv_grid.addWidget(chk_sweep, 8, 0)
            sw_freq_row = W.QHBoxLayout()
            sw_freq_row.setSpacing(4)
            freq_sweep_start = FreqUnitInput(min_hz=1.0, max_hz=25_000_000.0, default_hz=100_000.0, allow_full=False)
            freq_sweep_start.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            freq_sweep_stop = FreqUnitInput(min_hz=1.0, max_hz=25_000_000.0, default_hz=1_000_000.0, allow_full=False)
            freq_sweep_stop.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            sw_freq_row.addWidget(freq_sweep_start, 1)
            sw_freq_row.addWidget(W.QLabel("→"))
            sw_freq_row.addWidget(freq_sweep_stop, 1)
            adv_grid.addLayout(sw_freq_row, 8, 1)

            adv_grid.addWidget(W.QLabel("周期 / 间隔 / 次数:"), 9, 0)
            sw_time_row = W.QHBoxLayout()
            sw_time_row.setSpacing(4)
            spn_sweep_ms = W.QDoubleSpinBox()
            spn_sweep_ms.setKeyboardTracking(False)
            spn_sweep_ms.setRange(0.05, 100_000.0)
            spn_sweep_ms.setDecimals(2)
            spn_sweep_ms.setValue(10.0)
            spn_sweep_ms.setSuffix(" ms周期")
            spn_sweep_ms.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))

            spn_sweep_wait_ms = W.QDoubleSpinBox()
            spn_sweep_wait_ms.setKeyboardTracking(False)
            spn_sweep_wait_ms.setRange(0.0, 100_000.0)
            spn_sweep_wait_ms.setDecimals(2)
            spn_sweep_wait_ms.setValue(0.0)
            spn_sweep_wait_ms.setSuffix(" ms间隔")
            spn_sweep_wait_ms.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))

            spn_sweep_rep = W.QSpinBox()
            spn_sweep_rep.setKeyboardTracking(False)
            spn_sweep_rep.setRange(0, 1_000_000)
            spn_sweep_rep.setValue(0)
            spn_sweep_rep.setSpecialValueText("无限 (0)")
            spn_sweep_rep.setSuffix(" 次")
            spn_sweep_rep.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            sw_time_row.addWidget(spn_sweep_ms, 1)
            sw_time_row.addWidget(spn_sweep_wait_ms, 1)
            sw_time_row.addWidget(spn_sweep_rep, 1)
            adv_grid.addLayout(sw_time_row, 9, 1)

            p_layout.addWidget(adv_box)
            sub.setVisible(idx in self._active_awg_channels)
            card.body_layout.addWidget(sub)
            self.awg_subcards.append(sub)
            self.awg_widgets.append((chk_en, cmb_fn, spn_freq, spn_amp, spn_off, spn_sym))
            self.awg_adv_widgets.append({
                "phase": spn_phase,
                "chk_am": chk_am,
                "cmb_am_fn": cmb_am_fn,
                "freq_am": freq_am,
                "cmb_am_wav": cmb_am_wav,
                "spn_am_depth": spn_am_depth,
                "chk_fm": chk_fm,
                "cmb_fm_fn": cmb_fm_fn,
                "freq_fm": freq_fm,
                "cmb_fm_wav": cmb_fm_wav,
                "spn_fm_depth": spn_fm_depth,
                "spn_run_ms": spn_run_ms,
                "spn_wait_ms": spn_wait_ms,
                "spn_repeat": spn_repeat,
                "chk_rep_trig": chk_rep_trig,
                "cmb_awg_trig": cmb_awg_trig,
                "cmb_awg_slope": cmb_awg_slope,
                "cmb_idle": cmb_idle,
                "spn_limit_v": spn_limit_v,
                "chk_sync_w1": chk_sync_w1,
                "btn_load_custom": btn_load_custom,
                "btn_edit_poly": btn_edit_poly,
                "cmb_ssb": cmb_ssb,
                "chk_sweep": chk_sweep,
                "freq_sweep_start": freq_sweep_start,
                "freq_sweep_stop": freq_sweep_stop,
                "spn_sweep_ms": spn_sweep_ms,
                "spn_sweep_wait_ms": spn_sweep_wait_ms,
                "spn_sweep_rep": spn_sweep_rep,
                "btn_adv": btn_adv,
                "adv_box": adv_box,
            })

        self._sync_awg_channel_buttons()
        return card

    def _init_worker(self):
        self.thread = QtCore.QThread(self)
        self.worker = InstrumentWorker(demo=bool(self.args.demo))
        self.worker.moveToThread(self.thread)

        self.req_connect.connect(self.worker.connect_device)
        self.req_disconnect.connect(self.worker.disconnect_device)
        self.req_demo.connect(self.worker.set_demo_mode)
        self.req_start.connect(self.worker.start_acquisition)
        self.req_stop.connect(self.worker.stop_acquisition)
        self.req_single.connect(self.worker.single_acquisition)
        self.req_scope_cfg.connect(self.worker.update_scope_config)
        self.req_awg_cfg.connect(self.worker.update_awg_config)
        self.req_pc_trig.connect(self.worker.trigger_pc)
        self.req_force_scope_trig.connect(self.worker.force_scope_trigger)
        self.req_fft_cfg.connect(self.worker.update_fft_settings)
        self.req_power_cfg.connect(self.worker.update_power_supply)

        self.worker.connected.connect(self._on_device_connected)
        self.worker.status_changed.connect(self.lbl_status.setText)
        self.worker.error_occurred.connect(self._on_worker_error)
        self.worker.frame_ready.connect(self._on_frame_ready)

        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()

        QtCore.QTimer.singleShot(60, self.req_connect.emit)

    def _on_device_connected(self, info: dict):
        short_name = info["name"].replace("Analog Discovery 3", "AD3")
        self.lbl_dev_badge.setText(f"设备: {short_name} ({info['serial']})")
        self.lbl_gpu_badge.setText(f"⚡ {info['fft_device']}")

        backends = info.get("fft_backends", [])
        if backends:
            with QtCore.QSignalBlocker(self.cmb_fft_backend):
                self.cmb_fft_backend.clear()
                for code, label in backends:
                    self.cmb_fft_backend.addItem(label, code)

        # Apply initial scope and FFT settings
        if self.chk_pwr.isChecked():
            self._emit_power_config()
        self._emit_scope_config()
        self._emit_fft_config()

    def _on_worker_error(self, msg: str):
        self.lbl_status.setText(f"错误: {msg}")

    def _on_dpo_toggled(self, checked: bool):
        for curve in (self.scope_curve_ch1, self.scope_curve_ch2):
            curve.set_dpo(checked)

    def _step_timebase(self, delta: int):
        idx = int(np.clip(self.cmb_timebase.currentIndex() + delta, 0, self.cmb_timebase.count() - 1))
        self.cmb_timebase.setCurrentIndex(idx)

    def _step_voltage_range(self, delta: int):
        for ch_idx in sorted(self._active_scope_channels):
            cmb_range = self.ch_controls[ch_idx][0]
            idx = int(np.clip(cmb_range.currentIndex() + delta, 0, cmb_range.count() - 1))
            cmb_range.setCurrentIndex(idx)
            break

    def _on_timebase_changed(self, *_):
        self._apply_timebase_view()
        self._emit_scope_config()

    def _on_channel_range_changed(self, *_):
        self._apply_vertical_view()
        self._emit_scope_config()

    def _apply_timebase_view(self):
        tb = float(self.cmb_timebase.currentData() or 5e-6)
        pos_s = float(self.spn_trig_pos.value()) * 1e-6
        half_span = 5.0 * tb
        x0, x1 = pos_s - half_span, pos_s + half_span
        self.scope_plot.setXRange(x0, x1, padding=0.0)
        if hasattr(self, "overview_bar"):
            self.overview_bar.set_region(x0, x1)

    def _on_scope_xrange_changed(self, *_):
        if not hasattr(self, "overview_bar"):
            return
        xr, _ = self.scope_vb.viewRange()
        self.overview_bar.set_region(float(xr[0]), float(xr[1]))

    def _on_overview_region_changed(self, r0: float, r1: float):
        with QtCore.QSignalBlocker(self.scope_vb):
            self.scope_plot.setXRange(float(r0), float(r1), padding=0.0)

    def _set_scope_channel_open(self, idx: int, is_open: bool):
        if is_open:
            self._active_scope_channels.add(idx)
            self.card_ch.set_expanded(True, emit=False)
        else:
            self._active_scope_channels.discard(idx)
        self.ch_subcards[idx].setVisible(is_open)
        self._sync_scope_channel_buttons()
        self._apply_vertical_view()
        self._emit_scope_config()

    def _sync_scope_channel_buttons(self):
        for idx, b in enumerate(self.btn_add_ch_btns):
            b.setVisible(idx not in self._active_scope_channels)
        self.lbl_scope_empty.setVisible(len(self._active_scope_channels) == 0)

    def _set_awg_channel_open(self, idx: int, is_open: bool):
        if is_open:
            self._active_awg_channels.add(idx)
            self.card_awg.set_expanded(True, emit=False)
        else:
            self._active_awg_channels.discard(idx)
            chk_en = self.awg_widgets[idx][0]
            if chk_en.isChecked():
                chk_en.setChecked(False)
            else:
                self._emit_awg_config(idx)
        self.awg_subcards[idx].setVisible(is_open)
        self._sync_awg_channel_buttons()

    def _sync_awg_channel_buttons(self):
        for idx, b in enumerate(self.btn_add_awg_btns):
            b.setVisible(idx not in self._active_awg_channels)
        self.lbl_awg_empty.setVisible(len(self._active_awg_channels) == 0)

    @staticmethod
    def _unpack_voltage_range(data) -> tuple[float, float]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return float(data[0]), float(data[1])
        val = float(data or 2.0)
        return val, val

    def _apply_vertical_view(self):
        # Match oscilloscope vertical window to the active primary channel display span, attenuation & offset.
        # During software zoom (disp_span < hw_range_v), scale around 0 V so the 0-V baseline never shifts on screen.
        for idx in sorted(self._active_scope_channels):
            cmb_range, _, _, spn_offset, cmb_atten, _ = self.ch_controls[idx]
            hw_range_v, disp_span = self._unpack_voltage_range(cmb_range.currentData())
            atten = float(cmb_atten.currentData() or 1.0)
            offset_v = float(spn_offset.value())
            hw_half = 0.5 * hw_range_v * atten
            sw_scale = disp_span / max(hw_range_v, 1e-9)
            y0 = (offset_v - hw_half) * sw_scale
            y1 = (offset_v + hw_half) * sw_scale
            self.scope_plot.setYRange(y0, y1, padding=0.0)
            break

    def _on_fft_xrange_changed(self, *_):
        if hasattr(self, "waterfall_overlay") and self.waterfall_overlay.isVisible():
            fx_r, _ = self.fft_vb.viewRange()
            self.waterfall_overlay.sync_x_range(float(fx_r[0]), float(fx_r[1]))

    def _on_trig_line_dragged(self):
        val = float(self.trig_line.value())
        with QtCore.QSignalBlocker(self.spn_trig_lvl):
            self.spn_trig_lvl.setValue(val)
        self._emit_scope_config()

    def _on_trig_spin_changed(self, val: float):
        self.trig_line.setValue(float(val))
        self._emit_scope_config()

    def _on_trig_pos_dragged(self):
        pos_s = float(self.trig_pos_line.value())
        with QtCore.QSignalBlocker(self.spn_trig_pos):
            self.spn_trig_pos.setValue(pos_s * 1e6)
        self._apply_timebase_view()
        self._emit_scope_config()

    def _on_trig_pos_spin_changed(self, val_us: float):
        self.trig_pos_line.setValue(float(val_us) * 1e-6)
        self._apply_timebase_view()
        self._emit_scope_config()

    def _on_fft_span_changed(self, *_):
        if self._last_frame_data is not None:
            fft_dict = self._last_frame_data.get("fft", {})
            primary_fft = fft_dict.get("ch1") or fft_dict.get("ch2")
            if primary_fft:
                self._update_fft_x_range(primary_fft)
        if hasattr(self, "chk_fft_zoom") and self.chk_fft_zoom.isChecked():
            self._emit_fft_config()

    def _on_acq_mode_changed(self, text: str):
        self.spn_rec_len.setVisible(text.startswith("Record"))
        self._emit_scope_config()

    def _on_trig_type_changed(self, text: str):
        need_len = text.startswith("Pulse") or text.startswith("Transition")
        self.lbl_trig_len.setVisible(need_len)
        self.cmb_trig_len_cond.setVisible(need_len)
        self.spn_trig_len_us.setVisible(need_len)
        self._emit_scope_config()

    def _emit_scope_config(self, *_):
        tb = float(self.cmb_timebase.currentData() or 5e-6)
        buf = int(self.cmb_buf.currentData() or 32768)
        rate_sel = float(self.cmb_rate.currentData() or 0.0)
        auto_sr = rate_sel <= 0.0
        fs = select_optimal_sample_rate(tb, buffer_size=buf) if auto_sr else rate_sel

        ch_cfgs = []
        primary_disp_span = 2.0
        for idx, (cmb_range, chk_ac, cmb_flt, spn_offset, cmb_atten, cmb_hw_flt) in enumerate(self.ch_controls):
            flt_val = cmb_flt.currentData() or (0.0, False)
            lpf_hz, show_env = float(flt_val[0]), bool(flt_val[1])
            hw_range_v, disp_span = self._unpack_voltage_range(cmb_range.currentData())
            enabled = idx in self._active_scope_channels
            if enabled and idx == min(self._active_scope_channels):
                primary_disp_span = disp_span * float(cmb_atten.currentData() or 1.0)
            ch_cfgs.append(
                ChannelConfig(
                    enabled=enabled,
                    range_v=hw_range_v,
                    offset_v=float(spn_offset.value()),
                    coupling_ac=chk_ac.isChecked(),
                    attenuation=float(cmb_atten.currentData() or 1.0),
                    hw_filter=cmb_hw_flt.currentText(),
                    lpf_cutoff_hz=lpf_hz,
                    show_envelope=show_env,
                )
            )
        user_hyst = float(self.spn_trig_hyst.value())
        eff_hyst = min(user_hyst, max(0.0005, 0.01 * primary_disp_span)) if primary_disp_span <= 1.0 else user_hyst
        self.spn_trig_lvl.setDecimals(3 if primary_disp_span < 0.5 else 2)
        self.spn_trig_lvl.setSingleStep(max(0.001, round(primary_disp_span * 0.02, 4)))
        trig = TriggerConfig(
            mode=self.cmb_trig_mode.currentText(),
            source=self.cmb_trig_src.currentText(),
            slope=self.cmb_trig_slope.currentText(),
            level_v=float(self.spn_trig_lvl.value()),
            position_s=float(self.spn_trig_pos.value()) * 1e-6,
            coupling=str(self.cmb_trig_coupling.currentData() or "Auto"),
            holdoff_s=float(self.spn_trig_holdoff.value()) * 1e-6,
            trig_type=self.cmb_trig_type.currentText(),
            length_s=float(self.spn_trig_len_us.value()) * 1e-6,
            length_cond=self.cmb_trig_len_cond.currentText(),
            hysteresis_v=eff_hyst,
            hw_filter=self.cmb_trig_hw_flt.currentText(),
        )
        self.scope_cfg = ScopeConfig(
            sample_rate_hz=fs,
            buffer_size=buf,
            timebase_s_per_div=tb,
            auto_sample_rate=auto_sr,
            acq_mode=self.cmb_acq_mode.currentText(),
            record_length_s=float(self.spn_rec_len.value()),
            hw_counter_enabled=self.chk_hw_counter.isChecked(),
            channels=ch_cfgs,
            trigger=trig,
        )
        self.scope_curve_ch1.setVisible(ch_cfgs[0].enabled)
        self.scope_curve_ch2.setVisible(ch_cfgs[1].enabled)
        self.scope_env_ch1.setVisible(ch_cfgs[0].enabled and ch_cfgs[0].show_envelope)
        self.fft_curve_ch1.setVisible(ch_cfgs[0].enabled and self.chk_fft_en.isChecked())
        self.fft_curve_ch2.setVisible(ch_cfgs[1].enabled and self.chk_fft_en.isChecked())

        tb_txt = format_si(tb, "s/div", decimals=0 if tb >= 1e-6 else 1)
        fs_txt = format_si(fs, "S/s", decimals=0)
        flt_desc = "LPF " + format_si(ch_cfgs[0].lpf_cutoff_hz, "Hz", 0) if ch_cfgs[0].lpf_cutoff_hz > 0 else "全带宽"
        self.scope_plot.setTitle(
            f"Time Domain · 实时稳相示波器 (时基: {tb_txt} · 采样率: {fs_txt} · CH1: {flt_desc})",
            color="#9fb3c8",
            size="11pt",
        )
        self.req_scope_cfg.emit(self.scope_cfg)

    def _on_awg_fn_changed(self, idx: int, txt: str):
        adv = self.awg_adv_widgets[idx]
        adv["btn_load_custom"].setVisible(txt == "Custom (自定义)")
        is_poly = txt == "Polynomial (多正弦)"
        adv["btn_edit_poly"].setVisible(is_poly)
        if is_poly and not self.awg_poly_terms[idx]:
            base_f = float(self.awg_widgets[idx][2].value())
            self.awg_poly_terms[idx] = [(base_f, 1.0, 0.0), (min(25_000_000.0, base_f * 1.2), 0.5, 0.0)]
            self._edit_poly_awg_waveform(idx)
        else:
            self._emit_awg_config(idx)

    def _edit_poly_awg_waveform(self, idx: int):
        base_f = float(self.awg_widgets[idx][2].value())
        cur_terms = self.awg_poly_terms[idx] or [(base_f, 1.0, 0.0), (min(25_000_000.0, base_f * 1.2), 0.5, 0.0)]
        dlg = PolynomialWaveformDialog(idx, cur_terms, base_f, parent=self)
        if dlg.exec() == W.QDialog.DialogCode.Accepted:
            terms = dlg.get_terms()
            if terms:
                self.awg_poly_terms[idx] = terms
                self.lbl_status.setText(f"W{idx + 1} 多项式波形已更新 ({len(terms)} 个正弦项)")
                self._emit_awg_config(idx)

    def _load_custom_awg_waveform(self, idx: int):
        path, _ = W.QFileDialog.getOpenFileName(
            self,
            f"加载 W{idx + 1} 自定义波形",
            "",
            "Waveform Files (*.csv *.txt *.npy);;All Files (*)",
        )
        if not path:
            return
        try:
            if path.lower().endswith(".npy"):
                arr = np.load(path).astype(np.float64).ravel()
            else:
                arr = np.loadtxt(path, delimiter=",", dtype=np.float64).ravel()
            if arr.size < 2:
                raise ValueError("采样点数不足 2 点")
            peak = float(np.max(np.abs(arr)))
            if peak > 1.0:
                arr = arr / peak
            self.awg_custom_samples[idx] = arr[:4096].tolist()
            self.lbl_status.setText(f"W{idx + 1} 已加载自定义波形 ({len(self.awg_custom_samples[idx])} pts)")
            self._emit_awg_config(idx)
        except Exception as exc:
            self.lbl_status.setText(f"加载自定义波形失败: {exc}")

    @staticmethod
    def _populate_wav_combo(cmb: W.QComboBox) -> W.QComboBox:
        hints = {
            "fft_bounce_10k_90k_4s.wav": "10~90kHz 往返弹跳, 4.0s",
            "fft_arpeggio_chord_3s.wav": "10~75kHz 跳频琶音, 3.0s",
            "fft_chirp_5k_to_80k_3s.wav": "5~80kHz 线性扫频, 3.0s",
            "fft_step_5k_to_80k_2s.wav": "5~80kHz 阶梯跃迁, 2.0s",
            "fft_radar_burst_20k_80k_3s.wav": "20/50/80kHz 雷达脉冲, 3.0s",
            "fft_tremolo_60k_2s.wav": "±60kHz 呼吸起伏, 2.0s",
            "fft_morse_30k_2s.wav": "±30kHz 摩尔斯, 2.0s",
        }
        audio_dir = Path(__file__).resolve().parents[2] / "resources" / "audio"
        if audio_dir.is_dir():
            # Put wideband FFT clips first so they are immediately discoverable
            files = sorted(audio_dir.glob("*.wav"), key=lambda p: (0 if p.name.startswith("fft_") else 1, p.name))
            for wf_path in files:
                hint = hints.get(wf_path.name)
                if not hint:
                    try:
                        with wave.open(str(wf_path), "rb") as wf:
                            sr = wf.getframerate()
                            dur = wf.getnframes() / max(sr, 1)
                        hint = f"≤{sr/2000:.1f}kHz, {dur:.1f}s"
                    except Exception:
                        hint = "WAV"
                cmb.addItem(f"{wf_path.name} [{hint}]", str(wf_path))
        cmb.addItem("📂 选择本地 WAV…", "__browse__")
        return cmb

    def _on_mod_fn_changed(self, idx: int, kind: str, txt: str):
        adv = self.awg_adv_widgets[idx]
        is_wav = txt == "WAV (音频文件)"
        adv[f"freq_{kind}"].setVisible(not is_wav)
        adv[f"cmb_{kind}_wav"].setVisible(is_wav)
        store = self.awg_am_wav if kind == "am" else self.awg_fm_wav
        if is_wav and store[idx] is None:
            self._on_mod_wav_selected(idx, kind)
        else:
            self._emit_awg_config(idx)

    def _on_mod_wav_selected(self, idx: int, kind: str):
        cmb = self.awg_adv_widgets[idx][f"cmb_{kind}_wav"]
        data = cmb.currentData()
        if data == "__browse__":
            path, _ = W.QFileDialog.getOpenFileName(
                self, f"选择 W{idx + 1} {kind.upper()} 音频 (WAV)", "", "WAV Audio (*.wav);;All Files (*)"
            )
            if not path:
                return
            self._load_mod_wav_file(idx, kind, path)
        elif data:
            self._load_mod_wav_file(idx, kind, str(data))

    def _load_mod_wav_file(self, idx: int, kind: str, path: str):
        try:
            with wave.open(path, "rb") as wf:
                n_ch = wf.getnchannels()
                sw = wf.getsampwidth()
                sr = float(wf.getframerate())
                raw_bytes = wf.readframes(wf.getnframes())
            if sw == 1:
                arr = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
            elif sw == 2:
                arr = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float64) / 32768.0
            elif sw == 4:
                arr = np.frombuffer(raw_bytes, dtype="<i4").astype(np.float64) / 2147483648.0
            else:
                raise ValueError(f"不支持的 WAV 位深: {sw * 8}-bit")
            if n_ch > 1:
                arr = arr.reshape(-1, n_ch).mean(axis=1)
            peak = float(np.max(np.abs(arr))) if arr.size else 0.0
            if peak > 1e-6:
                arr = arr / peak
            store = self.awg_am_wav if kind == "am" else self.awg_fm_wav
            store[idx] = (arr.tolist(), sr, Path(path).name)
            self.lbl_status.setText(
                f"W{idx + 1} {kind.upper()} 已加载: {Path(path).name} ({arr.size} pts @ {int(sr)} Hz)"
            )
            self._emit_awg_config(idx)
        except Exception as exc:
            self.lbl_status.setText(f"加载 WAV 失败: {exc}")

    def _emit_awg_config(self, idx: int):
        chk_en, cmb_fn, spn_freq, spn_amp, spn_off, spn_sym = self.awg_widgets[idx]
        adv = self.awg_adv_widgets[idx]
        am_wav = self.awg_am_wav[idx]
        fm_wav = self.awg_fm_wav[idx]
        cfg = AwgChannelConfig(
            enabled=(idx in self._active_awg_channels) and chk_en.isChecked(),
            function=cmb_fn.currentText(),
            frequency_hz=float(spn_freq.value()),
            amplitude_v=float(spn_amp.value()),
            offset_v=float(spn_off.value()),
            symmetry_pct=float(spn_sym.value()),
            phase_deg=float(adv["phase"].value()),
            custom_samples=self.awg_custom_samples[idx],
            poly_terms=self.awg_poly_terms[idx],
            am_enabled=adv["chk_am"].isChecked(),
            am_function=adv["cmb_am_fn"].currentText(),
            am_frequency_hz=float(adv["freq_am"].value()),
            am_depth_pct=float(adv["spn_am_depth"].value()),
            am_wav_samples=am_wav[0] if am_wav else None,
            am_wav_rate_hz=am_wav[1] if am_wav else 16_000.0,
            fm_enabled=adv["chk_fm"].isChecked(),
            fm_function=adv["cmb_fm_fn"].currentText(),
            fm_frequency_hz=float(adv["freq_fm"].value()),
            fm_depth_pct=float(adv["spn_fm_depth"].value()),
            fm_wav_samples=fm_wav[0] if fm_wav else None,
            fm_wav_rate_hz=fm_wav[1] if fm_wav else 16_000.0,
            ssb_mode=adv["cmb_ssb"].currentText(),
            sweep_enabled=adv["chk_sweep"].isChecked(),
            sweep_start_hz=float(adv["freq_sweep_start"].value()),
            sweep_stop_hz=float(adv["freq_sweep_stop"].value()),
            sweep_time_s=float(adv["spn_sweep_ms"].value()) * 1e-3,
            sweep_wait_s=float(adv["spn_sweep_wait_ms"].value()) * 1e-3,
            sweep_repeat=int(adv["spn_sweep_rep"].value()),
            run_s=float(adv["spn_run_ms"].value()) * 1e-3,
            wait_s=float(adv["spn_wait_ms"].value()) * 1e-3,
            repeat_count=int(adv["spn_repeat"].value()),
            repeat_trigger=adv["chk_rep_trig"].isChecked(),
            trigger_source=adv["cmb_awg_trig"].currentText(),
            trigger_slope=adv["cmb_awg_slope"].currentText(),
            idle_mode=adv["cmb_idle"].currentText(),
            sync_to_master=adv["chk_sync_w1"].isChecked() if idx == 1 else False,
            limitation_v=float(adv["spn_limit_v"].value()),
        )
        self.awg_cfgs[idx] = cfg
        self.req_awg_cfg.emit(idx, cfg)

    def _on_ssb_changed(self, idx: int, txt: str):
        self._emit_awg_config(idx)

    def _on_waterfall_closed(self):
        for btn in (self.btn_waterfall, self.btn_top_waterfall):
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)

    def _toggle_waterfall(self, checked: bool):
        for btn in (self.btn_waterfall, self.btn_top_waterfall):
            if btn.isChecked() != checked:
                btn.blockSignals(True)
                btn.setChecked(checked)
                btn.blockSignals(False)
        if checked:
            cw = self.centralWidget()
            if self.isVisible() and cw is not None and cw.width() > self.waterfall_overlay.width() + 24 and cw.height() > self.waterfall_overlay.height() + 68:
                x = max(12, min(self.waterfall_overlay.x(), cw.width() - self.waterfall_overlay.width() - 12))
                y = max(56, min(self.waterfall_overlay.y(), cw.height() - self.waterfall_overlay.height() - 12))
                self.waterfall_overlay.move(x, y)
            self.waterfall_overlay.show()
            self.waterfall_overlay.raise_()
            fx_r, _ = self.fft_vb.viewRange()
            self.waterfall_overlay.sync_x_range(float(fx_r[0]), float(fx_r[1]))
        else:
            self.waterfall_overlay.hide()

    def _clear_fft_hold(self):
        self.req_fft_cfg.emit({"clear_hold": True})

    def _on_fft_unit_changed(self, *_):
        unit = self.cmb_fft_unit.currentData() or "dbv"
        if unit == "dbm":
            self.fft_plot.setLabel("left", "Magnitude", units="dBm", color="#8b9bb0")
            self.fft_plot.getViewBox().setLimits(yMin=-130.0, yMax=45.0, minYRange=10.0, maxYRange=175.0)
            self.fft_plot.setYRange(-92.0, 23.0)
        elif unit == "vrms":
            self.fft_plot.setLabel("left", "Magnitude", units="Vrms", color="#8b9bb0")
            self.fft_plot.getViewBox().setLimits(yMin=0.0, yMax=10.0, minYRange=1e-4, maxYRange=10.0)
            self.fft_plot.setYRange(0.0, 1.2)
        else:
            self.fft_plot.setLabel("left", "Magnitude", units="dBV", color="#8b9bb0")
            self.fft_plot.getViewBox().setLimits(yMin=-140.0, yMax=30.0, minYRange=10.0, maxYRange=170.0)
            self.fft_plot.setYRange(-105.0, 10.0)

    def _fft_convert_mag(self, dbv_arr: np.ndarray) -> np.ndarray:
        unit = self.cmb_fft_unit.currentData() or "dbv"
        if unit == "dbm":
            return dbv_arr + np.float32(13.0103)
        if unit == "vrms":
            return np.power(np.float32(10.0), dbv_arr * np.float32(0.05))
        return dbv_arr

    def _fft_format_mag(self, dbv_val: float) -> str:
        unit = self.cmb_fft_unit.currentData() or "dbv"
        if unit == "dbm":
            return f"{dbv_val + 13.0103:6.1f} dBm"
        if unit == "vrms":
            v = 10.0 ** (dbv_val / 20.0)
            return f"{format_si(v, 'Vrms', 2)}"
        return f"{dbv_val:6.1f} dBV"

    def _emit_fft_config(self, *_):
        span_hz = float(self.fft_span_input.value())
        settings = {
            "enabled": self.chk_fft_en.isChecked(),
            "backend": self.cmb_fft_backend.currentData() or "opencl",
            "window": self.cmb_fft_win.currentData() or "FlatTop",
            "average": int(self.spn_fft_avg.value()),
            "n_fft": int(self.cmb_fft_size.currentData() or 0),
            "trace_mode": self.cmb_fft_trace.currentData() or "live",
            "zoom_enabled": self.chk_fft_zoom.isChecked(),
            "zoom_stop_hz": span_hz if span_hz > 0.0 else float(self.scope_cfg.sample_rate_hz * 0.5),
            "goertzel_enabled": self.chk_goertzel.isChecked(),
            "goertzel_freq_hz": float(self.goertzel_freq.value()),
        }
        self.fft_curve_ch1.setVisible(self.scope_cfg.channels[0].enabled and settings["enabled"])
        self.fft_curve_ch2.setVisible(self.scope_cfg.channels[1].enabled and settings["enabled"])
        self.req_fft_cfg.emit(settings)

    def auto_set_scope(self):
        """Intelligently set Time/div and Vertical Range to frame ~4 periods of the signal (preferring IF envelope frequency on mixer signals)."""
        if self._last_frame_data is None:
            self.autoscale_views()
            return
        meas = self._last_frame_data.get("meas", {})
        m1 = meas.get("ch1", {})
        m2 = meas.get("ch2", {})
        use_ch1 = self.scope_cfg.channels[0].enabled and bool(m1)
        primary_m = m1 if use_ch1 else m2
        if not primary_m:
            self.autoscale_views()
            return

        # Prefer IF envelope frequency, then time-domain freq, then hardware counter, then FFT fundamental peak
        target_f = primary_m.get("if_freq_hz", 0.0) or primary_m.get("freq_hz", 0.0)
        if target_f <= 10.0:
            target_f = float(self._last_frame_data.get("hw_freq_hz") or 0.0)
        if target_f <= 10.0:
            fft_dict = self._last_frame_data.get("fft", {})
            ch_fft = fft_dict.get("ch1" if use_ch1 else "ch2") or fft_dict.get("ch1") or fft_dict.get("ch2")
            if ch_fft:
                target_f = float(ch_fft.get("metrics", {}).get("fund_freq", 0.0) or 0.0)

        if target_f > 10.0:
            ideal_tb = (4.0 / target_f) / 10.0  # 4 periods across 10 divisions
            best_idx = min(range(len(TIMEBASES)), key=lambda i: abs(np.log10(TIMEBASES[i][1] / ideal_tb)))
            self.cmb_timebase.setCurrentIndex(best_idx)

        vpp = primary_m.get("vpp", 1.0)
        if vpp > 0.001:
            needed_span = max(0.01, vpp * 1.35)
            ch_idx = 0 if use_ch1 else 1
            cmb_range = self.ch_controls[ch_idx][0]
            matched_idx = len(VOLTAGE_RANGES) - 1
            for i, (_, r_data) in enumerate(VOLTAGE_RANGES):
                _, disp_span = self._unpack_voltage_range(r_data)
                if disp_span >= needed_span:
                    matched_idx = i
                    break
            cmb_range.setCurrentIndex(matched_idx)
        self._apply_timebase_view()
        self._apply_vertical_view()

    def autoscale_views(self):
        self._apply_timebase_view()
        self._apply_vertical_view()
        if self._last_frame_data is not None:
            fft_dict = self._last_frame_data.get("fft", {})
            primary_fft = fft_dict.get("ch1") or fft_dict.get("ch2")
            if primary_fft:
                self._update_fft_x_range(primary_fft)
        self._on_fft_unit_changed()

    def _update_fft_x_range(self, primary_fft: dict, saved_xrange: tuple[float, float] | None = None):
        freqs = primary_fft["freqs"]
        nyq = float(freqs[-1]) if freqs.size > 1 else 1e6
        span_sel = float(self.fft_span_input.value())
        fmax = nyq if span_sel <= 0.0 else min(nyq, max(5_000.0, span_sel))
        min_x_rng = min(5_000.0, fmax)
        self.fft_plot.getViewBox().setLimits(xMin=0.0, xMax=fmax, minXRange=min_x_rng, maxXRange=fmax)
        if saved_xrange is not None and saved_xrange[1] > saved_xrange[0]:
            x0 = max(0.0, min(saved_xrange[0], fmax - min_x_rng))
            x1 = min(fmax, max(x0 + min_x_rng, saved_xrange[1]))
            self.fft_plot.setXRange(x0, x1, padding=0.0)
        else:
            self.fft_plot.setXRange(0.0, fmax, padding=0.0)

    def _on_frame_ready(self, data: dict):
        self._last_frame_data = data
        t = data["t"]
        ch1 = data["ch1"]
        ch2 = data["ch2"]
        ch1_env = data.get("ch1_env")
        telemetry_updated = data.get("telemetry_updated", True)

        if t.size >= 2:
            buf_span = max(float(t[-1] - t[0]), 1e-6)
            half_span = 0.5 * buf_span
            pos_s = float(self.spn_trig_pos.value()) * 1e-6
            lim_min = pos_s - half_span
            lim_max = pos_s + half_span
            prev = self._last_overview_bounds
            tol = max(1e-9, 1e-6 * buf_span)
            if prev is None or abs(lim_min - prev[0]) > tol or abs(lim_max - prev[1]) > tol:
                self._last_overview_bounds = (lim_min, lim_max)
                self.overview_bar.set_bounds(lim_min, lim_max)
                self.scope_vb.setLimits(xMin=lim_min, xMax=lim_max, minXRange=500e-9, maxXRange=buf_span)

        if self.scope_cfg.channels[0].enabled:
            self.scope_curve_ch1.set_fast_data(t, ch1)
            n1 = data.get("ch1_noise")
            if n1 is not None:
                n_pts = min(t.size, n1[0].size)
                step = max(1, t.size // n_pts)
                t_n = t[: n_pts * step : step][:n_pts]
                self.scope_noise_ch1_min.setData(t_n, n1[0][: t_n.size], skipFiniteCheck=True)
                self.scope_noise_ch1_max.setData(t_n, n1[1][: t_n.size], skipFiniteCheck=True)
                self.scope_noise_ch1_min.setVisible(True)
                self.scope_noise_ch1_max.setVisible(True)
            else:
                self.scope_noise_ch1_min.setVisible(False)
                self.scope_noise_ch1_max.setVisible(False)
            if self.scope_cfg.channels[0].show_envelope and ch1_env is not None:
                env_step = max(1, t.size // 1200)
                self.scope_env_ch1.setData(t[::env_step], ch1_env[::env_step], skipFiniteCheck=True)
                self.scope_env_ch1.setVisible(True)
            else:
                self.scope_env_ch1.setVisible(False)
        else:
            self.scope_env_ch1.setVisible(False)
            self.scope_noise_ch1_min.setVisible(False)
            self.scope_noise_ch1_max.setVisible(False)

        if self.scope_cfg.channels[1].enabled:
            self.scope_curve_ch2.set_fast_data(t, ch2)
            n2 = data.get("ch2_noise")
            if n2 is not None:
                n_pts = min(t.size, n2[0].size)
                step = max(1, t.size // n_pts)
                t_n = t[: n_pts * step : step][:n_pts]
                self.scope_noise_ch2_min.setData(t_n, n2[0][: t_n.size], skipFiniteCheck=True)
                self.scope_noise_ch2_max.setData(t_n, n2[1][: t_n.size], skipFiniteCheck=True)
                self.scope_noise_ch2_min.setVisible(True)
                self.scope_noise_ch2_max.setVisible(True)
            else:
                self.scope_noise_ch2_min.setVisible(False)
                self.scope_noise_ch2_max.setVisible(False)
        else:
            self.scope_noise_ch2_min.setVisible(False)
            self.scope_noise_ch2_max.setVisible(False)

        fft_dict = data.get("fft", {})
        fft_ms = 0.0
        f1 = fft_dict.get("ch1")
        f2 = fft_dict.get("ch2")

        if f1:
            self.fft_curve_ch1.set_fast_data(f1["freqs"], self._fft_convert_mag(f1["dbv"]))
            fft_ms += f1.get("compute_ms", 0.0)
        if f2:
            self.fft_curve_ch2.set_fast_data(f2["freqs"], self._fft_convert_mag(f2["dbv"]))
            fft_ms += f2.get("compute_ms", 0.0)

        primary_fft = f1 or f2
        if primary_fft:
            if not getattr(self, "_fft_x_initialized", False):
                self._fft_x_initialized = True
                saved_xr = getattr(self, "_saved_fft_xrange", None)
                self._saved_fft_xrange = None
                self._update_fft_x_range(primary_fft, saved_xrange=saved_xr)

            if self.waterfall_overlay.isVisible():
                fx_r, _ = self.fft_plot.getViewBox().viewRange()
                self.waterfall_overlay.push_spectrum(
                    primary_fft["freqs"],
                    primary_fft["dbv"],
                    float(fx_r[0]),
                    float(fx_r[1]),
                )

        if not telemetry_updated:
            return

        meas = data.get("meas", {})
        m1 = meas.get("ch1", {})
        m2 = meas.get("ch2", {})
        hw_f = data.get("hw_freq_hz")
        hw_tag = f" | 硬件计频: {_format_si_fixed(hw_f, 'Hz', 3, 10)}" if hw_f else ""
        trig_tag = " [Auto超时]" if data.get("auto_triggered") else ""
        if m1 and self.scope_cfg.channels[0].enabled:
            if_f = m1.get("if_freq_hz", 0.0)
            if_vpp = m1.get("if_vpp", 0.0)
            if if_f > 10.0 and m1["freq_hz"] > if_f * 1.8:
                self._set_label_if_changed(
                    self.lbl_meas_ch1,
                    f"总Vpp: {_format_si_fixed(m1['vpp'], 'V', 2, 8)} | Vrms: {_format_si_fixed(m1['vrms'], 'V', 2, 9)} | "
                    f"IF差频(绿虚线): {_format_si_fixed(if_f, 'Hz', 2, 10)} ({_format_si_fixed(if_vpp, 'V', 2, 9)}pp) | "
                    f"和频/载波: {_format_si_fixed(m1['freq_hz'], 'Hz', 2, 10)}{hw_tag}{trig_tag}",
                )
            else:
                self._set_label_if_changed(
                    self.lbl_meas_ch1,
                    f"Vpp: {_format_si_fixed(m1['vpp'], 'V', 2, 8)} | Vrms: {_format_si_fixed(m1['vrms'], 'V', 2, 9)} | "
                    f"Freq: {_format_si_fixed(m1['freq_hz'], 'Hz', 2, 10)} | Period: {_format_si_fixed(m1['period_s'], 's', 2, 9)} | "
                    f"Duty: {m1['duty_pct']:5.1f}%{hw_tag}{trig_tag}",
                )
        else:
            self._set_label_if_changed(self.lbl_meas_ch1, "已禁用")

        if m2 and self.scope_cfg.channels[1].enabled:
            self._set_label_if_changed(
                self.lbl_meas_ch2,
                f"Vpp: {_format_si_fixed(m2['vpp'], 'V', 2, 8)} | Vrms: {_format_si_fixed(m2['vrms'], 'V', 2, 9)} | "
                f"Freq: {_format_si_fixed(m2['freq_hz'], 'Hz', 2, 10)}",
            )
        else:
            self._set_label_if_changed(self.lbl_meas_ch2, "已禁用")

        if primary_fft:
            met = primary_fft.get("metrics", {})
            if met:
                peaks = met.get("peaks", [])
                fx_r, _ = self.fft_plot.getViewBox().viewRange()
                vis_peaks = GpuFftEngine.find_peaks_in_band(
                    primary_fft["freqs"],
                    primary_fft["dbv"],
                    float(fx_r[0]),
                    float(fx_r[1]),
                    max_peaks=8,
                )
                if not vis_peaks:
                    vis_peaks = [p for p in peaks if fx_r[0] <= p[0] <= fx_r[1]] or peaks
                if vis_peaks:
                    px = [p[0] for p in vis_peaks[:12]]
                    py_dbv = np.asarray([p[1] for p in vis_peaks[:12]], dtype=np.float32)
                    py = self._fft_convert_mag(py_dbv).tolist()
                    self.fft_peaks_scatter.setData(px, py)
                    f_span = max(1.0, float(fx_r[1] - fx_r[0]))
                    for idx_lbl, lbl in enumerate(self._peak_labels):
                        if idx_lbl < len(vis_peaks):
                            pf, pd = vis_peaks[idx_lbl]
                            py_disp = py[idx_lbl]
                            rel_x = (pf - fx_r[0]) / f_span
                            ax_anc = 0.0 if rel_x < 0.08 else (1.0 if rel_x > 0.92 else 0.5)
                            lbl.setAnchor((ax_anc, 1.12))
                            lbl.setText(f"{format_si(pf, 'Hz', 2)}\n{self._fft_format_mag(pd).strip()}")
                            lbl.setPos(pf, py_disp)
                            lbl.setVisible(True)
                        else:
                            lbl.setVisible(False)

                    # Format top 3 visible peaks sorted by frequency for intuitive mixer/sideband reading
                    top_by_mag = vis_peaks[:3]
                    top_sorted_f = sorted(top_by_mag, key=lambda item: item[0])
                    peaks_str = "  ·  ".join(
                        f"f{i+1}: {_format_si_fixed(pf, 'Hz', 2, 10)} ({self._fft_format_mag(pd)})"
                        for i, (pf, pd) in enumerate(top_sorted_f)
                    )
                    self._set_label_if_changed(
                        self.lbl_meas_fft,
                        f"{peaks_str}  |  SNR: {met['snr_db']:5.1f} dB  |  SFDR: {met['sfdr_db']:5.1f} dB",
                    )
                else:
                    self.fft_peaks_scatter.setData([], [])
                    for lbl in self._peak_labels:
                        lbl.setVisible(False)
                    self._set_label_if_changed(self.lbl_meas_fft, "无显著波峰 (底噪状态)")
            gz = primary_fft.get("goertzel")
            if gz and self.chk_goertzel.isChecked():
                self._set_label_if_changed(
                    self.lbl_goertzel,
                    f"{self._fft_format_mag(gz['dbv']).strip()} · {gz['phase_deg']:+6.1f}°",
                )
            elif not self.chk_goertzel.isChecked():
                self._set_label_if_changed(self.lbl_goertzel, "-- · -- °")
            dev_name = primary_fft.get("device_name", "FFT")
            n_pts = primary_fft.get("n_fft", len(primary_fft.get("freqs", ())))
            self._set_label_if_changed(
                self.lbl_gpu_badge,
                f"⚡ {dev_name} · {n_pts} pts · {fft_ms:5.2f} ms",
            )

        fps = data.get("fps", 0.0)
        self._set_label_if_changed(self.lbl_fps, f"UI/Acq: {fps:5.1f} FPS  |  FFT 耗时: {fft_ms:5.2f} ms")

    @staticmethod
    def _set_label_if_changed(label: W.QLabel, text: str):
        if label.text() != text:
            label.setText(text)

    def _load_settings(self):
        s = QtCore.QSettings("WaveDesk", "WaveDesk")
        if (geom := s.value("window/geometry")) is not None:
            self.restoreGeometry(geom)
        if (bs := s.value("window/body_split")) is not None:
            self.body_split.restoreState(bs)
        if (ps := s.value("window/plot_split")) is not None:
            self.plot_splitter.restoreState(ps)

        # Collapsible cards
        self.card_tb.set_expanded(s.value("cards/timebase", True, type=bool), emit=False)
        self.card_trig.set_expanded(s.value("cards/trigger", True, type=bool), emit=False)
        self.card_ch.set_expanded(s.value("cards/channels", True, type=bool), emit=False)
        self.card_fft.set_expanded(s.value("cards/fft", True, type=bool), emit=False)
        self.card_awg.set_expanded(s.value("cards/awg", False, type=bool), emit=False)
        self.card_misc.set_expanded(s.value("cards/misc", False, type=bool), emit=False)

        # Top toggles (respect CLI --demo if explicitly passed)
        with QtCore.QSignalBlocker(self.chk_dpo):
            dpo_on = s.value("scope/dpo", False, type=bool)
            self.chk_dpo.setChecked(dpo_on)
            self._on_dpo_toggled(dpo_on)

        # Timebase & Trigger
        widgets_to_block = [
            self.cmb_timebase,
            self.spn_trig_pos,
            self.cmb_rate,
            self.cmb_buf,
            self.cmb_acq_mode,
            self.spn_rec_len,
            self.chk_hw_counter,
            self.cmb_trig_coupling,
            self.cmb_trig_mode,
            self.cmb_trig_src,
            self.cmb_trig_slope,
            self.spn_trig_lvl,
            self.cmb_trig_type,
            self.spn_trig_hyst,
            self.cmb_trig_len_cond,
            self.spn_trig_len_us,
            self.spn_trig_holdoff,
            self.cmb_trig_hw_flt,
            self.chk_fft_en,
            self.fft_span_input,
            self.cmb_fft_win,
            self.spn_fft_avg,
            self.cmb_fft_size,
            self.cmb_fft_unit,
            self.cmb_fft_trace,
            self.chk_fft_zoom,
            self.chk_goertzel,
            self.goertzel_freq,
        ]
        blockers = [QtCore.QSignalBlocker(w) for w in widgets_to_block]
        try:
            self.cmb_timebase.setCurrentIndex(int(np.clip(s.value("scope/tb_idx", 6, type=int), 0, self.cmb_timebase.count() - 1)))
            self.spn_trig_pos.setValue(s.value("scope/trig_pos_us", 0.0, type=float))
            self.trig_pos_line.setValue(self.spn_trig_pos.value() * 1e-6)
            self.cmb_rate.setCurrentIndex(int(np.clip(s.value("scope/rate_idx", 0, type=int), 0, self.cmb_rate.count() - 1)))
            self.cmb_buf.setCurrentIndex(int(np.clip(s.value("scope/buf_idx", 0, type=int), 0, self.cmb_buf.count() - 1)))
            self.cmb_acq_mode.setCurrentText(s.value("scope/acq_mode", "Single (标准单帧)", type=str))
            self.spn_rec_len.setValue(s.value("scope/rec_len", 0.05, type=float))
            self.spn_rec_len.setVisible(self.cmb_acq_mode.currentText().startswith("Record"))
            self.chk_hw_counter.setChecked(s.value("scope/hw_counter", False, type=bool))
            self.cmb_trig_coupling.setCurrentIndex(int(np.clip(s.value("scope/trig_cpl_idx", 0, type=int), 0, self.cmb_trig_coupling.count() - 1)))
            self.cmb_trig_mode.setCurrentText(s.value("scope/trig_mode", "Auto", type=str))
            self.cmb_trig_src.setCurrentText(s.value("scope/trig_src", "CH1", type=str))
            self.cmb_trig_slope.setCurrentText(s.value("scope/trig_slope", "Rising", type=str))
            lvl = s.value("scope/trig_lvl", 0.0, type=float)
            self.spn_trig_lvl.setValue(lvl)
            self.trig_line.setValue(lvl)
            self.cmb_trig_type.setCurrentText(s.value("scope/trig_type", "Edge (边沿)", type=str))
            self.spn_trig_hyst.setValue(s.value("scope/trig_hyst", 0.025, type=float))
            self.cmb_trig_len_cond.setCurrentText(s.value("scope/trig_len_cond", "More (>)", type=str))
            self.spn_trig_len_us.setValue(s.value("scope/trig_len_us", 1.0, type=float))
            need_len = self.cmb_trig_type.currentText().startswith(("Pulse", "Transition"))
            self.lbl_trig_len.setVisible(need_len)
            self.cmb_trig_len_cond.setVisible(need_len)
            self.spn_trig_len_us.setVisible(need_len)
            self.spn_trig_holdoff.setValue(s.value("scope/trig_holdoff", 0.0, type=float))
            self.cmb_trig_hw_flt.setCurrentText(s.value("scope/trig_hw_flt", "Average", type=str))

            # Scope active channels & per-channel settings
            open_ch = s.value("scope/open_channels", ["0"])
            self._active_scope_channels = {int(x) for x in open_ch if str(x).isdigit() and int(x) in (0, 1)}
            for idx, (cmb_range, chk_ac, cmb_flt, spn_offset, cmb_atten, cmb_hw_flt) in enumerate(self.ch_controls):
                with (
                    QtCore.QSignalBlocker(cmb_range),
                    QtCore.QSignalBlocker(chk_ac),
                    QtCore.QSignalBlocker(cmb_flt),
                    QtCore.QSignalBlocker(spn_offset),
                    QtCore.QSignalBlocker(cmb_atten),
                    QtCore.QSignalBlocker(cmb_hw_flt),
                ):
                    default_span = 2.0 if idx == 0 else 5.0
                    saved_span = s.value(f"ch{idx}/range_span", default_span, type=float)
                    best_r_idx = min(
                        range(cmb_range.count()),
                        key=lambda i: abs(self._unpack_voltage_range(cmb_range.itemData(i))[1] - saved_span),
                    )
                    cmb_range.setCurrentIndex(best_r_idx)
                    chk_ac.setChecked(s.value(f"ch{idx}/ac", False, type=bool))
                    cmb_flt.setCurrentIndex(int(np.clip(s.value(f"ch{idx}/flt_idx", 0 if idx == 0 else 1, type=int), 0, cmb_flt.count() - 1)))
                    spn_offset.setValue(s.value(f"ch{idx}/offset_v", 0.0, type=float))
                    cmb_atten.setCurrentIndex(int(np.clip(s.value(f"ch{idx}/atten_idx", 0, type=int), 0, cmb_atten.count() - 1)))
                    cmb_hw_flt.setCurrentText(s.value(f"ch{idx}/hw_flt", "Auto (自动抗混叠)", type=str))
                self.ch_subcards[idx].setVisible(idx in self._active_scope_channels)
            self._sync_scope_channel_buttons()

            # FFT settings
            self.chk_fft_en.setChecked(s.value("fft/enabled", True, type=bool))
            self.fft_span_input.setValue(s.value("fft/span_hz", 5_000_000.0, type=float))
            self.cmb_fft_win.setCurrentIndex(int(np.clip(s.value("fft/win_idx", 0, type=int), 0, self.cmb_fft_win.count() - 1)))
            self.spn_fft_avg.setValue(int(np.clip(s.value("fft/avg", 4, type=int), 1, 64)))
            self.cmb_fft_size.setCurrentIndex(int(np.clip(s.value("fft/size_idx", 0, type=int), 0, self.cmb_fft_size.count() - 1)))
            self.cmb_fft_unit.setCurrentIndex(int(np.clip(s.value("fft/unit_idx", 0, type=int), 0, self.cmb_fft_unit.count() - 1)))
            self.cmb_fft_trace.setCurrentIndex(int(np.clip(s.value("fft/trace_idx", 0, type=int), 0, self.cmb_fft_trace.count() - 1)))
            self.chk_fft_zoom.setChecked(s.value("fft/zoom", False, type=bool))
            self.chk_goertzel.setChecked(s.value("fft/goertzel_en", False, type=bool))
            self.goertzel_freq.setValue(s.value("fft/goertzel_hz", 200_000.0, type=float))
            self._on_fft_unit_changed()

            # AWG settings: restore parameters & open cards, but ALWAYS keep output disabled on startup!
            open_awg = s.value("awg/open_channels", [])
            self._active_awg_channels = {int(x) for x in open_awg if str(x).isdigit() and int(x) in (0, 1)}
            for idx, (chk_en, cmb_fn, spn_freq, spn_amp, spn_off, spn_sym) in enumerate(self.awg_widgets):
                with (
                    QtCore.QSignalBlocker(chk_en),
                    QtCore.QSignalBlocker(cmb_fn),
                    QtCore.QSignalBlocker(spn_freq),
                    QtCore.QSignalBlocker(spn_amp),
                    QtCore.QSignalBlocker(spn_off),
                    QtCore.QSignalBlocker(spn_sym),
                ):
                    chk_en.setChecked(False)  # Waveform generator defaults to OFF on startup
                    cmb_fn.setCurrentText(s.value(f"awg{idx}/fn", "Sine", type=str))
                    spn_freq.setValue(s.value(f"awg{idx}/freq", 1_000_000.0 if idx == 0 else 1_100_000.0, type=float))
                    spn_amp.setValue(s.value(f"awg{idx}/amp", 1.0, type=float))
                    spn_off.setValue(s.value(f"awg{idx}/off", 0.0, type=float))
                    spn_sym.setValue(s.value(f"awg{idx}/sym", 50.0, type=float))
                    adv = self.awg_adv_widgets[idx]
                    adv_blockers = [
                        QtCore.QSignalBlocker(adv[k])
                        for k in (
                            "phase",
                            "chk_am",
                            "cmb_am_fn",
                            "freq_am",
                            "spn_am_depth",
                            "chk_fm",
                            "cmb_fm_fn",
                            "freq_fm",
                            "spn_fm_depth",
                            "spn_run_ms",
                            "spn_wait_ms",
                            "spn_repeat",
                            "chk_rep_trig",
                            "cmb_awg_trig",
                            "cmb_awg_slope",
                            "cmb_idle",
                            "spn_limit_v",
                            "chk_sync_w1",
                            "cmb_ssb",
                            "chk_sweep",
                            "freq_sweep_start",
                            "freq_sweep_stop",
                            "spn_sweep_ms",
                            "spn_sweep_wait_ms",
                            "spn_sweep_rep",
                            "cmb_am_wav",
                            "cmb_fm_wav",
                        )
                    ]
                    try:
                        adv["phase"].setValue(s.value(f"awg{idx}/phase", 0.0, type=float))
                        adv["chk_am"].setChecked(s.value(f"awg{idx}/am_en", False, type=bool))
                        adv["cmb_am_fn"].setCurrentText(s.value(f"awg{idx}/am_fn", "Sine", type=str))
                        adv["freq_am"].setValue(s.value(f"awg{idx}/am_freq", 1_000.0, type=float))
                        adv["spn_am_depth"].setValue(s.value(f"awg{idx}/am_depth", 50.0, type=float))
                        adv["chk_fm"].setChecked(s.value(f"awg{idx}/fm_en", False, type=bool))
                        adv["cmb_fm_fn"].setCurrentText(s.value(f"awg{idx}/fm_fn", "Sine", type=str))
                        adv["freq_fm"].setValue(s.value(f"awg{idx}/fm_freq", 1_000.0, type=float))
                        adv["spn_fm_depth"].setValue(s.value(f"awg{idx}/fm_depth", 10.0, type=float))
                        adv["spn_run_ms"].setValue(s.value(f"awg{idx}/run_ms", 0.0, type=float))
                        adv["spn_wait_ms"].setValue(s.value(f"awg{idx}/wait_ms", 0.0, type=float))
                        adv["spn_repeat"].setValue(s.value(f"awg{idx}/repeat", 0, type=int))
                        adv["chk_rep_trig"].setChecked(s.value(f"awg{idx}/rep_trig", False, type=bool))
                        adv["cmb_awg_trig"].setCurrentText(s.value(f"awg{idx}/trig_src", "None (连续自由运行)", type=str))
                        adv["cmb_awg_slope"].setCurrentText(s.value(f"awg{idx}/trig_slope", "Rising", type=str))
                        adv["cmb_idle"].setCurrentText(s.value(f"awg{idx}/idle", "Offset (偏置电平)", type=str))
                        adv["spn_limit_v"].setValue(s.value(f"awg{idx}/limit_v", 0.0, type=float))
                        adv["chk_sync_w1"].setChecked(s.value(f"awg{idx}/sync_w1", False, type=bool))
                        adv["cmb_ssb"].setCurrentText(s.value(f"awg{idx}/ssb_mode", "Off (关闭)", type=str))
                        adv["chk_sweep"].setChecked(s.value(f"awg{idx}/sweep_en", False, type=bool))
                        adv["freq_sweep_start"].setValue(s.value(f"awg{idx}/sweep_start", 100_000.0, type=float))
                        adv["freq_sweep_stop"].setValue(s.value(f"awg{idx}/sweep_stop", 1_000_000.0, type=float))
                        adv["spn_sweep_ms"].setValue(s.value(f"awg{idx}/sweep_ms", 10.0, type=float))
                        adv["spn_sweep_wait_ms"].setValue(s.value(f"awg{idx}/sweep_wait_ms", 0.0, type=float))
                        adv["spn_sweep_rep"].setValue(s.value(f"awg{idx}/sweep_rep", 0, type=int))
                        adv_open = s.value(f"awg{idx}/adv_open", False, type=bool)
                        adv["btn_adv"].setChecked(adv_open)
                        adv["adv_box"].setVisible(adv_open)
                        adv["btn_adv"].setText(("▾ " if adv_open else "▸ ") + "调制 / 突发 / 触发 / 同步 (AM/FM/Burst)")
                        for kind in ("am", "fm"):
                            wav_name = s.value(f"awg{idx}/{kind}_wav", "", type=str)
                            if wav_name:
                                widx = adv[f"cmb_{kind}_wav"].findText(wav_name)
                                if widx >= 0:
                                    adv[f"cmb_{kind}_wav"].setCurrentIndex(widx)
                        adv["btn_load_custom"].setVisible(cmb_fn.currentText() == "Custom (自定义)")
                        adv["btn_edit_poly"].setVisible(cmb_fn.currentText() == "Polynomial (多正弦)")
                        poly_raw = s.value(f"awg{idx}/poly_terms", "", type=str)
                        if poly_raw:
                            try:
                                parsed = json.loads(poly_raw)
                                if isinstance(parsed, list):
                                    self.awg_poly_terms[idx] = [
                                        (float(t[0]), float(t[1]), float(t[2]))
                                        for t in parsed
                                        if isinstance(t, (list, tuple)) and len(t) == 3
                                    ]
                            except Exception:
                                pass
                        for kind in ("am", "fm"):
                            is_wav = adv[f"cmb_{kind}_fn"].currentText() == "WAV (音频文件)"
                            adv[f"freq_{kind}"].setVisible(not is_wav)
                            adv[f"cmb_{kind}_wav"].setVisible(is_wav)
                    finally:
                        del adv_blockers
                    for kind in ("am", "fm"):
                        store = self.awg_am_wav if kind == "am" else self.awg_fm_wav
                        if adv[f"cmb_{kind}_fn"].currentText() == "WAV (音频文件)" and store[idx] is None:
                            self._on_mod_wav_selected(idx, kind)
                self.awg_subcards[idx].setVisible(idx in self._active_awg_channels)
            self._sync_awg_channel_buttons()

            wf_w = max(360, s.value("waterfall/w", 620, type=int))
            wf_h = max(170, s.value("waterfall/h", 250, type=int))
            wf_x = s.value("waterfall/x", 28, type=int)
            wf_y = s.value("waterfall/y", 86, type=int)
            self.waterfall_overlay.resize(wf_w, wf_h)
            self.waterfall_overlay.move(wf_x, wf_y)
            wf_open = s.value("waterfall/open", False, type=bool)
            self._toggle_waterfall(wf_open)

            vx0 = s.value("fft/view_x0", None)
            vx1 = s.value("fft/view_x1", None)
            vy0 = s.value("fft/view_y0", None)
            vy1 = s.value("fft/view_y1", None)
            if vx0 is not None and vx1 is not None:
                fx0, fx1 = float(vx0), float(vx1)
                if fx1 > fx0:
                    self._saved_fft_xrange = (fx0, fx1)
                    self.fft_plot.setXRange(fx0, fx1, padding=0.0)
            if vy0 is not None and vy1 is not None:
                fy0, fy1 = float(vy0), float(vy1)
                if fy1 > fy0:
                    self.fft_plot.setYRange(fy0, fy1, padding=0.0)
        finally:
            del blockers

    def _save_settings(self):
        s = QtCore.QSettings("WaveDesk", "WaveDesk")
        s.setValue("window/geometry", self.saveGeometry())
        s.setValue("window/body_split", self.body_split.saveState())
        s.setValue("window/plot_split", self.plot_splitter.saveState())

        s.setValue("cards/timebase", self.card_tb.is_expanded())
        s.setValue("cards/trigger", self.card_trig.is_expanded())
        s.setValue("cards/channels", self.card_ch.is_expanded())
        s.setValue("cards/fft", self.card_fft.is_expanded())
        s.setValue("cards/awg", self.card_awg.is_expanded())
        s.setValue("cards/misc", self.card_misc.is_expanded())

        s.setValue("scope/dpo", self.chk_dpo.isChecked())
        s.setValue("scope/tb_idx", self.cmb_timebase.currentIndex())
        s.setValue("scope/trig_pos_us", self.spn_trig_pos.value())
        s.setValue("scope/rate_idx", self.cmb_rate.currentIndex())
        s.setValue("scope/buf_idx", self.cmb_buf.currentIndex())
        s.setValue("scope/acq_mode", self.cmb_acq_mode.currentText())
        s.setValue("scope/rec_len", self.spn_rec_len.value())
        s.setValue("scope/hw_counter", self.chk_hw_counter.isChecked())
        s.setValue("scope/trig_cpl_idx", self.cmb_trig_coupling.currentIndex())
        s.setValue("scope/trig_mode", self.cmb_trig_mode.currentText())
        s.setValue("scope/trig_src", self.cmb_trig_src.currentText())
        s.setValue("scope/trig_slope", self.cmb_trig_slope.currentText())
        s.setValue("scope/trig_lvl", self.spn_trig_lvl.value())
        s.setValue("scope/trig_type", self.cmb_trig_type.currentText())
        s.setValue("scope/trig_hyst", self.spn_trig_hyst.value())
        s.setValue("scope/trig_len_cond", self.cmb_trig_len_cond.currentText())
        s.setValue("scope/trig_len_us", self.spn_trig_len_us.value())
        s.setValue("scope/trig_holdoff", self.spn_trig_holdoff.value())
        s.setValue("scope/trig_hw_flt", self.cmb_trig_hw_flt.currentText())

        s.setValue("scope/open_channels", [str(i) for i in sorted(self._active_scope_channels)])
        for idx, (cmb_range, chk_ac, cmb_flt, spn_offset, cmb_atten, cmb_hw_flt) in enumerate(self.ch_controls):
            _, disp_span = self._unpack_voltage_range(cmb_range.currentData())
            s.setValue(f"ch{idx}/range_span", disp_span)
            s.setValue(f"ch{idx}/ac", chk_ac.isChecked())
            s.setValue(f"ch{idx}/flt_idx", cmb_flt.currentIndex())
            s.setValue(f"ch{idx}/offset_v", spn_offset.value())
            s.setValue(f"ch{idx}/atten_idx", cmb_atten.currentIndex())
            s.setValue(f"ch{idx}/hw_flt", cmb_hw_flt.currentText())

        s.setValue("fft/enabled", self.chk_fft_en.isChecked())
        s.setValue("fft/span_hz", self.fft_span_input.value())
        s.setValue("fft/win_idx", self.cmb_fft_win.currentIndex())
        s.setValue("fft/avg", self.spn_fft_avg.value())
        s.setValue("fft/size_idx", self.cmb_fft_size.currentIndex())
        s.setValue("fft/unit_idx", self.cmb_fft_unit.currentIndex())
        s.setValue("fft/trace_idx", self.cmb_fft_trace.currentIndex())
        s.setValue("fft/zoom", self.chk_fft_zoom.isChecked())
        s.setValue("fft/goertzel_en", self.chk_goertzel.isChecked())
        s.setValue("fft/goertzel_hz", self.goertzel_freq.value())
        fft_xr, fft_yr = self.fft_plot.getViewBox().viewRange()
        s.setValue("fft/view_x0", float(fft_xr[0]))
        s.setValue("fft/view_x1", float(fft_xr[1]))
        s.setValue("fft/view_y0", float(fft_yr[0]))
        s.setValue("fft/view_y1", float(fft_yr[1]))

        s.setValue("awg/open_channels", [str(i) for i in sorted(self._active_awg_channels)])
        for idx, (_, cmb_fn, spn_freq, spn_amp, spn_off, spn_sym) in enumerate(self.awg_widgets):
            s.setValue(f"awg{idx}/fn", cmb_fn.currentText())
            s.setValue(f"awg{idx}/freq", spn_freq.value())
            s.setValue(f"awg{idx}/amp", spn_amp.value())
            s.setValue(f"awg{idx}/off", spn_off.value())
            s.setValue(f"awg{idx}/sym", spn_sym.value())
            adv = self.awg_adv_widgets[idx]
            s.setValue(f"awg{idx}/phase", adv["phase"].value())
            s.setValue(f"awg{idx}/am_en", adv["chk_am"].isChecked())
            s.setValue(f"awg{idx}/am_fn", adv["cmb_am_fn"].currentText())
            s.setValue(f"awg{idx}/am_freq", adv["freq_am"].value())
            s.setValue(f"awg{idx}/am_depth", adv["spn_am_depth"].value())
            s.setValue(f"awg{idx}/fm_en", adv["chk_fm"].isChecked())
            s.setValue(f"awg{idx}/fm_fn", adv["cmb_fm_fn"].currentText())
            s.setValue(f"awg{idx}/fm_freq", adv["freq_fm"].value())
            s.setValue(f"awg{idx}/fm_depth", adv["spn_fm_depth"].value())
            s.setValue(f"awg{idx}/run_ms", adv["spn_run_ms"].value())
            s.setValue(f"awg{idx}/wait_ms", adv["spn_wait_ms"].value())
            s.setValue(f"awg{idx}/repeat", adv["spn_repeat"].value())
            s.setValue(f"awg{idx}/rep_trig", adv["chk_rep_trig"].isChecked())
            s.setValue(f"awg{idx}/trig_src", adv["cmb_awg_trig"].currentText())
            s.setValue(f"awg{idx}/trig_slope", adv["cmb_awg_slope"].currentText())
            s.setValue(f"awg{idx}/idle", adv["cmb_idle"].currentText())
            s.setValue(f"awg{idx}/limit_v", adv["spn_limit_v"].value())
            s.setValue(f"awg{idx}/sync_w1", adv["chk_sync_w1"].isChecked())
            s.setValue(f"awg{idx}/ssb_mode", adv["cmb_ssb"].currentText())
            s.setValue(f"awg{idx}/sweep_en", adv["chk_sweep"].isChecked())
            s.setValue(f"awg{idx}/sweep_start", adv["freq_sweep_start"].value())
            s.setValue(f"awg{idx}/sweep_stop", adv["freq_sweep_stop"].value())
            s.setValue(f"awg{idx}/sweep_ms", adv["spn_sweep_ms"].value())
            s.setValue(f"awg{idx}/sweep_wait_ms", adv["spn_sweep_wait_ms"].value())
            s.setValue(f"awg{idx}/sweep_rep", adv["spn_sweep_rep"].value())
            s.setValue(f"awg{idx}/adv_open", adv["btn_adv"].isChecked())
            s.setValue(f"awg{idx}/am_wav", adv["cmb_am_wav"].currentText())
            s.setValue(f"awg{idx}/fm_wav", adv["cmb_fm_wav"].currentText())
            if self.awg_poly_terms[idx] is not None:
                s.setValue(f"awg{idx}/poly_terms", json.dumps(self.awg_poly_terms[idx]))
        s.setValue("waterfall/open", self.waterfall_overlay.isVisible())
        s.setValue("waterfall/x", self.waterfall_overlay.x())
        s.setValue("waterfall/y", self.waterfall_overlay.y())
        s.setValue("waterfall/w", self.waterfall_overlay.width())
        s.setValue("waterfall/h", self.waterfall_overlay.height())
        s.sync()

    def closeEvent(self, event: QtGui.QCloseEvent):
        self._save_settings()
        if hasattr(self, "thread") and self.thread.isRunning():
            QtCore.QMetaObject.invokeMethod(
                self.worker, "disconnect_device", QtCore.Qt.ConnectionType.BlockingQueuedConnection
            )
            self.thread.quit()
            self.thread.wait(1500)
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="WaveDesk: GPU-Accelerated AD3 Workbench")
    parser.add_argument("--demo", action="store_true", help="Run with built-in signal simulator")
    parser.add_argument("--renderer", choices=("opengl", "software"), default="opengl")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test and exit automatically")
    args = parser.parse_args()

    app = W.QApplication(sys.argv)
    app.setApplicationName("WaveDesk")
    app.setWindowIcon(_create_app_icon())
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#0e131a"))
    pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#dbe4ee"))
    pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#121a24"))
    pal.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#151d28"))
    pal.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#eef4fa"))
    pal.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#1d2837"))
    pal.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#dbe4ee"))
    pal.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#1f4c7a"))
    pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(pal)
    app.setStyleSheet(DARK_THEME_QSS)

    win = MainWindow(args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
