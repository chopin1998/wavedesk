"""WaveDesk: Next-Gen GPU-Accelerated AD3 Oscilloscope, Spectrum Analyzer & AWG Workbench."""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets as W
from shiboken6 import isValid

from .driver import AwgChannelConfig, ChannelConfig, ScopeConfig, TriggerConfig, WAVEFORM_FUNCTIONS
from .gpu_fft import WINDOWS
from .gpu_view import GpuWaveformCurve
from .measurements import format_si
from .worker import InstrumentWorker


DARK_THEME_QSS = """
QMainWindow, QWidget#root {
    background-color: #0e131a;
    color: #dbe4ee;
}
QWidget {
    color: #dbe4ee;
    font-size: 10pt;
}
QFrame#card {
    background-color: #151d28;
    border: 1px solid #243244;
    border-radius: 8px;
}
QFrame#metricBar {
    background-color: #131a24;
    border: 1px solid #223042;
    border-radius: 7px;
}
QLabel#brandTitle {
    font-size: 15pt;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: 0.5px;
}
QLabel#secTitle {
    font-size: 10.5pt;
    font-weight: 700;
    color: #8ab4f8;
}
QLabel#ch1Badge {
    color: #00f2fe;
    font-weight: 700;
    font-size: 10.5pt;
}
QLabel#ch2Badge {
    color: #ffb800;
    font-weight: 700;
    font-size: 10.5pt;
}
QLabel#muted {
    color: #8b9bb0;
    font-size: 9.2pt;
}
QLabel#metricValue {
    font-family: monospace;
    font-size: 11pt;
    font-weight: 600;
    color: #eef4fa;
}
QLabel#gpuBadge {
    background-color: #132f2c;
    color: #38ef7d;
    border: 1px solid #235c53;
    border-radius: 5px;
    padding: 4px 8px;
    font-weight: 600;
    font-size: 9.2pt;
}
QLabel#devBadge {
    background-color: #182638;
    color: #79b8ff;
    border: 1px solid #294466;
    border-radius: 5px;
    padding: 4px 8px;
    font-weight: 600;
    font-size: 9.2pt;
}
QPushButton {
    background-color: #1d2837;
    border: 1px solid #30425a;
    border-radius: 5px;
    padding: 5px 11px;
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
    padding: 6px 12px;
    color: #8b9bb0;
}
QTabBar::tab:selected {
    background-color: #151d28;
    color: #00f2fe;
    font-weight: 700;
}
QScrollArea {
    border: none;
    background: transparent;
}
QStatusBar {
    background-color: #0b0f15;
    color: #8b9bb0;
    border-top: 1px solid #1e2938;
}
"""


SAMPLE_RATES = [
    ("125 MS/s", 125_000_000.0),
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
    ("32768 pts (AD3 硬件采集)", 32768),
    ("16384 pts", 16384),
    ("8192 pts", 8192),
    ("4096 pts", 4096),
    ("2048 pts", 2048),
]

FFT_SIZES = [
    ("跟随采样深度 (1:1)", 0),
    ("65,536 pts (64K 超清频谱)", 65536),
    ("131,072 pts (128K 超高分辨)", 131072),
    ("262,144 pts (256K 极限分辨)", 262144),
    ("524,288 pts (512K GPU 专属)", 524288),
]

VOLTAGE_RANGES = [
    ("±0.5 V (1 Vpp)", 1.0),
    ("±1.0 V (2 Vpp)", 2.0),
    ("±2.5 V (5 Vpp)", 5.0),
    ("±5.0 V (10 Vpp)", 10.0),
    ("±10 V (20 Vpp)", 20.0),
    ("±25 V (50 Vpp)", 50.0),
]


class FastPlotWidget(pg.PlotWidget):
    """OpenGL-accelerated oscilloscope/spectrum plot with dark oscilloscope grid."""

    def __init__(self, title: str, x_label: str, x_units: str, y_label: str, y_units: str, use_opengl: bool = True):
        super().__init__()
        self.setBackground("#0b0f15")
        self.useOpenGL(use_opengl)
        self.showGrid(x=True, y=True, alpha=0.22)
        self.setTitle(title, color="#9fb3c8", size="10pt")
        self.setLabel("bottom", x_label, units=x_units, color="#8b9bb0")
        self.setLabel("left", y_label, units=y_units, color="#8b9bb0")
        for side in ("left", "bottom"):
            ax = self.getAxis(side)
            ax.setPen(pg.mkPen("#29384b", width=1))
            ax.setTextPen(pg.mkPen("#8b9bb0"))
        self.getViewBox().setBorder(pg.mkPen("#243244", width=1))


class MainWindow(W.QMainWindow):
    req_connect = QtCore.Signal()
    req_disconnect = QtCore.Signal()
    req_demo = QtCore.Signal(bool)
    req_start = QtCore.Signal()
    req_stop = QtCore.Signal()
    req_single = QtCore.Signal()
    req_scope_cfg = QtCore.Signal(object)
    req_awg_cfg = QtCore.Signal(int, object)
    req_fft_cfg = QtCore.Signal(dict)

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.use_opengl = getattr(args, "renderer", "opengl") == "opengl"
        self.setWindowTitle("WaveDesk · AD3 GPU 超算示波器与频谱分析工作台")
        self.resize(1640, 980)
        self.setMinimumSize(1240, 760)

        self.scope_cfg = ScopeConfig(
            sample_rate_hz=2_000_000.0,
            buffer_size=16384,
            channels=[ChannelConfig(True, 5.0), ChannelConfig(True, 5.0)],
            trigger=TriggerConfig(mode="Auto", source="CH1", slope="Rising", level_v=0.0),
        )
        self.awg_cfgs = [
            AwgChannelConfig(enabled=True, function="Sine", frequency_hz=1_000.0, amplitude_v=1.2),
            AwgChannelConfig(enabled=False, function="Square", frequency_hz=2_000.0, amplitude_v=0.8),
        ]

        self._build_ui()
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
        top_layout.setContentsMargins(12, 7, 12, 7)
        top_layout.setSpacing(10)

        brand = W.QLabel("WaveDesk")
        brand.setObjectName("brandTitle")
        top_layout.addWidget(brand)

        sub = W.QLabel("AD3 GPU Super-Workbench")
        sub.setObjectName("muted")
        top_layout.addWidget(sub)
        top_layout.addSpacing(12)

        self.btn_run = W.QPushButton("▶ RUN 连续采集")
        self.btn_run.setObjectName("runBtn")
        self.btn_run.clicked.connect(self.req_start.emit)
        top_layout.addWidget(self.btn_run)

        self.btn_single = W.QPushButton("⏭ SINGLE 单次触发")
        self.btn_single.clicked.connect(self.req_single.emit)
        top_layout.addWidget(self.btn_single)

        self.btn_stop = W.QPushButton("⏹ STOP 暂停")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.clicked.connect(self.req_stop.emit)
        top_layout.addWidget(self.btn_stop)

        self.btn_autoscale = W.QPushButton("⤢ 视图自适应")
        self.btn_autoscale.clicked.connect(self.autoscale_views)
        top_layout.addWidget(self.btn_autoscale)

        self.chk_dpo = W.QCheckBox("GPU DPO 数字荧光余晖")
        self.chk_dpo.setChecked(True)
        self.chk_dpo.toggled.connect(self._on_dpo_toggled)
        top_layout.addWidget(self.chk_dpo)

        self.chk_demo = W.QCheckBox("模拟信号源 (Demo)")
        self.chk_demo.setChecked(bool(self.args.demo))
        self.chk_demo.toggled.connect(self.req_demo.emit)
        top_layout.addWidget(self.chk_demo)

        top_layout.addStretch(1)

        self.lbl_gpu_badge = W.QLabel("GPU FFT: 检测中…")
        self.lbl_gpu_badge.setObjectName("gpuBadge")
        top_layout.addWidget(self.lbl_gpu_badge)

        self.lbl_dev_badge = W.QLabel("设备: 未连接")
        self.lbl_dev_badge.setObjectName("devBadge")
        top_layout.addWidget(self.lbl_dev_badge)

        self.btn_reconnect = W.QPushButton("重连硬件")
        self.btn_reconnect.clicked.connect(self.req_connect.emit)
        top_layout.addWidget(self.btn_reconnect)

        main_layout.addWidget(top_bar)

        # Body Splitter: Left plots + Right control rack
        body_split = W.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(body_split, 1)

        # Left side: Dual Vertical Splitter (Scope on top, GPU FFT Spectrum on bottom) + Measurement Dock
        left_container = W.QWidget()
        left_layout = W.QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.plot_splitter = W.QSplitter(QtCore.Qt.Orientation.Vertical)

        # 1. Time-Domain Oscilloscope Plot
        self.scope_plot = FastPlotWidget(
            "Time Domain · 实时双通道示波器 (GPU SDF Anti-Aliasing + DPO)",
            "Time",
            "s",
            "Voltage",
            "V",
            use_opengl=self.use_opengl,
        )
        self.scope_plot.setYRange(-2.5, 2.5)
        self.scope_curve_ch1 = GpuWaveformCurve(color="#00f2fe", width=2.2, dpo_enabled=True)
        self.scope_curve_ch2 = GpuWaveformCurve(color="#ffb800", width=2.2, dpo_enabled=True)
        self.scope_plot.addItem(self.scope_curve_ch1)
        self.scope_plot.addItem(self.scope_curve_ch2)

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
        self.plot_splitter.addWidget(self.scope_plot)

        # 2. Frequency-Domain GPU FFT Spectrum Plot
        self.fft_plot = FastPlotWidget(
            "Frequency Domain · GPU 实时频谱分析仪 (Cooley-Tukey Radix-2 OpenCL/CUDA)",
            "Frequency",
            "Hz",
            "Magnitude",
            "dBV",
            use_opengl=self.use_opengl,
        )
        self.fft_plot.setYRange(-115.0, 15.0)
        self.fft_curve_ch1 = GpuWaveformCurve(color="#00f2fe", width=2.0, dpo_enabled=True)
        self.fft_curve_ch2 = GpuWaveformCurve(color="#ffb800", width=2.0, dpo_enabled=True)
        self.fft_plot.addItem(self.fft_curve_ch1)
        self.fft_plot.addItem(self.fft_curve_ch2)

        # Peak markers on FFT
        self.fft_peaks_scatter = pg.ScatterPlotItem(size=8, brush=pg.mkBrush("#ff4e79"), pen=pg.mkPen("#ffffff", width=1))
        self.fft_peaks_scatter.setZValue(50)
        self.fft_plot.addItem(self.fft_peaks_scatter)

        self.plot_splitter.addWidget(self.fft_plot)
        self.plot_splitter.setSizes([500, 380])
        left_layout.addWidget(self.plot_splitter, 1)

        # Bottom Live Telemetry Cards (CH1, CH2, Spectral Metrics)
        metric_bar = W.QFrame()
        metric_bar.setObjectName("metricBar")
        m_layout = W.QHBoxLayout(metric_bar)
        m_layout.setContentsMargins(12, 8, 12, 8)
        m_layout.setSpacing(18)

        self.lbl_meas_ch1 = W.QLabel("CH1: 等待采集…")
        self.lbl_meas_ch1.setObjectName("metricValue")
        self.lbl_meas_ch2 = W.QLabel("CH2: 等待采集…")
        self.lbl_meas_ch2.setObjectName("metricValue")
        self.lbl_meas_fft = W.QLabel("FFT: 等待数据…")
        self.lbl_meas_fft.setObjectName("metricValue")

        c1_badge = W.QLabel("● CH1")
        c1_badge.setObjectName("ch1Badge")
        m_layout.addWidget(c1_badge)
        m_layout.addWidget(self.lbl_meas_ch1, 1)

        c2_badge = W.QLabel("● CH2")
        c2_badge.setObjectName("ch2Badge")
        m_layout.addWidget(c2_badge)
        m_layout.addWidget(self.lbl_meas_ch2, 1)

        fft_badge = W.QLabel("◆ GPU 频谱指标")
        fft_badge.setObjectName("secTitle")
        m_layout.addWidget(fft_badge)
        m_layout.addWidget(self.lbl_meas_fft, 1)

        left_layout.addWidget(metric_bar)
        body_split.addWidget(left_container)

        # Right side: Instrument Control Rack
        right_scroll = W.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(390)
        rack = W.QWidget()
        rack_layout = W.QVBoxLayout(rack)
        rack_layout.setContentsMargins(4, 0, 4, 0)
        rack_layout.setSpacing(10)
        right_scroll.setWidget(rack)

        rack_layout.addWidget(self._build_timebase_trigger_card())
        rack_layout.addWidget(self._build_channels_card())
        rack_layout.addWidget(self._build_fft_card())
        rack_layout.addWidget(self._build_awg_card())
        rack_layout.addStretch(1)

        body_split.addWidget(right_scroll)
        body_split.setStretchFactor(0, 1)
        body_split.setStretchFactor(1, 0)

        # Status Bar
        self.status_bar = W.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_status = W.QLabel("就绪")
        self.lbl_fps = W.QLabel("FPS: -- | FFT: -- ms")
        self.status_bar.addWidget(self.lbl_status, 1)
        self.status_bar.addPermanentWidget(self.lbl_fps)

    def _build_timebase_trigger_card(self) -> W.QFrame:
        card = W.QFrame()
        card.setObjectName("card")
        box = W.QVBoxLayout(card)
        box.setContentsMargins(12, 10, 12, 10)
        box.setSpacing(8)

        title = W.QLabel("⏱ 时基与硬件触发 (Timebase & Trigger)")
        title.setObjectName("secTitle")
        box.addWidget(title)

        grid = W.QGridLayout()
        grid.setSpacing(6)

        grid.addWidget(W.QLabel("采样率 (Sample Rate):"), 0, 0)
        self.cmb_rate = W.QComboBox()
        for label, val in SAMPLE_RATES:
            self.cmb_rate.addItem(label, val)
        self.cmb_rate.setCurrentIndex(6)  # 2 MS/s default (8.192 ms span for 16384 pts)
        self.cmb_rate.currentIndexChanged.connect(self._emit_scope_config)
        grid.addWidget(self.cmb_rate, 0, 1)

        grid.addWidget(W.QLabel("采样深度 (Buffer):"), 1, 0)
        self.cmb_buf = W.QComboBox()
        for label, val in BUFFER_SIZES:
            self.cmb_buf.addItem(label, val)
        self.cmb_buf.setCurrentIndex(1)  # 16384 default
        self.cmb_buf.currentIndexChanged.connect(self._emit_scope_config)
        grid.addWidget(self.cmb_buf, 1, 1)

        grid.addWidget(W.QLabel("触发模式 / 源:"), 2, 0)
        trig_row = W.QHBoxLayout()
        self.cmb_trig_mode = W.QComboBox()
        self.cmb_trig_mode.addItems(["Auto", "Normal", "Single", "None"])
        self.cmb_trig_mode.currentTextChanged.connect(self._emit_scope_config)
        self.cmb_trig_src = W.QComboBox()
        self.cmb_trig_src.addItems(["CH1", "CH2", "W1", "W2"])
        self.cmb_trig_src.currentTextChanged.connect(self._emit_scope_config)
        trig_row.addWidget(self.cmb_trig_mode)
        trig_row.addWidget(self.cmb_trig_src)
        grid.addLayout(trig_row, 2, 1)

        grid.addWidget(W.QLabel("触发边沿 / 电平:"), 3, 0)
        edge_row = W.QHBoxLayout()
        self.cmb_trig_slope = W.QComboBox()
        self.cmb_trig_slope.addItems(["Rising", "Falling", "Either"])
        self.cmb_trig_slope.currentTextChanged.connect(self._emit_scope_config)
        self.spn_trig_lvl = W.QDoubleSpinBox()
        self.spn_trig_lvl.setRange(-25.0, 25.0)
        self.spn_trig_lvl.setSingleStep(0.1)
        self.spn_trig_lvl.setDecimals(2)
        self.spn_trig_lvl.setSuffix(" V")
        self.spn_trig_lvl.valueChanged.connect(self._on_trig_spin_changed)
        edge_row.addWidget(self.cmb_trig_slope)
        edge_row.addWidget(self.spn_trig_lvl)
        grid.addLayout(edge_row, 3, 1)

        box.addLayout(grid)
        return card

    def _build_channels_card(self) -> W.QFrame:
        card = W.QFrame()
        card.setObjectName("card")
        box = W.QVBoxLayout(card)
        box.setContentsMargins(12, 10, 12, 10)
        box.setSpacing(8)

        title = W.QLabel("📈 模拟输入通道 (Oscilloscope CH1 / CH2)")
        title.setObjectName("secTitle")
        box.addWidget(title)

        self.ch_controls = []
        for idx, (name, badge_id) in enumerate([("CH1 通道 1", "ch1Badge"), ("CH2 通道 2", "ch2Badge")]):
            row_box = W.QVBoxLayout()
            hdr = W.QHBoxLayout()
            chk = W.QCheckBox(name)
            chk.setChecked(True)
            chk.setStyleSheet("color: #00f2fe; font-weight: 700;" if idx == 0 else "color: #ffb800; font-weight: 700;")
            chk.toggled.connect(self._emit_scope_config)
            hdr.addWidget(chk)

            cmb_range = W.QComboBox()
            for lbl, val in VOLTAGE_RANGES:
                cmb_range.addItem(lbl, val)
            cmb_range.setCurrentIndex(2)  # 5 Vpp default
            cmb_range.currentIndexChanged.connect(self._emit_scope_config)
            hdr.addWidget(cmb_range, 1)

            chk_ac = W.QCheckBox("AC耦合")
            chk_ac.toggled.connect(self._emit_scope_config)
            hdr.addWidget(chk_ac)

            row_box.addLayout(hdr)
            box.addLayout(row_box)
            self.ch_controls.append((chk, cmb_range, chk_ac))

        return card

    def _build_fft_card(self) -> W.QFrame:
        card = W.QFrame()
        card.setObjectName("card")
        box = W.QVBoxLayout(card)
        box.setContentsMargins(12, 10, 12, 10)
        box.setSpacing(8)

        hdr = W.QHBoxLayout()
        title = W.QLabel("🚀 GPU 频谱加速引擎 (FFT Analyzer)")
        title.setObjectName("secTitle")
        hdr.addWidget(title, 1)
        self.chk_fft_en = W.QCheckBox("启用")
        self.chk_fft_en.setChecked(True)
        self.chk_fft_en.toggled.connect(self._emit_fft_config)
        hdr.addWidget(self.chk_fft_en)
        box.addLayout(hdr)

        grid = W.QGridLayout()
        grid.setSpacing(6)

        grid.addWidget(W.QLabel("计算后端 (Backend):"), 0, 0)
        self.cmb_fft_backend = W.QComboBox()
        self.cmb_fft_backend.addItem("OpenCL GPU (自动检测)", "opencl")
        self.cmb_fft_backend.addItem("CPU (SciPy PocketFFT)", "cpu")
        self.cmb_fft_backend.currentIndexChanged.connect(self._emit_fft_config)
        grid.addWidget(self.cmb_fft_backend, 0, 1)

        grid.addWidget(W.QLabel("窗函数 (Window):"), 1, 0)
        self.cmb_fft_win = W.QComboBox()
        for wname, (_, desc) in WINDOWS.items():
            self.cmb_fft_win.addItem(f"{wname} · {desc}", wname)
        self.cmb_fft_win.currentIndexChanged.connect(self._emit_fft_config)
        grid.addWidget(self.cmb_fft_win, 1, 1)

        grid.addWidget(W.QLabel("指数平滑平均 (Avg):"), 2, 0)
        self.spn_fft_avg = W.QSpinBox()
        self.spn_fft_avg.setRange(1, 64)
        self.spn_fft_avg.setValue(2)
        self.spn_fft_avg.setSuffix(" 帧")
        self.spn_fft_avg.valueChanged.connect(self._emit_fft_config)
        grid.addWidget(self.spn_fft_avg, 2, 1)

        grid.addWidget(W.QLabel("FFT 点数 (Resolution):"), 3, 0)
        self.cmb_fft_size = W.QComboBox()
        for lbl, val in FFT_SIZES:
            self.cmb_fft_size.addItem(lbl, val)
        self.cmb_fft_size.setCurrentIndex(0)  # 1:1 follow buffer size by default
        self.cmb_fft_size.currentIndexChanged.connect(self._emit_fft_config)
        grid.addWidget(self.cmb_fft_size, 3, 1)

        box.addLayout(grid)
        return card

    def _build_awg_card(self) -> W.QFrame:
        card = W.QFrame()
        card.setObjectName("card")
        box = W.QVBoxLayout(card)
        box.setContentsMargins(12, 10, 12, 10)
        box.setSpacing(8)

        title = W.QLabel("🌊 信号发生器 (Wavegen W1 / W2)")
        title.setObjectName("secTitle")
        box.addWidget(title)

        self.awg_tabs = W.QTabWidget()
        self.awg_widgets = []

        for idx in range(2):
            page = W.QWidget()
            p_layout = W.QVBoxLayout(page)
            p_layout.setContentsMargins(8, 8, 8, 8)
            p_layout.setSpacing(6)

            chk_en = W.QPushButton(f"⚡ 启用 W{idx + 1} 输出")
            chk_en.setCheckable(True)
            chk_en.setChecked(idx == 0)
            chk_en.toggled.connect(lambda _, i=idx: self._emit_awg_config(i))
            p_layout.addWidget(chk_en)

            grid = W.QGridLayout()
            grid.setSpacing(6)

            grid.addWidget(W.QLabel("波形类型:"), 0, 0)
            cmb_fn = W.QComboBox()
            cmb_fn.addItems(list(WAVEFORM_FUNCTIONS.keys()))
            if idx == 1:
                cmb_fn.setCurrentText("Square")
            cmb_fn.currentTextChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            grid.addWidget(cmb_fn, 0, 1)

            grid.addWidget(W.QLabel("频率 (Hz):"), 1, 0)
            freq_row = W.QHBoxLayout()
            spn_freq = W.QDoubleSpinBox()
            spn_freq.setRange(0.1, 25_000_000.0)
            spn_freq.setDecimals(1)
            spn_freq.setSingleStep(1000.0)
            spn_freq.setValue(1_000.0 if idx == 0 else 2_000.0)
            spn_freq.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            freq_row.addWidget(spn_freq, 1)
            grid.addLayout(freq_row, 1, 1)

            # Quick frequency presets
            preset_row = W.QHBoxLayout()
            for lbl, hz in [("1k", 1e3), ("10k", 1e4), ("100k", 1e5), ("1M", 1e6), ("5M", 5e6)]:
                b = W.QPushButton(lbl)
                b.setFixedHeight(24)
                b.clicked.connect(lambda _, s=spn_freq, v=hz: s.setValue(v))
                preset_row.addWidget(b)
            grid.addLayout(preset_row, 2, 0, 1, 2)

            grid.addWidget(W.QLabel("幅值 (Amplitude):"), 3, 0)
            spn_amp = W.QDoubleSpinBox()
            spn_amp.setRange(0.01, 5.0)
            spn_amp.setDecimals(3)
            spn_amp.setSingleStep(0.1)
            spn_amp.setValue(1.0)
            spn_amp.setSuffix(" Vpk")
            spn_amp.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            grid.addWidget(spn_amp, 3, 1)

            grid.addWidget(W.QLabel("直流偏置 (Offset):"), 4, 0)
            spn_off = W.QDoubleSpinBox()
            spn_off.setRange(-5.0, 5.0)
            spn_off.setDecimals(3)
            spn_off.setSingleStep(0.1)
            spn_off.setValue(0.0)
            spn_off.setSuffix(" V")
            spn_off.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            grid.addWidget(spn_off, 4, 1)

            grid.addWidget(W.QLabel("占空比 / 对称度:"), 5, 0)
            spn_sym = W.QDoubleSpinBox()
            spn_sym.setRange(0.0, 100.0)
            spn_sym.setDecimals(1)
            spn_sym.setValue(50.0)
            spn_sym.setSuffix(" %")
            spn_sym.valueChanged.connect(lambda _, i=idx: self._emit_awg_config(i))
            grid.addWidget(spn_sym, 5, 1)

            p_layout.addLayout(grid)
            self.awg_tabs.addTab(page, f"W{idx + 1} 通道")
            self.awg_widgets.append((chk_en, cmb_fn, spn_freq, spn_amp, spn_off, spn_sym))

        box.addWidget(self.awg_tabs)
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
        self.req_fft_cfg.connect(self.worker.update_fft_settings)

        self.worker.connected.connect(self._on_device_connected)
        self.worker.status_changed.connect(self.lbl_status.setText)
        self.worker.error_occurred.connect(self._on_worker_error)
        self.worker.frame_ready.connect(self._on_frame_ready)

        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()

        QtCore.QTimer.singleShot(80, self.req_connect.emit)

    def _on_device_connected(self, info: dict):
        self.lbl_dev_badge.setText(f"设备: {info['name']} · {info['serial']}")
        self.lbl_gpu_badge.setText(f"⚡ FFT: {info['fft_device']}")

        backends = info.get("fft_backends", [])
        if backends:
            with QtCore.QSignalBlocker(self.cmb_fft_backend):
                self.cmb_fft_backend.clear()
                for code, label in backends:
                    self.cmb_fft_backend.addItem(label, code)

        # Apply initial scope and AWG settings
        self._emit_scope_config()
        self._emit_awg_config(0)
        self._emit_awg_config(1)

    def _on_worker_error(self, msg: str):
        self.lbl_status.setText(f"错误: {msg}")

    def _on_dpo_toggled(self, checked: bool):
        for curve in (self.scope_curve_ch1, self.scope_curve_ch2, self.fft_curve_ch1, self.fft_curve_ch2):
            curve.set_dpo(checked)

    def _on_trig_line_dragged(self):
        val = float(self.trig_line.value())
        with QtCore.QSignalBlocker(self.spn_trig_lvl):
            self.spn_trig_lvl.setValue(val)
        self._emit_scope_config()

    def _on_trig_spin_changed(self, val: float):
        self.trig_line.setValue(float(val))
        self._emit_scope_config()

    def _emit_scope_config(self, *_):
        fs = float(self.cmb_rate.currentData() or 20_000_000.0)
        buf = int(self.cmb_buf.currentData() or 16384)
        ch_cfgs = []
        for chk, cmb_range, chk_ac in self.ch_controls:
            ch_cfgs.append(
                ChannelConfig(
                    enabled=chk.isChecked(),
                    range_v=float(cmb_range.currentData() or 5.0),
                    offset_v=0.0,
                    coupling_ac=chk_ac.isChecked(),
                )
            )
        trig = TriggerConfig(
            mode=self.cmb_trig_mode.currentText(),
            source=self.cmb_trig_src.currentText(),
            slope=self.cmb_trig_slope.currentText(),
            level_v=float(self.spn_trig_lvl.value()),
            position_s=0.0,
        )
        self.scope_cfg = ScopeConfig(sample_rate_hz=fs, buffer_size=buf, channels=ch_cfgs, trigger=trig)
        self.scope_curve_ch1.setVisible(ch_cfgs[0].enabled)
        self.scope_curve_ch2.setVisible(ch_cfgs[1].enabled)
        self.fft_curve_ch1.setVisible(ch_cfgs[0].enabled and self.chk_fft_en.isChecked())
        self.fft_curve_ch2.setVisible(ch_cfgs[1].enabled and self.chk_fft_en.isChecked())
        self.req_scope_cfg.emit(self.scope_cfg)

    def _emit_awg_config(self, idx: int):
        chk_en, cmb_fn, spn_freq, spn_amp, spn_off, spn_sym = self.awg_widgets[idx]
        cfg = AwgChannelConfig(
            enabled=chk_en.isChecked(),
            function=cmb_fn.currentText(),
            frequency_hz=float(spn_freq.value()),
            amplitude_v=float(spn_amp.value()),
            offset_v=float(spn_off.value()),
            symmetry_pct=float(spn_sym.value()),
        )
        self.awg_cfgs[idx] = cfg
        self.req_awg_cfg.emit(idx, cfg)

    def _emit_fft_config(self, *_):
        settings = {
            "enabled": self.chk_fft_en.isChecked(),
            "backend": self.cmb_fft_backend.currentData() or "opencl",
            "window": self.cmb_fft_win.currentData() or "FlatTop",
            "average": int(self.spn_fft_avg.value()),
            "n_fft": int(self.cmb_fft_size.currentData() or 0),
        }
        self.fft_curve_ch1.setVisible(self.scope_cfg.channels[0].enabled and settings["enabled"])
        self.fft_curve_ch2.setVisible(self.scope_cfg.channels[1].enabled and settings["enabled"])
        self.req_fft_cfg.emit(settings)

    def autoscale_views(self):
        self.scope_plot.enableAutoRange(axis="xy", enable=True)
        self.scope_plot.enableAutoRange(axis="xy", enable=False)
        self.fft_plot.enableAutoRange(axis="x", enable=True)
        self.fft_plot.enableAutoRange(axis="x", enable=False)
        self.fft_plot.setYRange(-115.0, 15.0)

    def _on_frame_ready(self, data: dict):
        t = data["t"]
        ch1 = data["ch1"]
        ch2 = data["ch2"]
        telemetry_updated = data.get("telemetry_updated", True)

        if self.scope_cfg.channels[0].enabled:
            self.scope_curve_ch1.set_fast_data(t, ch1)
        if self.scope_cfg.channels[1].enabled:
            self.scope_curve_ch2.set_fast_data(t, ch2)

        # Auto-fit X range if view hasn't matched current time span
        xr, _ = self.scope_plot.getViewBox().viewRange()
        span_t = float(t[-1] - t[0]) if t.size > 1 else 1e-3
        if abs((xr[1] - xr[0]) - span_t) > span_t * 0.5:
            self.scope_plot.setXRange(float(t[0]), float(t[-1]), padding=0.0)

        fft_dict = data.get("fft", {})
        fft_ms = 0.0
        f1 = fft_dict.get("ch1")
        f2 = fft_dict.get("ch2")

        if f1:
            self.fft_curve_ch1.set_fast_data(f1["freqs"], f1["dbv"])
            fft_ms += f1.get("compute_ms", 0.0)
        if f2:
            self.fft_curve_ch2.set_fast_data(f2["freqs"], f2["dbv"])
            fft_ms += f2.get("compute_ms", 0.0)

        if not telemetry_updated:
            return

        meas = data.get("meas", {})
        m1 = meas.get("ch1", {})
        m2 = meas.get("ch2", {})
        if m1:
            self._set_label_if_changed(
                self.lbl_meas_ch1,
                f"Vpp: {format_si(m1['vpp'], 'V')}  |  Vrms: {format_si(m1['vrms'], 'V')}  |  "
                f"Freq: {format_si(m1['freq_hz'], 'Hz')}  |  Duty: {m1['duty_pct']:.1f}%",
            )
        else:
            self._set_label_if_changed(self.lbl_meas_ch1, "已禁用")

        if m2:
            self._set_label_if_changed(
                self.lbl_meas_ch2,
                f"Vpp: {format_si(m2['vpp'], 'V')}  |  Vrms: {format_si(m2['vrms'], 'V')}  |  "
                f"Freq: {format_si(m2['freq_hz'], 'Hz')}  |  Duty: {m2['duty_pct']:.1f}%",
            )
        else:
            self._set_label_if_changed(self.lbl_meas_ch2, "已禁用")

        primary_fft = f1 or f2
        if primary_fft:
            freqs = primary_fft["freqs"]
            fx_r, _ = self.fft_plot.getViewBox().viewRange()
            met = primary_fft.get("metrics", {})
            fund_f = met.get("fund_freq", 0.0) if met else 0.0
            # Focus initial X range on meaningful harmonic span (up to 15x fundamental or Nyquist)
            target_fmax = float(freqs[-1]) if freqs.size > 1 else 1e6
            if fund_f > 10.0:
                target_fmax = min(target_fmax, max(fund_f * 15.0, float(freqs[-1]) * 0.02))
            if not getattr(self, "_fft_x_initialized", False) or abs(fx_r[1] - self._last_fft_fmax) < 1e-3:
                self._fft_x_initialized = True
                self._last_fft_fmax = target_fmax
                self.fft_plot.setXRange(0.0, target_fmax, padding=0.0)
            if met:
                peaks = met.get("peaks", [])
                if peaks:
                    px = [p[0] for p in peaks]
                    py = [p[1] for p in peaks]
                    self.fft_peaks_scatter.setData(px, py)
                self._set_label_if_changed(
                    self.lbl_meas_fft,
                    f"Peak: {format_si(met['fund_freq'], 'Hz')} ({met['fund_dbv']:.2f} dBV)  |  "
                    f"THD: {met['thd_db']:.1f} dB ({met['thd_pct']:.2f}%)  |  "
                    f"SNR: {met['snr_db']:.1f} dB  |  SFDR: {met['sfdr_db']:.1f} dB",
                )
            self._set_label_if_changed(
                self.lbl_gpu_badge,
                f"⚡ {primary_fft['device_name']} · {primary_fft['n_fft']} pts · {fft_ms:.2f} ms",
            )

        fps = data.get("fps", 0.0)
        self._set_label_if_changed(self.lbl_fps, f"UI/Acq: {fps:.1f} FPS  |  FFT 耗时: {fft_ms:.2f} ms")

    @staticmethod
    def _set_label_if_changed(label: W.QLabel, text: str):
        if label.text() != text:
            label.setText(text)

    def closeEvent(self, event: QtGui.QCloseEvent):
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
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_THEME_QSS)

    win = MainWindow(args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
