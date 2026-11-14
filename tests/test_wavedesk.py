from __future__ import annotations
import numpy as np
from wavedesk.gpu_fft import GpuFftEngine
from wavedesk.driver import AD3Driver, AwgChannelConfig, select_optimal_sample_rate
from wavedesk.measurements import measure_channel
from wavedesk.gpu_view import build_sdf_ribbon_vertices


def test_gpu_fft_matches_cpu():
    fs = 50_000_000.0
    n = 16384
    t = np.arange(n, dtype=np.float32) / fs
    # 1.25 MHz 1 Vrms (1.4142 Vpk) sine wave + 3rd harmonic
    sig = 1.41421356 * np.sin(2.0 * np.pi * 1_250_000.0 * t) + 0.04 * np.sin(2.0 * np.pi * 3_750_000.0 * t)

    eng_auto = GpuFftEngine("auto")
    res_auto = eng_auto.compute(sig, fs, window_name="FlatTop")

    eng_cpu = GpuFftEngine("cpu")
    res_cpu = eng_cpu.compute(sig, fs, window_name="FlatTop")

    assert res_auto["n_fft"] == n
    assert np.allclose(res_auto["vrms"], res_cpu["vrms"], atol=1e-4)
    met = res_auto["metrics"]
    assert abs(met["fund_freq"] - 1_250_000.0) < 4000.0
    assert abs(met["fund_dbv"] - 0.0) < 0.05


def test_measurements_and_demo_driver():
    drv = AD3Driver(demo=True)
    info = drv.open()
    assert info["demo"] is True
    drv.configure_awg(0, AwgChannelConfig(enabled=True, function="Sine", frequency_hz=200_000.0, amplitude_v=1.5))
    drv.configure_awg(1, AwgChannelConfig(enabled=False, function="Sine", frequency_hz=200_000.0, amplitude_v=1.0))
    frame = drv.capture_frame()
    assert frame is not None
    m1 = measure_channel(frame["t"], frame["ch1"], frame["fs"])
    assert abs(m1["vpp"] - 3.0) < 0.25
    assert abs(m1["freq_hz"] - 200_000.0) < 5000.0
    drv.close()


def test_mixer_if_trigger_stabilization_and_lpf():
    drv = AD3Driver(demo=True)
    drv.open()
    drv.configure_awg(0, AwgChannelConfig(enabled=True, function="Sine", frequency_hz=1_000_000.0, amplitude_v=1.0))
    drv.configure_awg(1, AwgChannelConfig(enabled=True, function="Sine", frequency_hz=1_100_000.0, amplitude_v=1.0))
    drv.scope_cfg.sample_rate_hz = select_optimal_sample_rate(5e-6, buffer_size=32768)
    assert drv.scope_cfg.sample_rate_hz >= 50_000_000.0

    f_wb = drv.capture_frame()
    assert f_wb is not None
    assert f_wb["ch1_env"] is not None
    assert abs(f_wb["if_freq_hz"] - 100_000.0) < 3_000.0

    # Enable 500 kHz LPF to extract clean 100 kHz IF difference frequency
    drv.scope_cfg.channels[0].lpf_cutoff_hz = 500_000.0
    f_lpf = drv.capture_frame()
    m_lpf = measure_channel(f_lpf["t"], f_lpf["ch1"], f_lpf["fs"])
    assert abs(m_lpf["freq_hz"] - 100_000.0) < 3_000.0
    assert 0.65 < m_lpf["vpp"] < 0.95
    drv.close()


def test_sdf_ribbon_envelope_decimation():
    x = np.linspace(-1e-3, 1e-3, 32768, dtype=np.float32)
    y = np.sin(2 * np.pi * 50000 * x).astype(np.float32)
    verts = build_sdf_ribbon_vertices(x, y, ((-1e-3, 1e-3), (-2.0, 2.0)), (1000.0, 400.0), max_segments=1024)
    # 1024 buckets * 2 points/bucket * 2 vertices/point = 4096 vertices
    assert verts.shape == (4096, 5)
    assert np.all(np.isfinite(verts))
    # Verify adjacent vertex normals never flip orientation (no bowtie twists)
    nx = verts[0::2, 2]
    ny = verts[0::2, 3]
    dots = nx[:-1] * nx[1:] + ny[:-1] * ny[1:]
    assert np.all(dots > 0.0)


def test_small_signal_measurement_and_demo_am_fm():
    drv = AD3Driver(demo=True)
    drv.open()
    drv.configure_awg(
        0,
        AwgChannelConfig(
            enabled=True,
            function="Sine",
            frequency_hz=500_000.0,
            amplitude_v=0.004,
            am_enabled=True,
            am_frequency_hz=10_000.0,
            am_depth_pct=50.0,
        ),
    )
    drv.configure_awg(1, AwgChannelConfig(enabled=False))
    frame = drv.capture_frame()
    assert frame is not None
    m1 = measure_channel(frame["t"], frame["ch1"], frame["fs"])
    assert m1["freq_hz"] > 100_000.0
    drv.close()

    # Verify clean small-signal (< 25 mVpp) frequency measurement
    fs = 10_000_000.0
    t = np.arange(8192, dtype=np.float32) / fs
    y_small = (0.006 * np.sin(2.0 * np.pi * 250_000.0 * t)).astype(np.float32)
    ms = measure_channel(t, y_small, fs)
    assert 0.010 < ms["vpp"] < 0.015
    assert abs(ms["freq_hz"] - 250_000.0) < 2000.0


def test_sdf_shader_smoothstep_edges():
    import re
    from wavedesk.gpu_view import SDF_LINE_FRAG

    calls = re.findall(r"smoothstep\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,", SDF_LINE_FRAG)
    assert calls
    for low, high in calls:
        assert float(low) < float(high)


def test_in_band_peak_detection_not_masked_by_strong_lo_harmonics():
    freqs = np.linspace(0.0, 25_000_000.0, 16385, dtype=np.float32)
    dbv = np.full_like(freqs, -105.0)
    # 9 huge out-of-band MHz harmonics (> -20 dBV) that would fill a global top-8 list
    for mhz in range(1, 10):
        idx = int(np.argmin(np.abs(freqs - mhz * 2_500_000.0)))
        dbv[idx] = -10.0
    # Modest in-band IF peak at 250 kHz (-68 dBV)
    if_idx = int(np.argmin(np.abs(freqs - 250_000.0)))
    dbv[if_idx] = -68.0
    in_band = GpuFftEngine.find_peaks_in_band(freqs, dbv, 150_000.0, 400_000.0)
    assert len(in_band) == 1
    assert abs(in_band[0][0] - 250_000.0) < 3000.0


def test_ssb_with_carrier_synthesis_purity_and_frequency():
    for f_c in (2_250_000.0, 5_250_000.0):
        f_m = 25_000.0
        for mode, wanted_sb, unwanted_sb in (
            ("USB+C (带载波上边带)", f_c + f_m, f_c - f_m),
            ("LSB+C (带载波下边带)", f_c - f_m, f_c + f_m),
        ):
            y, rep_hz = AD3Driver._synthesize_ssb_vsb(f_c, f_m, 50.0, mode, n_pts=16384)
            assert y.size == 16384
            assert abs(rep_hz - f_m) < 1e-6
            spec = np.abs(np.fft.rfft(y)) / (y.size / 2.0)
            freqs = np.fft.rfftfreq(y.size, d=1.0 / (y.size * rep_hz))
            c_idx = int(np.argmin(np.abs(freqs - f_c)))
            w_idx = int(np.argmin(np.abs(freqs - wanted_sb)))
            u_idx = int(np.argmin(np.abs(freqs - unwanted_sb)))
            assert abs(freqs[c_idx] - f_c) < 1.0
            assert abs(freqs[w_idx] - wanted_sb) < 1.0
            # Sideband should be ~12.04 dB below carrier (0.5 * m = 0.25 at 50% AM depth), unwanted sideband > 80 dB suppressed
            ratio_db = 20.0 * np.log10(spec[w_idx] / spec[c_idx])
            supp_db = 20.0 * np.log10(spec[w_idx] / max(spec[u_idx], 1e-12))
            assert abs(ratio_db - (-12.04)) < 0.1
            assert supp_db > 80.0


def test_zoom_fft_never_undersamples_main_lobe():
    eng = GpuFftEngine("cpu")
    fs = 2_000_000.0
    n = 32768
    t = np.arange(n, dtype=np.float32) / fs
    sig = (1.0 + 0.5 * np.cos(2.0 * np.pi * 25_000.0 * t)) * np.cos(2.0 * np.pi * 200_000.0 * t)
    res = eng.compute_zoom_fft(sig, fs, 0.0, 1_000_000.0, m_bins=2048, window_name="FlatTop")
    peaks = eng.find_peaks_in_band(res["freqs"], res["dbv"], 140_000.0, 260_000.0)
    assert len(peaks) == 3
    # Sorted by magnitude: 200 kHz carrier (-3.01 dBV), then 175 kHz & 225 kHz sidebands (-15.05 dBV)
    assert abs(peaks[0][0] - 200_000.0) < 30.0
    assert abs(peaks[0][1] - (-3.01)) < 0.15
    sb_freqs = sorted([peaks[1][0], peaks[2][0]])
    assert abs(sb_freqs[0] - 175_000.0) < 30.0
    assert abs(sb_freqs[1] - 225_000.0) < 30.0
    assert abs(peaks[1][1] - peaks[2][1]) < 0.15


def test_sdf_ribbon_decimation_covers_full_xrange():
    from wavedesk.gpu_view import build_sdf_ribbon_vertices

    for n in (7505, 10004, 16385):
        x = np.linspace(-250e-6, 250e-6, n, dtype=np.float32)
        y = np.sin(2.0 * np.pi * 200e3 * x).astype(np.float32)
        verts = build_sdf_ribbon_vertices(
            x, y, ((-250e-6, 250e-6), (-0.1, 0.1)), (1668.0, 400.0), max_segments=2502
        )
        assert verts.shape[0] >= 4
        assert float(verts[0, 0]) <= -249e-6
        assert float(verts[-1, 0]) >= 249e-6


def test_polynomial_multi_sine_synthesis_and_demo():
    terms = [
        (100_000.0, 1.0, 0.0),
        (250_000.0, 0.5, 90.0),
        (400_000.0, 0.25, -45.0),
    ]
    y, rep_hz = AD3Driver._synthesize_polynomial(terms, n_pts=16384)
    assert y.size == 16384
    assert float(np.max(np.abs(y))) <= 1.0001
    spec = np.abs(np.fft.rfft(y)) / (y.size / 2.0)
    freqs = np.fft.rfftfreq(y.size, d=1.0 / (y.size * rep_hz))
    a1 = spec[int(np.argmin(np.abs(freqs - 100_000.0)))]
    a2 = spec[int(np.argmin(np.abs(freqs - 250_000.0)))]
    a3 = spec[int(np.argmin(np.abs(freqs - 400_000.0)))]
    assert abs(a2 / a1 - 0.5) < 0.02
    assert abs(a3 / a1 - 0.25) < 0.02

    drv = AD3Driver(demo=True)
    drv.open()
    drv.configure_awg(
        0,
        AwgChannelConfig(
            enabled=True,
            function="Polynomial (多正弦)",
            amplitude_v=1.0,
            poly_terms=terms,
        ),
    )
    frame = drv.capture_frame()
    assert frame is not None and frame["ch1"].size > 1000
    drv.close()


def test_input_interaction_filter_combo_and_cursor_digit():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtGui, QtWidgets as W
    from wavedesk.app import InputInteractionFilter

    app = W.QApplication.instance() or W.QApplication([])
    flt = InputInteractionFilter()

    cmb = W.QComboBox()
    cmb.addItems(["A", "B", "C"])
    cmb.setCurrentIndex(0)
    ev = QtGui.QWheelEvent(
        QtCore.QPointF(5, 5),
        QtCore.QPointF(5, 5),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, 120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    # ComboBox wheel must be blocked
    assert flt.eventFilter(cmb, ev) is True
    assert cmb.currentIndex() == 0

    spn = W.QDoubleSpinBox()
    spn.setRange(0.0, 10000.0)
    spn.setDecimals(3)
    spn.setSuffix(" kHz")
    spn.setValue(99.0)
    # Unfocused spinbox wheel must be blocked without changing value
    assert flt.eventFilter(spn, ev) is True
    assert abs(spn.value() - 99.0) < 1e-6

    # Cursor after tens digit ('9' at index 1 in "99.000 kHz") -> +10 -> 109.000
    spn.lineEdit().setCursorPosition(1)
    assert flt.step_spinbox_digit_at_cursor(spn, 1) is True
    assert abs(spn.value() - 109.0) < 1e-6
    # Cursor stays right after tens digit ('0' at index 2 in "109.000 kHz")
    assert spn.lineEdit().cursorPosition() == 2

    # Cursor after tenths digit ('.' is index 3, tenths '0' is index 4 -> cursor=5) -> -0.1 -> 108.900
    spn.lineEdit().setCursorPosition(5)
    assert flt.step_spinbox_digit_at_cursor(spn, -1) is True
    assert abs(spn.value() - 108.9) < 1e-6
    _ = app


def test_linear_sweep_chirp_demo_and_params():
    drv = AD3Driver(demo=True)
    drv.open()
    drv.scope_cfg.sample_rate_hz = 10_000_000.0
    drv.scope_cfg.buffer_size = 16384
    drv.configure_awg(1, AwgChannelConfig(enabled=False))
    drv.configure_awg(
        0,
        AwgChannelConfig(
            enabled=True,
            function="Sine",
            amplitude_v=1.0,
            sweep_enabled=True,
            sweep_start_hz=100_000.0,
            sweep_stop_hz=1_000_000.0,
            sweep_time_s=0.002,
            sweep_wait_s=0.0,
            sweep_repeat=0,
        ),
    )
    frame = drv.capture_frame()
    assert frame is not None
    y = frame["ch1"]
    assert float(np.ptp(y)) > 1.5
    # Verify broadband chirp energy spans [100 kHz, 1 MHz]
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    freqs = np.fft.rfftfreq(y.size, d=1.0 / frame["fs"])
    in_band = spec[(freqs >= 100_000.0) & (freqs <= 1_000_000.0)]
    out_band = spec[freqs > 1_500_000.0]
    assert float(np.max(in_band)) > 5.0 * float(np.max(out_band))
    drv.close()



