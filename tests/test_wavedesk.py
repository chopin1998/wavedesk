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
