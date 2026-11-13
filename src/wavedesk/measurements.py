"""Fast vectorized time-domain measurements for WaveDesk oscilloscope."""
from __future__ import annotations
import numpy as np


def measure_channel(t: np.ndarray, y: np.ndarray, fs: float) -> dict:
    """Compute Vpp, Vmax, Vmin, Vmean, Vrms, AC-RMS, Frequency, Period, Duty Cycle, Rise Time."""
    if y is None or y.size < 8:
        return {}

    vmax = float(np.max(y))
    vmin = float(np.min(y))
    vpp = vmax - vmin
    vmean = float(np.mean(y))
    vrms = float(np.sqrt(np.mean(y * y)))
    vac_rms = float(np.std(y))

    # Estimate frequency and duty cycle via hysteresis zero-crossing around midpoint
    mid = 0.5 * (vmax + vmin)
    hyst = max(0.05 * vpp, 0.005)
    freq_hz = 0.0
    period_s = 0.0
    duty_pct = 50.0
    rise_s = 0.0

    if vpp > 0.015:
        above = y > mid
        # Rising edges where y crosses mid upward with hysteresis check
        crossings = np.where((~above[:-1]) & above[1:])[0]
        if crossings.size >= 2:
            # Sub-sample linear interpolation for rising edges
            idx = crossings
            y0 = y[idx]
            y1 = y[idx + 1]
            frac = np.where(np.abs(y1 - y0) > 1e-9, (mid - y0) / (y1 - y0), 0.0)
            edge_pos = idx.astype(np.float64) + frac
            # Filter out noise bounce (edges closer than 3 samples)
            diffs = np.diff(edge_pos)
            valid_diffs = diffs[diffs >= 3.0]
            if valid_diffs.size >= 1:
                avg_samples = float(np.median(valid_diffs))
                if avg_samples > 0:
                    freq_hz = float(fs / avg_samples)
                    period_s = 1.0 / freq_hz
                    duty_pct = float(np.mean(above) * 100.0)

        # Estimate 10% - 90% rise time on the cleanest rising edge
        v10 = vmin + 0.1 * vpp
        v90 = vmin + 0.9 * vpp
        if crossings.size >= 1:
            c0 = int(crossings[crossings.size // 2])
            span = max(8, int((fs / max(freq_hz, 100.0)) * 0.45)) if freq_hz > 0 else 64
            seg = y[max(0, c0 - span) : min(y.size, c0 + span)]
            i10 = np.where(seg >= v10)[0]
            i90 = np.where(seg >= v90)[0]
            if i10.size > 0 and i90.size > 0 and i90[0] >= i10[0]:
                rise_s = float((i90[0] - i10[0]) / fs)

    return {
        "vpp": vpp,
        "vmax": vmax,
        "vmin": vmin,
        "vmean": vmean,
        "vrms": vrms,
        "vac_rms": vac_rms,
        "freq_hz": freq_hz,
        "period_s": period_s,
        "duty_pct": duty_pct,
        "rise_s": rise_s,
    }


def format_si(value: float, unit: str, decimals: int = 2) -> str:
    """Format physical quantity with SI prefix (G, M, k, m, u, n)."""
    if not np.isfinite(value):
        return f"-- {unit}"
    abs_v = abs(value)
    if abs_v == 0.0:
        return f"0.{ '0' * decimals } {unit}"
    prefixes = [
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "k"),
        (1.0, ""),
        (1e-3, "m"),
        (1e-6, "µ"),
        (1e-9, "n"),
    ]
    for scale, pfx in prefixes:
        if abs_v >= scale * 0.999:
            return f"{value / scale:.{decimals}f} {pfx}{unit}"
    return f"{value * 1e9:.{decimals}f} n{unit}"
