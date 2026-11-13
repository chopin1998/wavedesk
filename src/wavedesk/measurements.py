"""Fast vectorized time-domain measurements for WaveDesk oscilloscope."""
from __future__ import annotations
import numpy as np


def measure_channel(
    t: np.ndarray,
    y: np.ndarray,
    fs: float,
    env: np.ndarray | None = None,
    if_freq_hz: float = 0.0,
) -> dict:
    """Compute Vpp, Vmax, Vmin, Vmean, Vrms, AC-RMS, Frequency, Period, Duty Cycle, Rise Time, and IF envelope metrics."""
    if y is None or y.size < 8:
        return {}

    margin = y.size // 16 if y.size >= 128 else 0
    y_core = y[margin : y.size - margin] if margin > 0 else y

    vmax = float(np.max(y_core))
    vmin = float(np.min(y_core))
    vpp = vmax - vmin
    vmean = float(np.mean(y_core))
    vrms = float(np.sqrt(np.mean(y_core * y_core)))
    vac_rms = float(np.std(y_core))

    mid = 0.5 * (vmax + vmin)
    hyst = max(0.06 * vpp, 0.008)
    freq_hz = 0.0
    period_s = 0.0
    duty_pct = 50.0
    rise_s = 0.0

    if vpp > 0.025:
        above = y_core > mid
        raw_cross = np.where((~above[:-1]) & above[1:])[0]
        if raw_cross.size >= 2:
            # Validate crossings against hysteresis: require signal to have been below (mid - hyst) before crossing
            # and reach above (mid + hyst) before the next rising crossing
            valid_edges = []
            for k in range(raw_cross.size):
                c_idx = int(raw_cross[k])
                prev_idx = int(raw_cross[k - 1]) if k > 0 else max(0, c_idx - 256)
                next_idx = int(raw_cross[k + 1]) if k + 1 < raw_cross.size else min(y_core.size, c_idx + 256)
                if np.min(y_core[prev_idx : c_idx + 1]) <= mid - hyst and np.max(y_core[c_idx + 1 : next_idx]) >= mid + hyst:
                    y0 = float(y_core[c_idx])
                    y1 = float(y_core[c_idx + 1])
                    frac = (mid - y0) / (y1 - y0) if abs(y1 - y0) > 1e-9 else 0.0
                    valid_edges.append(float(c_idx) + frac)

            if len(valid_edges) >= 2:
                diffs = np.diff(np.asarray(valid_edges, dtype=np.float64))
                valid_diffs = diffs[diffs >= 2.5]
                if valid_diffs.size >= 1:
                    avg_samples = float(np.median(valid_diffs))
                    if avg_samples > 0:
                        freq_hz = float(fs / avg_samples)
                        period_s = 1.0 / freq_hz
                        duty_pct = float(np.mean(above) * 100.0)

        # Estimate 10% - 90% rise time on the cleanest rising edge
        v10 = vmin + 0.1 * vpp
        v90 = vmin + 0.9 * vpp
        if raw_cross.size >= 1:
            c0 = int(raw_cross[raw_cross.size // 2])
            span = max(8, int((fs / max(freq_hz, 100.0)) * 0.45)) if freq_hz > 0 else 64
            seg = y_core[max(0, c0 - span) : min(y_core.size, c0 + span)]
            i10 = np.where(seg >= v10)[0]
            i90 = np.where(seg >= v90)[0]
            if i10.size > 0 and i90.size > 0 and i90[0] >= i10[0]:
                rise_s = float((i90[0] - i10[0]) / fs)

    if env is not None and env.size >= 128:
        em = env.size // 16
        if_vpp = float(np.ptp(env[em : env.size - em]))
    elif env is not None and env.size >= 8:
        if_vpp = float(np.ptp(env))
    else:
        if_vpp = 0.0

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
        "if_freq_hz": float(if_freq_hz),
        "if_vpp": if_vpp,
    }


def format_si(value: float, unit: str, decimals: int = 2) -> str:
    """Format physical quantity with SI prefix (G, M, k, m, u, n)."""
    if not np.isfinite(value):
        return f"-- {unit}"
    abs_v = abs(value)
    if abs_v == 0.0:
        return f"0.{'0' * decimals} {unit}"
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
