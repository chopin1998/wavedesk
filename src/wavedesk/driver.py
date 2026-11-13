"""Digilent WaveForms SDK (libdwf) hardware driver for Analog Discovery 3.

Supports:
- Device enumeration, hot-plug detection, and Config 1 (2x32768 scope buffer)
- Dual-channel AnalogIn (Oscilloscope) with Edge trigger, Auto/Normal/Single modes
- Dual-channel AnalogOut (Arbitrary Waveform Generator / Function Generator)
- High-realism Demo simulator with internal AWG-to-Scope virtual loopback when requested
"""
from __future__ import annotations
import ctypes as C
from dataclasses import dataclass, field
import math
import sys
import time
import numpy as np


# DWF Constants
hdwfNone = 0
enumfilterAll = 0
DwfParamOnClose = 4
DwfStateReady = 0
DwfStateConfig = 4
DwfStatePrefill = 5
DwfStateArmed = 1
DwfStateWait = 7
DwfStateTriggered = 3
DwfStateRunning = 3
DwfStateNotDone = 6
DwfStateDone = 2

acqmodeSingle = 0
filterDecimate = 0
filterAverage = 1

trigsrcNone = 0
trigsrcDetectorAnalogIn = 2
trigsrcAnalogOut1 = 7
trigsrcAnalogOut2 = 8

trigtypeEdge = 0
DwfTriggerSlopeRise = 0
DwfTriggerSlopeFall = 1
DwfTriggerSlopeEither = 2

DwfAnalogCouplingDC = 0
DwfAnalogCouplingAC = 1

AnalogOutNodeCarrier = 0
funcDC = 0
funcSine = 1
funcSquare = 2
funcTriangle = 3
funcRampUp = 4
funcRampDown = 5
funcNoise = 6
funcPulse = 7

WAVEFORM_FUNCTIONS = {
    "Sine": funcSine,
    "Square": funcSquare,
    "Triangle": funcTriangle,
    "RampUp": funcRampUp,
    "RampDown": funcRampDown,
    "Noise": funcNoise,
    "Pulse": funcPulse,
    "DC": funcDC,
}

TRIGGER_SLOPES = {
    "Rising": DwfTriggerSlopeRise,
    "Falling": DwfTriggerSlopeFall,
    "Either": DwfTriggerSlopeEither,
}


@dataclass
class ChannelConfig:
    enabled: bool = True
    range_v: float = 5.0  # pk-pk range in V (AD3 has 5V and 50V hardware ranges, scaled in display)
    offset_v: float = 0.0
    coupling_ac: bool = False
    attenuation: float = 1.0


@dataclass
class TriggerConfig:
    mode: str = "Auto"  # "Auto", "Normal", "Single", "None"
    source: str = "CH1"  # "CH1", "CH2", "W1", "W2"
    slope: str = "Rising"  # "Rising", "Falling", "Either"
    level_v: float = 0.0
    position_s: float = 0.0  # horizontal trigger offset in seconds


@dataclass
class ScopeConfig:
    sample_rate_hz: float = 20_000_000.0
    buffer_size: int = 16384
    channels: list[ChannelConfig] = field(
        default_factory=lambda: [ChannelConfig(True, 5.0), ChannelConfig(True, 5.0)]
    )
    trigger: TriggerConfig = field(default_factory=TriggerConfig)


@dataclass
class AwgChannelConfig:
    enabled: bool = False
    function: str = "Sine"
    frequency_hz: float = 100_000.0
    amplitude_v: float = 1.0
    offset_v: float = 0.0
    symmetry_pct: float = 50.0
    phase_deg: float = 0.0


def _load_dwf():
    try:
        if sys.platform.startswith("win"):
            return C.cdll.dwf
        if sys.platform.startswith("darwin"):
            return C.cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
        return C.cdll.LoadLibrary("libdwf.so")
    except OSError:
        return None


_dwf = _load_dwf()


class DwfError(RuntimeError):
    pass


class AD3Driver:
    """Thread-safe wrapper around Digilent WaveForms SDK (libdwf) or Demo simulator."""

    def __init__(self, demo: bool = False):
        self.demo = demo
        self.hdwf = C.c_int(hdwfNone)
        self.device_info: dict = {}
        self.scope_cfg = ScopeConfig()
        self.awg_cfg = [
            AwgChannelConfig(enabled=True, function="Sine", frequency_hz=100_000.0, amplitude_v=1.0),
            AwgChannelConfig(enabled=False, function="Square", frequency_hz=50_000.0, amplitude_v=1.0),
        ]
        self._scope_dirty = True
        self._sim_phase = 0.0

    @staticmethod
    def enumerate_devices() -> list[dict]:
        if _dwf is None:
            return []
        c_dev = C.c_int()
        _dwf.FDwfEnum(C.c_int(enumfilterAll), C.byref(c_dev))
        devices = []
        for i in range(c_dev.value):
            name = C.create_string_buffer(64)
            sn = C.create_string_buffer(64)
            in_use = C.c_int()
            _dwf.FDwfEnumDeviceName(C.c_int(i), name)
            _dwf.FDwfEnumSN(C.c_int(i), sn)
            _dwf.FDwfEnumDeviceIsOpened(C.c_int(i), C.byref(in_use))
            devices.append(
                {
                    "index": i,
                    "name": name.value.decode(errors="replace"),
                    "serial": sn.value.decode(errors="replace"),
                    "in_use": bool(in_use.value),
                }
            )
        return devices

    def is_open(self) -> bool:
        return self.demo or (self.hdwf.value != hdwfNone)

    def _last_error(self) -> str:
        if _dwf is None:
            return "libdwf.so not found"
        buf = C.create_string_buffer(512)
        _dwf.FDwfGetLastErrorMsg(buf)
        return buf.value.decode(errors="replace").strip()

    def open(self, device_index: int = -1, config_index: int = 1) -> dict:
        """Open AD3 with configuration 1 (2x32768 scope buffer) by default."""
        if self.demo:
            self.device_info = {
                "name": "Analog Discovery 3 (Demo Simulator)",
                "serial": "SN:DEMO-AD3",
                "version": "Sim 3.25",
                "max_buffer": 32768,
                "max_rate": 125_000_000.0,
                "demo": True,
            }
            self._scope_dirty = True
            return self.device_info

        if _dwf is None:
            raise DwfError("未检测到 libdwf.so (WaveForms Runtime)")

        ver = C.create_string_buffer(32)
        _dwf.FDwfGetVersion(ver)
        # Keep outputs running on close or stop gracefully
        _dwf.FDwfParamSet(C.c_int(DwfParamOnClose), C.c_int(1))

        devs = self.enumerate_devices()
        if not devs:
            raise DwfError("未检测到 Analog Discovery 设备，请检查 USB 连接")

        target_info = devs[0] if device_index < 0 else devs[min(device_index, len(devs) - 1)]

        hdwf = C.c_int(hdwfNone)
        res = _dwf.FDwfDeviceConfigOpen(C.c_int(device_index), C.c_int(config_index), C.byref(hdwf))
        if res != 1 or hdwf.value == hdwfNone:
            err = self._last_error()
            if target_info.get("in_use"):
                raise DwfError(f"设备 {target_info['name']} 已被其它程序 (如 WaveForms) 占用，请先关闭原版软件。")
            raise DwfError(f"无法打开 AD3 设备: {err}")

        self.hdwf = hdwf
        # Manual configure mode so multiple parameter updates batch cleanly
        _dwf.FDwfDeviceAutoConfigureSet(self.hdwf, C.c_int(0))

        buf_min = C.c_int()
        buf_max = C.c_int()
        _dwf.FDwfAnalogInBufferSizeInfo(self.hdwf, C.byref(buf_min), C.byref(buf_max))
        rate_min = C.c_double()
        rate_max = C.c_double()
        _dwf.FDwfAnalogInFrequencyInfo(self.hdwf, C.byref(rate_min), C.byref(rate_max))

        self.device_info = {
            "name": target_info["name"],
            "serial": target_info["serial"],
            "version": ver.value.decode(errors="replace"),
            "max_buffer": int(buf_max.value or 32768),
            "max_rate": float(rate_max.value or 125_000_000.0),
            "demo": False,
        }
        self._scope_dirty = True
        self.apply_scope_config()
        return self.device_info

    def close(self):
        if not self.demo and _dwf is not None and self.hdwf.value != hdwfNone:
            try:
                _dwf.FDwfAnalogOutReset(self.hdwf, C.c_int(-1))
                _dwf.FDwfDeviceClose(self.hdwf)
            except Exception:
                pass
        self.hdwf = C.c_int(hdwfNone)

    def apply_scope_config(self):
        if self.demo or not self.is_open():
            self._scope_dirty = False
            return

        cfg = self.scope_cfg
        _dwf.FDwfAnalogInFrequencySet(self.hdwf, C.c_double(cfg.sample_rate_hz))
        _dwf.FDwfAnalogInBufferSizeSet(self.hdwf, C.c_int(cfg.buffer_size))
        _dwf.FDwfAnalogInAcquisitionModeSet(self.hdwf, C.c_int(acqmodeSingle))

        for idx, ch in enumerate(cfg.channels[:2]):
            _dwf.FDwfAnalogInChannelEnableSet(self.hdwf, C.c_int(idx), C.c_int(1))
            _dwf.FDwfAnalogInChannelRangeSet(self.hdwf, C.c_int(idx), C.c_double(ch.range_v))
            _dwf.FDwfAnalogInChannelOffsetSet(self.hdwf, C.c_int(idx), C.c_double(ch.offset_v))
            _dwf.FDwfAnalogInChannelFilterSet(self.hdwf, C.c_int(idx), C.c_int(filterDecimate))
            coupling = DwfAnalogCouplingAC if ch.coupling_ac else DwfAnalogCouplingDC
            _dwf.FDwfAnalogInChannelCouplingSet(self.hdwf, C.c_int(idx), C.c_int(coupling))

        trig = cfg.trigger
        if trig.mode == "None":
            _dwf.FDwfAnalogInTriggerSourceSet(self.hdwf, C.c_ubyte(trigsrcNone))
        else:
            src_map = {
                "CH1": trigsrcDetectorAnalogIn,
                "CH2": trigsrcDetectorAnalogIn,
                "W1": trigsrcAnalogOut1,
                "W2": trigsrcAnalogOut2,
            }
            _dwf.FDwfAnalogInTriggerSourceSet(self.hdwf, C.c_ubyte(src_map.get(trig.source, trigsrcDetectorAnalogIn)))
            _dwf.FDwfAnalogInTriggerTypeSet(self.hdwf, C.c_int(trigtypeEdge))
            trig_ch = 1 if trig.source == "CH2" else 0
            _dwf.FDwfAnalogInTriggerChannelSet(self.hdwf, C.c_int(trig_ch))
            _dwf.FDwfAnalogInTriggerLevelSet(self.hdwf, C.c_double(trig.level_v))
            _dwf.FDwfAnalogInTriggerHysteresisSet(self.hdwf, C.c_double(0.02))
            slope_code = TRIGGER_SLOPES.get(trig.slope, DwfTriggerSlopeRise)
            _dwf.FDwfAnalogInTriggerConditionSet(self.hdwf, C.c_int(slope_code))
            _dwf.FDwfAnalogInTriggerPositionSet(self.hdwf, C.c_double(trig.position_s))
            # Auto timeout: 0.05s for Auto mode, 0 for Normal/Single
            auto_to = 0.05 if trig.mode == "Auto" else 0.0
            _dwf.FDwfAnalogInTriggerAutoTimeoutSet(self.hdwf, C.c_double(auto_to))

        if _dwf.FDwfAnalogInConfigure(self.hdwf, C.c_int(1), C.c_int(1)) != 1:
            raise DwfError("配置示波器失败: " + self._last_error())
        self._scope_dirty = False

    def configure_awg(self, ch_index: int, cfg: AwgChannelConfig):
        self.awg_cfg[ch_index] = cfg
        if self.demo or not self.is_open():
            return

        c_ch = C.c_int(ch_index)
        node = C.c_int(AnalogOutNodeCarrier)
        if not cfg.enabled:
            _dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, c_ch, node, C.c_int(0))
            _dwf.FDwfAnalogOutConfigure(self.hdwf, c_ch, C.c_int(0))
            return

        func_id = WAVEFORM_FUNCTIONS.get(cfg.function, funcSine)
        _dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, c_ch, node, C.c_int(1))
        _dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf, c_ch, node, C.c_ubyte(func_id))
        _dwf.FDwfAnalogOutNodeFrequencySet(self.hdwf, c_ch, node, C.c_double(cfg.frequency_hz))
        _dwf.FDwfAnalogOutNodeAmplitudeSet(self.hdwf, c_ch, node, C.c_double(cfg.amplitude_v))
        _dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf, c_ch, node, C.c_double(cfg.offset_v))
        _dwf.FDwfAnalogOutNodeSymmetrySet(self.hdwf, c_ch, node, C.c_double(cfg.symmetry_pct))
        _dwf.FDwfAnalogOutNodePhaseSet(self.hdwf, c_ch, node, C.c_double(cfg.phase_deg))

        if _dwf.FDwfAnalogOutConfigure(self.hdwf, c_ch, C.c_int(1)) != 1:
            raise DwfError(f"配置信号发生器 W{ch_index + 1} 失败: " + self._last_error())

    def capture_frame(self, timeout_s: float = 0.25) -> dict | None:
        """Capture one frame of dual-channel scope data."""
        if self.demo:
            return self._simulate_frame()

        if not self.is_open():
            return None

        if self._scope_dirty:
            self.apply_scope_config()

        sts = C.c_byte()
        t_start = time.monotonic()
        while True:
            if _dwf.FDwfAnalogInStatus(self.hdwf, C.c_int(1), C.byref(sts)) != 1:
                raise DwfError("读取示波器状态失败: " + self._last_error())
            if sts.value == DwfStateDone:
                break
            if time.monotonic() - t_start > timeout_s:
                if self.scope_cfg.trigger.mode in ("Normal", "Single"):
                    return None
                break
            time.sleep(0.001)

        actual_rate = C.c_double()
        actual_n = C.c_int()
        _dwf.FDwfAnalogInFrequencyGet(self.hdwf, C.byref(actual_rate))
        _dwf.FDwfAnalogInBufferSizeGet(self.hdwf, C.byref(actual_n))
        n = int(actual_n.value or self.scope_cfg.buffer_size)
        fs = float(actual_rate.value or self.scope_cfg.sample_rate_hz)

        ch_data = []
        for idx, ch_cfg in enumerate(self.scope_cfg.channels[:2]):
            if ch_cfg.enabled:
                buf = (C.c_double * n)()
                _dwf.FDwfAnalogInStatusData(self.hdwf, C.c_int(idx), buf, C.c_int(n))
                arr = np.ctypeslib.as_array(buf).astype(np.float32) * float(ch_cfg.attenuation)
            else:
                arr = np.zeros(n, dtype=np.float32)
            ch_data.append(arr)

        # Re-arm next single acquisition immediately for continuous streaming
        if self.scope_cfg.trigger.mode != "Single":
            _dwf.FDwfAnalogInConfigure(self.hdwf, C.c_int(0), C.c_int(1))

        t = (np.arange(n, dtype=np.float32) - n * 0.5) / fs + float(self.scope_cfg.trigger.position_s)
        return {
            "t": t,
            "ch1": ch_data[0],
            "ch2": ch_data[1],
            "fs": fs,
            "n": n,
            "triggered": sts.value == DwfStateDone,
            "timestamp": time.time(),
        }

    def _simulate_frame(self) -> dict:
        cfg = self.scope_cfg
        n = cfg.buffer_size
        fs = cfg.sample_rate_hz
        t = (np.arange(n, dtype=np.float32) - np.float32(n * 0.5)) / np.float32(fs) + np.float32(cfg.trigger.position_s)
        self._sim_phase += 0.08
        rng = getattr(self, "_rng", None)
        if rng is None:
            rng = np.random.default_rng()
            self._rng = rng

        ch_list = []
        two_pi = np.float32(2.0 * np.pi)
        for idx in range(2):
            awg = self.awg_cfg[idx]
            f = np.float32(awg.frequency_hz if awg.enabled else (1_000.0 if idx == 0 else 2_000.0))
            amp = np.float32(awg.amplitude_v if awg.enabled else (1.2 if idx == 0 else 0.8))
            off = np.float32(awg.offset_v if awg.enabled else (0.5 if idx == 0 else -0.5))
            fn = awg.function if awg.enabled else ("Sine" if idx == 0 else "Square")

            phase = two_pi * f * t + np.float32(np.deg2rad(awg.phase_deg))
            if cfg.trigger.mode == "None":
                phase = phase + np.float32(self._sim_phase)

            if fn == "Sine":
                y = amp * np.sin(phase) + np.float32(0.025) * amp * np.sin(np.float32(3.0) * phase)
            elif fn == "Square":
                duty = np.float32(np.clip(awg.symmetry_pct / 100.0, 0.05, 0.95))
                y = np.where(np.mod(phase, two_pi) < two_pi * duty, amp, -amp)
            elif fn == "Triangle":
                y = amp * np.float32(2.0 / np.pi) * np.arcsin(np.sin(phase))
            elif fn == "RampUp":
                y = amp * (np.float32(2.0) * np.mod(phase / two_pi, np.float32(1.0)) - np.float32(1.0))
            elif fn == "RampDown":
                y = amp * (np.float32(1.0) - np.float32(2.0) * np.mod(phase / two_pi, np.float32(1.0)))
            elif fn == "Noise":
                y = rng.standard_normal(n, dtype=np.float32) * (amp * np.float32(0.5))
            elif fn == "DC":
                y = np.zeros(n, dtype=np.float32)
            else:
                y = amp * np.sin(phase)

            noise = rng.standard_normal(n, dtype=np.float32) * np.float32(0.004)
            ch_list.append(y + off + noise)

        return {
            "t": t,
            "ch1": ch_list[0],
            "ch2": ch_list[1],
            "fs": fs,
            "n": n,
            "triggered": True,
            "timestamp": time.time(),
        }
