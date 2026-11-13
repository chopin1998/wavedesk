"""Digilent WaveForms SDK (libdwf) hardware driver for Analog Discovery 3.

Supports:
- Device enumeration, hot-plug detection, and Config 1 (2x32768 scope buffer)
- Dual-channel AnalogIn (Oscilloscope) with Edge trigger, Auto/Normal/Single modes,
  hardware anti-aliasing filterAverage, and sub-sample multi-tone/mixer IF trigger phase-lock
- Dual-channel AnalogOut (Arbitrary Waveform Generator / Function Generator) with persistent output on close
- High-realism Demo simulator with internal AWG/Mixer virtual loopback
"""
from __future__ import annotations
import ctypes as C
from dataclasses import dataclass, field
import sys
import time
import numpy as np
from scipy import signal


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
acqmodeScanShift = 1
acqmodeScanScreen = 2
acqmodeRecord = 3
acqmodeOvers = 4
filterDecimate = 0
filterAverage = 1
filterMinMax = 2
filterAverageFit = 3

trigsrcNone = 0
trigsrcPC = 1
trigsrcDetectorAnalogIn = 2
trigsrcAnalogOut1 = 7
trigsrcAnalogOut2 = 8
trigsrcExternal1 = 11
trigsrcExternal2 = 12

trigtypeEdge = 0
trigtypePulse = 1
trigtypeTransition = 2
trigtypeWindow = 3
triglenLess = 0
triglenTimeout = 1
triglenMore = 2
DwfTriggerSlopeRise = 0
DwfTriggerSlopeFall = 1
DwfTriggerSlopeEither = 2

DwfAnalogCouplingDC = 0
DwfAnalogCouplingAC = 1

AnalogOutNodeCarrier = 0
AnalogOutNodeFM = 1
AnalogOutNodeAM = 2

DwfAnalogOutIdleDisable = 0
DwfAnalogOutIdleOffset = 1
DwfAnalogOutIdleInitial = 2
DwfAnalogOutIdleHold = 3

funcDC = 0
funcSine = 1
funcSquare = 2
funcTriangle = 3
funcRampUp = 4
funcRampDown = 5
funcNoise = 6
funcPulse = 7
funcTrapezium = 8
funcSinePower = 9
funcCustom = 30

WAVEFORM_FUNCTIONS = {
    "Sine": funcSine,
    "Square": funcSquare,
    "Triangle": funcTriangle,
    "RampUp": funcRampUp,
    "RampDown": funcRampDown,
    "Noise": funcNoise,
    "Pulse": funcPulse,
    "Trapezium": funcTrapezium,
    "SinePower": funcSinePower,
    "DC": funcDC,
    "Custom (自定义)": funcCustom,
}

AWG_IDLE_MODES = {
    "Offset (偏置电平)": DwfAnalogOutIdleOffset,
    "Disable (关闭/高阻)": DwfAnalogOutIdleDisable,
    "Initial (初始相位电平)": DwfAnalogOutIdleInitial,
    "Hold (保持末值)": DwfAnalogOutIdleHold,
}

AWG_TRIGGER_SOURCES = {
    "None (连续自由运行)": trigsrcNone,
    "PC (手动触发)": trigsrcPC,
    "Scope (示波器触发)": trigsrcDetectorAnalogIn,
    "W1 (通道1)": trigsrcAnalogOut1,
    "W2 (通道2)": trigsrcAnalogOut2,
    "Ext T1 (外部触发1)": trigsrcExternal1,
    "Ext T2 (外部触发2)": trigsrcExternal2,
}

TRIGGER_SLOPES = {
    "Rising": DwfTriggerSlopeRise,
    "Falling": DwfTriggerSlopeFall,
    "Either": DwfTriggerSlopeEither,
}

ACQ_MODES = {
    "Single (标准单帧)": acqmodeSingle,
    "ScanShift (滚动时基)": acqmodeScanShift,
    "ScanScreen (扫描刷新)": acqmodeScanScreen,
    "Record (长时流盘)": acqmodeRecord,
    "Overs (等效过采样)": acqmodeOvers,
}

HW_FILTERS = {
    "Auto (自动抗混叠)": -1,
    "Average (箱车平均)": filterAverage,
    "Decimate (直接抽样)": filterDecimate,
    "MinMax (峰值毛刺)": filterMinMax,
    "AverageFit (自适应)": filterAverageFit,
}

TRIGGER_TYPES = {
    "Edge (边沿)": trigtypeEdge,
    "Pulse (脉宽)": trigtypePulse,
    "Transition (斜率)": trigtypeTransition,
    "Window (窗口)": trigtypeWindow,
}

TRIGGER_LENGTH_CONDS = {
    "More (>)": triglenMore,
    "Less (<)": triglenLess,
    "Timeout (超时)": triglenTimeout,
}


@dataclass
class ChannelConfig:
    enabled: bool = True
    range_v: float = 5.0  # pk-pk range in V (AD3 has 5V and 50V hardware ranges, scaled in display)
    offset_v: float = 0.0
    coupling_ac: bool = False
    attenuation: float = 1.0
    hw_filter: str = "Auto (自动抗混叠)"
    lpf_cutoff_hz: float = 0.0  # 0.0 = full bandwidth; >0 = 4th-order zero-phase Butterworth LPF cutoff
    show_envelope: bool = False  # overlay extracted low-frequency IF envelope on composite mixer signals


@dataclass
class TriggerConfig:
    mode: str = "Auto"  # "Auto", "Normal", "Single", "None"
    source: str = "CH1"  # "CH1", "CH2", "W1", "W2", "Ext T1", "Ext T2", "PC"
    slope: str = "Rising"  # "Rising", "Falling", "Either"
    level_v: float = 0.0
    position_s: float = 0.0  # horizontal trigger offset in seconds
    coupling: str = "Auto"  # "Auto" (Smart Envelope+Carrier Lock), "HF Reject" (IF Envelope Lock), "DC" (Raw Edge)
    holdoff_s: float = 0.0
    trig_type: str = "Edge (边沿)"
    length_s: float = 1e-6
    length_cond: str = "More (>)"
    hysteresis_v: float = 0.025
    hw_filter: str = "Average"  # "Average" | "Decimate"


@dataclass
class ScopeConfig:
    sample_rate_hz: float = 50_000_000.0
    buffer_size: int = 32768
    timebase_s_per_div: float = 5e-6  # 5 us/div (50 us across 10 divs)
    auto_sample_rate: bool = True
    acq_mode: str = "Single (标准单帧)"
    record_length_s: float = 0.05
    hw_counter_enabled: bool = False
    channels: list[ChannelConfig] = field(
        default_factory=lambda: [ChannelConfig(True, 5.0), ChannelConfig(False, 5.0)]
    )
    trigger: TriggerConfig = field(default_factory=TriggerConfig)


@dataclass
class AwgChannelConfig:
    enabled: bool = False
    function: str = "Sine"
    frequency_hz: float = 1_000_000.0
    amplitude_v: float = 1.0
    offset_v: float = 0.0
    symmetry_pct: float = 50.0
    phase_deg: float = 0.0
    custom_samples: list[float] | None = None
    # AM / FM modulation nodes
    am_enabled: bool = False
    am_function: str = "Sine"
    am_frequency_hz: float = 1_000.0
    am_depth_pct: float = 50.0
    fm_enabled: bool = False
    fm_function: str = "Sine"
    fm_frequency_hz: float = 1_000.0
    fm_depth_pct: float = 10.0
    # Timing / Burst / Trigger / Idle / Sync / Limit
    run_s: float = 0.0  # 0.0 = continuous / infinite
    wait_s: float = 0.0
    repeat_count: int = 0  # 0 = infinite
    repeat_trigger: bool = False
    trigger_source: str = "None (连续自由运行)"
    trigger_slope: str = "Rising"
    idle_mode: str = "Offset (偏置电平)"
    sync_to_master: bool = False
    limitation_v: float = 0.0  # 0.0 = default hardware limit


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


def select_optimal_sample_rate(timebase_s_per_div: float, buffer_size: int = 32768, max_rate: float = 100_000_000.0) -> float:
    """Pick the highest standard sample rate whose capture window (buffer_size / fs) covers at least 1.5x the visible 10-div screen."""
    span_s = max(1e-7, float(timebase_s_per_div) * 10.0)
    # Require buffer duration >= max(1.6 * span_s, span_s + 20us) so trigger stabilization has ample margin
    required_duration = max(span_s * 1.6, span_s + 20e-6)
    max_fs_for_span = float(buffer_size) / required_duration
    standard_rates = [
        100_000_000.0,
        50_000_000.0,
        20_000_000.0,
        10_000_000.0,
        5_000_000.0,
        2_000_000.0,
        1_000_000.0,
        500_000.0,
        200_000.0,
        100_000.0,
    ]
    for rate in standard_rates:
        if rate <= max_rate and rate <= max_fs_for_span:
            return rate
    return standard_rates[-1]


class AD3Driver:
    """Thread-safe wrapper around Digilent WaveForms SDK (libdwf) or Demo simulator."""

    def __init__(self, demo: bool = False):
        self.demo = demo
        self.hdwf = C.c_int(hdwfNone)
        self.device_info: dict = {}
        self.scope_cfg = ScopeConfig()
        self.awg_cfg = [
            AwgChannelConfig(enabled=True, function="Sine", frequency_hz=1_000_000.0, amplitude_v=1.0),
            AwgChannelConfig(enabled=True, function="Sine", frequency_hz=1_100_000.0, amplitude_v=1.0),
        ]
        self._scope_dirty = True
        self._sim_phase = 0.0
        self._sos_cache: dict[tuple[int, float, float], np.ndarray] = {}

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

    def _get_lpf_sos(self, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray | None:
        nyq = 0.5 * fs
        if cutoff_hz <= 0.0 or cutoff_hz >= 0.92 * nyq:
            return None
        key = (order, round(cutoff_hz, 1), round(fs, 1))
        sos = self._sos_cache.get(key)
        if sos is None:
            sos = signal.butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
            if len(self._sos_cache) > 32:
                self._sos_cache.clear()
            self._sos_cache[key] = sos
        return sos

    def open(self, device_index: int = -1, config_index: int = 1) -> dict:
        """Open AD3 with configuration 1 (2x32768 scope buffer) by default."""
        if self.demo:
            self.device_info = {
                "name": "Analog Discovery 3 (Demo Simulator)",
                "serial": "SN:DEMO-AD3",
                "version": "Sim 3.25",
                "max_buffer": 32768,
                "max_rate": 100_000_000.0,
                "demo": True,
            }
            self._scope_dirty = True
            return self.device_info

        if _dwf is None:
            raise DwfError("未检测到 libdwf.so (WaveForms Runtime)")

        ver = C.create_string_buffer(32)
        _dwf.FDwfGetVersion(ver)
        # DwfParamOnClose = 0: keep AWG outputs running continuously even when application closes
        _dwf.FDwfParamSet(C.c_int(DwfParamOnClose), C.c_int(0))

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
            "max_rate": float(rate_max.value or 100_000_000.0),
            "demo": False,
        }
        self._scope_dirty = True
        self.apply_scope_config()
        return self.device_info

    def close(self, reset_awg: bool = False):
        if not self.demo and _dwf is not None and self.hdwf.value != hdwfNone:
            try:
                if reset_awg:
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
        acq_code = ACQ_MODES.get(cfg.acq_mode, acqmodeSingle)
        _dwf.FDwfAnalogInAcquisitionModeSet(self.hdwf, C.c_int(acq_code))
        if acq_code == acqmodeRecord and hasattr(_dwf, "FDwfAnalogInRecordLengthSet"):
            _dwf.FDwfAnalogInRecordLengthSet(self.hdwf, C.c_double(max(0.001, float(cfg.record_length_s))))

        # Use hardware boxcar averaging (filterAverage) whenever decimating below 100 MS/s to prevent aliasing
        auto_filter = filterAverage if cfg.sample_rate_hz < 99_000_000.0 else filterDecimate
        use_minmax = False
        for idx, ch in enumerate(cfg.channels[:2]):
            _dwf.FDwfAnalogInChannelEnableSet(self.hdwf, C.c_int(idx), C.c_int(1))
            if hasattr(_dwf, "FDwfAnalogInChannelAttenuationSet"):
                _dwf.FDwfAnalogInChannelAttenuationSet(self.hdwf, C.c_int(idx), C.c_double(float(ch.attenuation)))
            _dwf.FDwfAnalogInChannelRangeSet(self.hdwf, C.c_int(idx), C.c_double(ch.range_v * ch.attenuation))
            _dwf.FDwfAnalogInChannelOffsetSet(self.hdwf, C.c_int(idx), C.c_double(ch.offset_v))
            flt_code = HW_FILTERS.get(ch.hw_filter, -1)
            if flt_code < 0:
                flt_code = auto_filter
            if flt_code == filterMinMax:
                use_minmax = True
            _dwf.FDwfAnalogInChannelFilterSet(self.hdwf, C.c_int(idx), C.c_int(flt_code))
            coupling = DwfAnalogCouplingAC if ch.coupling_ac else DwfAnalogCouplingDC
            _dwf.FDwfAnalogInChannelCouplingSet(self.hdwf, C.c_int(idx), C.c_int(coupling))

        if use_minmax and hasattr(_dwf, "FDwfAnalogInNoiseSizeSet"):
            _dwf.FDwfAnalogInNoiseSizeSet(self.hdwf, C.c_int(min(cfg.buffer_size, 4096)))

        if cfg.hw_counter_enabled and hasattr(_dwf, "FDwfAnalogInCounterSet"):
            _dwf.FDwfAnalogInCounterSet(self.hdwf, C.c_double(0.2))
        elif hasattr(_dwf, "FDwfAnalogInCounterSet"):
            _dwf.FDwfAnalogInCounterSet(self.hdwf, C.c_double(0.0))

        trig = cfg.trigger
        if trig.mode == "None":
            _dwf.FDwfAnalogInTriggerSourceSet(self.hdwf, C.c_ubyte(trigsrcNone))
        else:
            src_map = {
                "CH1": trigsrcDetectorAnalogIn,
                "CH2": trigsrcDetectorAnalogIn,
                "W1": trigsrcAnalogOut1,
                "W2": trigsrcAnalogOut2,
                "Ext T1": trigsrcExternal1,
                "Ext T2": trigsrcExternal2,
                "PC": trigsrcPC,
            }
            _dwf.FDwfAnalogInTriggerSourceSet(self.hdwf, C.c_ubyte(src_map.get(trig.source, trigsrcDetectorAnalogIn)))
            ttype_code = TRIGGER_TYPES.get(trig.trig_type, trigtypeEdge)
            _dwf.FDwfAnalogInTriggerTypeSet(self.hdwf, C.c_int(ttype_code))
            trig_ch = 1 if trig.source == "CH2" else 0
            _dwf.FDwfAnalogInTriggerChannelSet(self.hdwf, C.c_int(trig_ch))
            if hasattr(_dwf, "FDwfAnalogInTriggerFilterSet"):
                t_flt = filterDecimate if trig.hw_filter == "Decimate" else filterAverage
                _dwf.FDwfAnalogInTriggerFilterSet(self.hdwf, C.c_int(t_flt))
            _dwf.FDwfAnalogInTriggerLevelSet(self.hdwf, C.c_double(trig.level_v))
            _dwf.FDwfAnalogInTriggerHysteresisSet(self.hdwf, C.c_double(max(0.0, float(trig.hysteresis_v))))
            slope_code = TRIGGER_SLOPES.get(trig.slope, DwfTriggerSlopeRise)
            _dwf.FDwfAnalogInTriggerConditionSet(self.hdwf, C.c_int(slope_code))
            if ttype_code in (trigtypePulse, trigtypeTransition):
                if hasattr(_dwf, "FDwfAnalogInTriggerLengthSet"):
                    _dwf.FDwfAnalogInTriggerLengthSet(self.hdwf, C.c_double(max(1e-9, float(trig.length_s))))
                if hasattr(_dwf, "FDwfAnalogInTriggerLengthConditionSet"):
                    lcond = TRIGGER_LENGTH_CONDS.get(trig.length_cond, triglenMore)
                    _dwf.FDwfAnalogInTriggerLengthConditionSet(self.hdwf, C.c_int(lcond))
            _dwf.FDwfAnalogInTriggerPositionSet(self.hdwf, C.c_double(trig.position_s))
            if hasattr(_dwf, "FDwfAnalogInTriggerHoldOffSet"):
                _dwf.FDwfAnalogInTriggerHoldOffSet(self.hdwf, C.c_double(max(0.0, float(trig.holdoff_s))))
            # Auto timeout: 0.05s for Auto mode, 0 for Normal/Single
            auto_to = 0.05 if trig.mode == "Auto" else 0.0
            _dwf.FDwfAnalogInTriggerAutoTimeoutSet(self.hdwf, C.c_double(auto_to))

        if _dwf.FDwfAnalogInConfigure(self.hdwf, C.c_int(1), C.c_int(1)) != 1:
            raise DwfError("配置示波器失败: " + self._last_error())
        self._scope_dirty = False

    def force_scope_trigger(self):
        if self.demo or not self.is_open():
            return
        if hasattr(_dwf, "FDwfAnalogInTriggerForce"):
            _dwf.FDwfAnalogInTriggerForce(self.hdwf)

    def configure_awg(self, ch_index: int, cfg: AwgChannelConfig):
        self.awg_cfg[ch_index] = cfg
        if self.demo or not self.is_open():
            return

        c_ch = C.c_int(ch_index)
        node_c = C.c_int(AnalogOutNodeCarrier)
        node_fm = C.c_int(AnalogOutNodeFM)
        node_am = C.c_int(AnalogOutNodeAM)
        if not cfg.enabled:
            _dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, c_ch, node_c, C.c_int(0))
            _dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, c_ch, node_fm, C.c_int(0))
            _dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, c_ch, node_am, C.c_int(0))
            _dwf.FDwfAnalogOutConfigure(self.hdwf, c_ch, C.c_int(0))
            return

        if hasattr(_dwf, "FDwfAnalogOutMasterSet"):
            master = 0 if (ch_index > 0 and cfg.sync_to_master) else -1
            _dwf.FDwfAnalogOutMasterSet(self.hdwf, c_ch, C.c_int(master))

        idle_id = AWG_IDLE_MODES.get(cfg.idle_mode, DwfAnalogOutIdleOffset)
        if hasattr(_dwf, "FDwfAnalogOutIdleSet"):
            _dwf.FDwfAnalogOutIdleSet(self.hdwf, c_ch, C.c_int(idle_id))

        if hasattr(_dwf, "FDwfAnalogOutRunSet"):
            _dwf.FDwfAnalogOutRunSet(self.hdwf, c_ch, C.c_double(max(0.0, cfg.run_s)))
        if hasattr(_dwf, "FDwfAnalogOutWaitSet"):
            _dwf.FDwfAnalogOutWaitSet(self.hdwf, c_ch, C.c_double(max(0.0, cfg.wait_s)))
        if hasattr(_dwf, "FDwfAnalogOutRepeatSet"):
            _dwf.FDwfAnalogOutRepeatSet(self.hdwf, c_ch, C.c_int(max(0, int(cfg.repeat_count))))
        if hasattr(_dwf, "FDwfAnalogOutRepeatTriggerSet"):
            _dwf.FDwfAnalogOutRepeatTriggerSet(self.hdwf, c_ch, C.c_int(1 if cfg.repeat_trigger else 0))

        trig_src = AWG_TRIGGER_SOURCES.get(cfg.trigger_source, trigsrcNone)
        if hasattr(_dwf, "FDwfAnalogOutTriggerSourceSet"):
            _dwf.FDwfAnalogOutTriggerSourceSet(self.hdwf, c_ch, C.c_ubyte(trig_src))
        if hasattr(_dwf, "FDwfAnalogOutTriggerSlopeSet"):
            _dwf.FDwfAnalogOutTriggerSlopeSet(
                self.hdwf, c_ch, C.c_int(TRIGGER_SLOPES.get(cfg.trigger_slope, DwfTriggerSlopeRise))
            )

        if hasattr(_dwf, "FDwfAnalogOutLimitationSet") and cfg.limitation_v > 0.0:
            _dwf.FDwfAnalogOutLimitationSet(self.hdwf, c_ch, C.c_double(cfg.limitation_v))

        func_id = WAVEFORM_FUNCTIONS.get(cfg.function, funcSine)
        _dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, c_ch, node_c, C.c_int(1))
        _dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf, c_ch, node_c, C.c_ubyte(func_id))
        _dwf.FDwfAnalogOutNodeFrequencySet(self.hdwf, c_ch, node_c, C.c_double(cfg.frequency_hz))
        _dwf.FDwfAnalogOutNodeAmplitudeSet(self.hdwf, c_ch, node_c, C.c_double(cfg.amplitude_v))
        _dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf, c_ch, node_c, C.c_double(cfg.offset_v))
        _dwf.FDwfAnalogOutNodeSymmetrySet(self.hdwf, c_ch, node_c, C.c_double(cfg.symmetry_pct))
        _dwf.FDwfAnalogOutNodePhaseSet(self.hdwf, c_ch, node_c, C.c_double(cfg.phase_deg))

        if func_id == funcCustom and cfg.custom_samples:
            samples = np.clip(np.asarray(cfg.custom_samples, dtype=np.float64), -1.0, 1.0)
            arr = (C.c_double * samples.size)(*samples.tolist())
            _dwf.FDwfAnalogOutNodeDataSet(self.hdwf, c_ch, node_c, arr, C.c_int(samples.size))

        if cfg.fm_enabled:
            fm_func = WAVEFORM_FUNCTIONS.get(cfg.fm_function, funcSine)
            _dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, c_ch, node_fm, C.c_int(1))
            _dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf, c_ch, node_fm, C.c_ubyte(fm_func))
            _dwf.FDwfAnalogOutNodeFrequencySet(self.hdwf, c_ch, node_fm, C.c_double(cfg.fm_frequency_hz))
            _dwf.FDwfAnalogOutNodeAmplitudeSet(self.hdwf, c_ch, node_fm, C.c_double(cfg.fm_depth_pct))
        else:
            _dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, c_ch, node_fm, C.c_int(0))

        if cfg.am_enabled:
            am_func = WAVEFORM_FUNCTIONS.get(cfg.am_function, funcSine)
            _dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, c_ch, node_am, C.c_int(1))
            _dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf, c_ch, node_am, C.c_ubyte(am_func))
            _dwf.FDwfAnalogOutNodeFrequencySet(self.hdwf, c_ch, node_am, C.c_double(cfg.am_frequency_hz))
            _dwf.FDwfAnalogOutNodeAmplitudeSet(self.hdwf, c_ch, node_am, C.c_double(cfg.am_depth_pct))
        else:
            _dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, c_ch, node_am, C.c_int(0))

        if _dwf.FDwfAnalogOutConfigure(self.hdwf, c_ch, C.c_int(1)) != 1:
            raise DwfError(f"配置信号发生器 W{ch_index + 1} 失败: " + self._last_error())

    def trigger_pc(self):
        if self.demo or not self.is_open():
            return
        if hasattr(_dwf, "FDwfDeviceTriggerPC"):
            _dwf.FDwfDeviceTriggerPC(self.hdwf)

    def _stabilize_and_extract_envelope(
        self,
        y_trig: np.ndarray,
        fs: float,
    ) -> tuple[float, np.ndarray | None, float]:
        """Compute sub-sample phase-locked trigger index and optional low-frequency IF envelope.

        Returns (trig_idx, lf_envelope_full_or_None, if_freq_hz).
        """
        n = y_trig.size
        center = float(n * 0.5)
        trig = self.scope_cfg.trigger
        if trig.mode == "None" or n < 64:
            return center, None, 0.0

        c_int = n // 2
        w = min(8192, n)
        s0 = max(0, c_int - w // 2)
        s1 = min(n, s0 + w)
        seg = y_trig[s0:s1]
        vpp = float(np.ptp(seg))
        if vpp < 0.01:
            return center, None, 0.0

        seg_zero = seg - float(np.mean(seg))
        spec = np.abs(np.fft.rfft(seg_zero * np.hanning(seg.size)))
        if spec.size > 3:
            spec[:3] = 0.0
        peak_bin = int(np.argmax(spec)) if spec.size > 3 else 0
        peak_amp = float(spec[peak_bin]) if peak_bin > 0 else 0.0

        # Detect if there is a lower-frequency IF / envelope component (e.g. 100 kHz alongside 2.1 MHz sum)
        sig_bins = np.where(spec[3:] >= peak_amp * 0.18)[0] + 3 if peak_amp > 0 else np.empty(0, dtype=int)
        low_bin = int(sig_bins[0]) if sig_bins.size > 0 else peak_bin
        df = fs / float(seg.size)
        f_low = low_bin * df
        f_peak = peak_bin * df
        is_multitone = (low_bin >= 3) and (f_peak > f_low * 2.2) and (f_low >= 50.0)

        lf_full: np.ndarray | None = None
        if_freq_hz = 0.0
        target_local = float(c_int - s0)

        if is_multitone or trig.coupling == "HF Reject":
            cutoff = min(0.42 * fs, max(f_low * 2.4, 250_000.0)) if is_multitone else min(0.42 * fs, 300_000.0)
            sos = self._get_lpf_sos(cutoff, fs, order=4)
            if sos is not None:
                lf_seg = signal.sosfiltfilt(sos, seg_zero).astype(np.float32)
                if is_multitone or self.scope_cfg.channels[0].show_envelope or self.scope_cfg.channels[1].show_envelope:
                    lf_full = signal.sosfiltfilt(sos, y_trig - float(np.mean(y_trig))).astype(np.float32)
                pad = max(16, seg.size // 8)
                lf_in = lf_seg[pad:-pad]
                if trig.slope == "Falling":
                    cross_lf = np.where((lf_in[:-1] >= 0.0) & (lf_in[1:] < 0.0))[0] + pad
                else:
                    cross_lf = np.where((lf_in[:-1] <= 0.0) & (lf_in[1:] > 0.0))[0] + pad
                if cross_lf.size >= 2:
                    diffs_lf = np.diff(cross_lf)
                    med_pts = float(np.median(diffs_lf[diffs_lf > 2])) if np.any(diffs_lf > 2) else 0.0
                    if med_pts > 0:
                        if_freq_hz = float(fs / med_pts)
                if cross_lf.size > 0:
                    best_lf = int(cross_lf[np.argmin(np.abs(cross_lf - (c_int - s0)))])
                    y0, y1 = float(lf_seg[best_lf]), float(lf_seg[best_lf + 1])
                    frac = (0.0 - y0) / (y1 - y0) if abs(y1 - y0) > 1e-9 else 0.0
                    target_local = float(best_lf) + frac
                    if trig.coupling == "HF Reject":
                        return float(s0) + target_local, lf_full, if_freq_hz

        # Snap to exact fractional level_v crossing of `seg` nearest `target_local`
        lvl = float(trig.level_v)
        # If level_v is outside the signal's range, clamp to midpoint so Auto mode still locks cleanly
        s_min, s_max = float(np.min(seg)), float(np.max(seg))
        if lvl <= s_min + 0.02 * vpp or lvl >= s_max - 0.02 * vpp:
            lvl = 0.5 * (s_min + s_max)

        if trig.slope == "Falling":
            cross_raw = np.where((seg[:-1] >= lvl) & (seg[1:] < lvl))[0]
        elif trig.slope == "Either":
            cross_raw = np.where(((seg[:-1] <= lvl) & (seg[1:] > lvl)) | ((seg[:-1] >= lvl) & (seg[1:] < lvl)))[0]
        else:
            cross_raw = np.where((seg[:-1] <= lvl) & (seg[1:] > lvl))[0]

        if cross_raw.size > 0:
            y0 = seg[cross_raw]
            y1 = seg[cross_raw + 1]
            denom = y1 - y0
            fracs = np.where(np.abs(denom) > 1e-9, (lvl - y0) / denom, 0.0)
            raw_pos = cross_raw.astype(np.float64) + fracs
            best_idx = int(np.argmin(np.abs(raw_pos - target_local)))
            return float(s0) + float(raw_pos[best_idx]), lf_full, if_freq_hz

        return float(s0) + target_local, lf_full, if_freq_hz

    def capture_frame(self, timeout_s: float = 0.25) -> dict | None:
        """Capture one frame of dual-channel scope data with sub-sample trigger stabilization."""
        if self.demo:
            raw_frame = self._simulate_frame()
        else:
            if not self.is_open():
                return None

            if self._scope_dirty:
                self.apply_scope_config()

            acq_code = ACQ_MODES.get(self.scope_cfg.acq_mode, acqmodeSingle)
            sts = C.c_byte()
            t_start = time.monotonic()

            actual_rate = C.c_double()
            _dwf.FDwfAnalogInFrequencyGet(self.hdwf, C.byref(actual_rate))
            fs = float(actual_rate.value or self.scope_cfg.sample_rate_hz)

            ch_noise: list[tuple[np.ndarray, np.ndarray] | None] = [None, None]
            auto_trig = False
            hw_freq_hz: float | None = None

            if acq_code == acqmodeRecord:
                target_n = int(min(500_000, max(1024, round(self.scope_cfg.record_length_s * fs))))
                chunks: list[list[np.ndarray]] = [[], []]
                collected = 0
                c_avail, c_lost, c_corr = C.c_int(), C.c_int(), C.c_int()
                while collected < target_n:
                    if _dwf.FDwfAnalogInStatus(self.hdwf, C.c_int(1), C.byref(sts)) != 1:
                        raise DwfError("读取流盘状态失败: " + self._last_error())
                    _dwf.FDwfAnalogInStatusRecord(self.hdwf, C.byref(c_avail), C.byref(c_lost), C.byref(c_corr))
                    avail = int(c_avail.value)
                    if avail > 0:
                        take = min(avail, target_n - collected)
                        for idx, ch_cfg in enumerate(self.scope_cfg.channels[:2]):
                            if ch_cfg.enabled:
                                buf = (C.c_double * take)()
                                _dwf.FDwfAnalogInStatusData(self.hdwf, C.c_int(idx), buf, C.c_int(take))
                                arr = np.ctypeslib.as_array(buf).astype(np.float32)
                                if not hasattr(_dwf, "FDwfAnalogInChannelAttenuationSet"):
                                    arr = arr * float(ch_cfg.attenuation)
                                chunks[idx].append(arr)
                        collected += take
                    if sts.value == DwfStateDone and avail == 0:
                        break
                    if time.monotonic() - t_start > max(timeout_s, self.scope_cfg.record_length_s + 0.5):
                        break
                    if avail == 0:
                        time.sleep(0.001)
                n = max(collected, 1)
                ch_data = []
                for idx, ch_cfg in enumerate(self.scope_cfg.channels[:2]):
                    if ch_cfg.enabled and chunks[idx]:
                        ch_data.append(np.concatenate(chunks[idx]))
                    else:
                        ch_data.append(np.zeros(n, dtype=np.float32))
                if self.scope_cfg.trigger.mode != "Single":
                    _dwf.FDwfAnalogInConfigure(self.hdwf, C.c_int(0), C.c_int(1))
            elif acq_code in (acqmodeScanShift, acqmodeScanScreen):
                if _dwf.FDwfAnalogInStatus(self.hdwf, C.c_int(1), C.byref(sts)) != 1:
                    raise DwfError("读取扫描状态失败: " + self._last_error())
                valid_c = C.c_int()
                _dwf.FDwfAnalogInStatusSamplesValid(self.hdwf, C.byref(valid_c))
                actual_n = C.c_int()
                _dwf.FDwfAnalogInBufferSizeGet(self.hdwf, C.byref(actual_n))
                buf_n = int(actual_n.value or self.scope_cfg.buffer_size)
                n = max(1, min(buf_n, int(valid_c.value or buf_n)))
                ch_data = []
                for idx, ch_cfg in enumerate(self.scope_cfg.channels[:2]):
                    if ch_cfg.enabled:
                        buf = (C.c_double * n)()
                        _dwf.FDwfAnalogInStatusData(self.hdwf, C.c_int(idx), buf, C.c_int(n))
                        arr = np.ctypeslib.as_array(buf).astype(np.float32)
                        if not hasattr(_dwf, "FDwfAnalogInChannelAttenuationSet"):
                            arr = arr * float(ch_cfg.attenuation)
                    else:
                        arr = np.zeros(n, dtype=np.float32)
                    ch_data.append(arr)
            else:
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

                actual_n = C.c_int()
                _dwf.FDwfAnalogInBufferSizeGet(self.hdwf, C.byref(actual_n))
                n = int(actual_n.value or self.scope_cfg.buffer_size)

                ch_data = []
                for idx, ch_cfg in enumerate(self.scope_cfg.channels[:2]):
                    if ch_cfg.enabled:
                        buf = (C.c_double * n)()
                        _dwf.FDwfAnalogInStatusData(self.hdwf, C.c_int(idx), buf, C.c_int(n))
                        arr = np.ctypeslib.as_array(buf).astype(np.float32)
                        if not hasattr(_dwf, "FDwfAnalogInChannelAttenuationSet"):
                            arr = arr * float(ch_cfg.attenuation)
                        if HW_FILTERS.get(ch_cfg.hw_filter, -1) == filterMinMax and hasattr(_dwf, "FDwfAnalogInStatusNoise"):
                            n_noise = min(n, 4096)
                            b_min = (C.c_double * n_noise)()
                            b_max = (C.c_double * n_noise)()
                            if _dwf.FDwfAnalogInStatusNoise(self.hdwf, C.c_int(idx), b_min, b_max, C.c_int(n_noise)) == 1:
                                ch_noise[idx] = (
                                    np.ctypeslib.as_array(b_min).astype(np.float32),
                                    np.ctypeslib.as_array(b_max).astype(np.float32),
                                )
                    else:
                        arr = np.zeros(n, dtype=np.float32)
                    ch_data.append(arr)

                # Re-arm next single acquisition immediately for continuous streaming
                if self.scope_cfg.trigger.mode != "Single":
                    _dwf.FDwfAnalogInConfigure(self.hdwf, C.c_int(0), C.c_int(1))

            if hasattr(_dwf, "FDwfAnalogInStatusAutoTriggered"):
                c_auto = C.c_int()
                if _dwf.FDwfAnalogInStatusAutoTriggered(self.hdwf, C.byref(c_auto)) == 1:
                    auto_trig = bool(c_auto.value)

            if self.scope_cfg.hw_counter_enabled and hasattr(_dwf, "FDwfAnalogInCounterStatus"):
                c_cnt, c_freq, c_tick = C.c_double(), C.c_double(), C.c_int()
                if _dwf.FDwfAnalogInCounterStatus(self.hdwf, C.byref(c_cnt), C.byref(c_freq), C.byref(c_tick)) == 1:
                    if c_freq.value > 0.0:
                        hw_freq_hz = float(c_freq.value)

            raw_frame = {
                "ch1": ch_data[0],
                "ch2": ch_data[1],
                "ch1_noise": ch_noise[0],
                "ch2_noise": ch_noise[1],
                "fs": fs,
                "n": n,
                "triggered": sts.value == DwfStateDone,
                "auto_triggered": auto_trig,
                "hw_freq_hz": hw_freq_hz,
                "timestamp": time.time(),
            }

        fs = raw_frame["fs"]
        n = raw_frame["n"]
        ch_arrays = [raw_frame["ch1"], raw_frame["ch2"]]

        # Apply optional per-channel digital low-pass filter (e.g. for IF extraction)
        for idx, ch_cfg in enumerate(self.scope_cfg.channels[:2]):
            if ch_cfg.enabled and ch_cfg.lpf_cutoff_hz > 0.0:
                sos = self._get_lpf_sos(ch_cfg.lpf_cutoff_hz, fs, order=4)
                if sos is not None and ch_arrays[idx].size >= 32:
                    ch_arrays[idx] = signal.sosfiltfilt(sos, ch_arrays[idx]).astype(np.float32)

        trig_ch_idx = 1 if self.scope_cfg.trigger.source == "CH2" else 0
        trig_idx, lf_env, if_freq_hz = self._stabilize_and_extract_envelope(ch_arrays[trig_ch_idx], fs)
        t = (np.arange(n, dtype=np.float32) - np.float32(trig_idx)) / np.float32(fs) + np.float32(
            self.scope_cfg.trigger.position_s
        )

        raw_frame["t"] = t
        raw_frame["ch1"] = ch_arrays[0]
        raw_frame["ch2"] = ch_arrays[1]
        raw_frame["ch1_env"] = lf_env if trig_ch_idx == 0 else None
        raw_frame["ch2_env"] = lf_env if trig_ch_idx == 1 else None
        raw_frame["if_freq_hz"] = if_freq_hz
        raw_frame.setdefault("auto_triggered", False)
        raw_frame.setdefault("ch1_noise", None)
        raw_frame.setdefault("ch2_noise", None)
        if self.demo and self.scope_cfg.hw_counter_enabled:
            awg0 = self.awg_cfg[0]
            raw_frame["hw_freq_hz"] = float(if_freq_hz if if_freq_hz > 0 else (awg0.frequency_hz if awg0.enabled else 1_000.0))
        else:
            raw_frame.setdefault("hw_freq_hz", None)
        if self.demo:
            for idx, ch_cfg in enumerate(self.scope_cfg.channels[:2]):
                if ch_cfg.enabled and HW_FILTERS.get(ch_cfg.hw_filter, -1) == filterMinMax:
                    y_arr = ch_arrays[idx]
                    raw_frame[f"ch{idx+1}_noise"] = (y_arr - np.float32(0.015), y_arr + np.float32(0.015))
        return raw_frame

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

        two_pi = np.float32(2.0 * np.pi)
        w1, w2 = self.awg_cfg[0], self.awg_cfg[1]

        # If both W1 and W2 are enabled at different frequencies (>100 kHz), simulate realistic RF mixer IF output on CH1
        if w1.enabled and w2.enabled and abs(w1.frequency_hz - w2.frequency_hz) >= 1_000.0 and min(w1.frequency_hz, w2.frequency_hz) >= 250_000.0:
            f_diff = np.float32(abs(w2.frequency_hz - w1.frequency_hz))
            f_sum = np.float32(w1.frequency_hz + w2.frequency_hz)
            scale = np.float32(0.5 * (w1.amplitude_v + w2.amplitude_v))
            ph = np.float32(self._sim_phase) if cfg.trigger.mode == "None" else np.float32(0.0)
            ch1 = (
                scale * np.float32(0.40) * np.sin(two_pi * f_diff * t + ph)
                + scale * np.float32(0.56) * np.sin(two_pi * f_sum * t + np.float32(0.35) + ph)
                + scale * np.float32(0.11) * np.sin(two_pi * (np.float32(3.0 * w1.frequency_hz + w2.frequency_hz)) * t)
                + rng.standard_normal(n, dtype=np.float32) * np.float32(0.004)
            )
            ch2 = scale * np.sin(two_pi * np.float32(w2.frequency_hz) * t + ph) + rng.standard_normal(n, dtype=np.float32) * np.float32(0.004)
            return {
                "t": t,
                "ch1": ch1.astype(np.float32),
                "ch2": ch2.astype(np.float32),
                "fs": fs,
                "n": n,
                "triggered": True,
                "timestamp": time.time(),
            }

        ch_list = []
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
