"""Background Qt worker managing AD3 hardware acquisition, AWG control, and GPU FFT pipeline."""
from __future__ import annotations
import time
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from .driver import AD3Driver, AwgChannelConfig, ScopeConfig, DwfError
from .gpu_fft import GpuFftEngine
from .measurements import measure_channel


class InstrumentWorker(QObject):
    connected = Signal(object)
    connection_changed = Signal(bool)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    frame_ready = Signal(object)

    def __init__(self, demo: bool = False):
        super().__init__()
        self.demo = demo
        self.driver = AD3Driver(demo=demo)
        self.fft_engine = GpuFftEngine("auto")
        self.running = False
        self.fft_enabled = True
        self.fft_window = "FlatTop"
        self.fft_avg_count = 1
        self.fft_n_pad = 0
        self.fft_zoom_enabled = False
        self.fft_zoom_stop_hz = 5_000_000.0
        self.fft_trace_mode = "live"  # "live" | "max_hold" | "min_hold"
        self.fft_goertzel_enabled = False
        self.fft_goertzel_freq_hz = 200_000.0
        self._fft_accum: dict[str, np.ndarray] = {}
        self._fft_hold: dict[str, np.ndarray] = {}
        self._last_metrics: dict[str, dict] = {}
        self._frame_seq = 0
        self.timer: QTimer | None = None
        self.fps = 0.0
        self._last_frame_ts = time.perf_counter()

    def _ensure_timer(self):
        if self.timer is None:
            self.timer = QTimer(self)
            self.timer.setInterval(50)  # ~20 Hz smooth, low-CPU oscilloscope refresh
            self.timer.timeout.connect(self._poll_frame)

    @Slot()
    def connect_device(self):
        self._ensure_timer()
        try:
            self.status_changed.emit("正在连接 AD3 设备…")
            info = self.driver.open()
            info["fft_backend"] = self.fft_engine.active_backend
            info["fft_device"] = self.fft_engine.device_name
            info["fft_backends"] = self.fft_engine.available_backends()
            self.connection_changed.emit(True)
            self.connected.emit(info)
            self.status_changed.emit(f"已连接 · {info['name']} ({info['serial']})")
            self.start_acquisition()
        except Exception as exc:
            self.connection_changed.emit(False)
            self.error_occurred.emit(str(exc))
            self.status_changed.emit("未连接 · " + str(exc))

    @Slot()
    def disconnect_device(self):
        self.stop_acquisition()
        self.driver.close(reset_awg=True)
        self.connection_changed.emit(False)
        self.status_changed.emit("已断开连接")

    @Slot(bool)
    def set_demo_mode(self, demo: bool):
        was_running = self.running
        self.disconnect_device()
        self.demo = bool(demo)
        self.driver = AD3Driver(demo=self.demo)
        self.connect_device()
        if not was_running:
            self.stop_acquisition()

    @Slot()
    def start_acquisition(self):
        self._ensure_timer()
        if not self.driver.is_open():
            self.connect_device()
            if not self.driver.is_open():
                return
        self.running = True
        self.driver._scope_dirty = True
        self.timer.start()
        self.status_changed.emit("采集中 · 实时稳相流")

    @Slot()
    def stop_acquisition(self):
        self.running = False
        if self.timer is not None:
            self.timer.stop()
        if self.driver.is_open():
            self.status_changed.emit("已暂停 (STOP)")

    @Slot()
    def single_acquisition(self):
        self.stop_acquisition()
        if not self.driver.is_open():
            return
        self.driver.scope_cfg.trigger.mode = "Single"
        self.driver._scope_dirty = True
        self.status_changed.emit("单次触发等待中 (SINGLE)…")
        self._poll_frame(single_shot=True)

    @Slot(object)
    def update_scope_config(self, cfg: ScopeConfig):
        self.driver.scope_cfg = cfg
        self.driver._scope_dirty = True
        self._fft_accum.clear()
        if self.driver.is_open() and not self.running:
            try:
                self.driver.apply_scope_config()
            except Exception as exc:
                self.error_occurred.emit(str(exc))

    @Slot(int, object)
    def update_awg_config(self, channel_idx: int, cfg: AwgChannelConfig):
        try:
            self.driver.configure_awg(channel_idx, cfg)
            state_txt = f"{cfg.function} {cfg.frequency_hz/1e3:.4g} kHz, {cfg.amplitude_v:.2f} V" if cfg.enabled else "已关闭"
            self.status_changed.emit(f"W{channel_idx + 1} 已更新 · {state_txt}")
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    @Slot()
    def trigger_pc(self):
        try:
            self.driver.trigger_pc()
            self.status_changed.emit("已发送 PC 手动触发脉冲")
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    @Slot()
    def force_scope_trigger(self):
        try:
            self.driver.force_scope_trigger()
            self.status_changed.emit("已发送示波器强制触发")
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    @Slot(dict)
    def update_fft_settings(self, settings: dict):
        if "enabled" in settings:
            self.fft_enabled = bool(settings["enabled"])
        if "window" in settings:
            self.fft_window = str(settings["window"])
            self._fft_accum.clear()
        if "average" in settings:
            self.fft_avg_count = max(1, int(settings["average"]))
            self._fft_accum.clear()
        if "backend" in settings:
            self.fft_engine.set_backend(str(settings["backend"]))
            self._fft_accum.clear()
        if "n_fft" in settings:
            self.fft_n_pad = int(settings["n_fft"])
            self._fft_accum.clear()
            self._fft_hold.clear()
        if "zoom_enabled" in settings:
            self.fft_zoom_enabled = bool(settings["zoom_enabled"])
            self._fft_accum.clear()
            self._fft_hold.clear()
        if "zoom_stop_hz" in settings:
            self.fft_zoom_stop_hz = float(settings["zoom_stop_hz"])
            if self.fft_zoom_enabled:
                self._fft_accum.clear()
                self._fft_hold.clear()
        if "trace_mode" in settings:
            self.fft_trace_mode = str(settings["trace_mode"])
            self._fft_hold.clear()
        if "goertzel_enabled" in settings:
            self.fft_goertzel_enabled = bool(settings["goertzel_enabled"])
        if "goertzel_freq_hz" in settings:
            self.fft_goertzel_freq_hz = float(settings["goertzel_freq_hz"])
        if settings.get("clear_hold"):
            self._fft_hold.clear()

    @Slot()
    def _poll_frame(self, single_shot: bool = False):
        if not self.driver.is_open():
            return
        if not self.running and not single_shot:
            return
        try:
            raw = self.driver.capture_frame(timeout_s=0.15)
            if raw is None:
                return

            now = time.perf_counter()
            dt = max(1e-4, now - self._last_frame_ts)
            self._last_frame_ts = now
            inst_fps = 1.0 / dt
            self.fps = inst_fps if self.fps == 0.0 else (0.85 * self.fps + 0.15 * inst_fps)

            fs = raw["fs"]
            t = raw["t"]
            ch1 = raw["ch1"]
            ch2 = raw["ch2"]
            ch1_env = raw.get("ch1_env")
            ch2_env = raw.get("ch2_env")
            if_freq_hz = raw.get("if_freq_hz", 0.0)
            self._frame_seq += 1
            do_telemetry = single_shot or (self._frame_seq % 3 == 1)

            if do_telemetry:
                m1 = (
                    measure_channel(t, ch1, fs, env=ch1_env, if_freq_hz=if_freq_hz if ch1_env is not None else 0.0)
                    if self.driver.scope_cfg.channels[0].enabled
                    else {}
                )
                m2 = (
                    measure_channel(t, ch2, fs, env=ch2_env, if_freq_hz=if_freq_hz if ch2_env is not None else 0.0)
                    if self.driver.scope_cfg.channels[1].enabled
                    else {}
                )
                self._last_meas = {"ch1": m1, "ch2": m2}
            else:
                self._last_meas = getattr(self, "_last_meas", {"ch1": {}, "ch2": {}})

            fft_res = {}
            if self.fft_enabled:
                pad_n = self.fft_n_pad if self.fft_n_pad > 0 else None
                for ch_name, idx, arr in [("ch1", 0, ch1), ("ch2", 1, ch2)]:
                    if self.driver.scope_cfg.channels[idx].enabled:
                        if self.fft_zoom_enabled and self.fft_zoom_stop_hz > 100.0:
                            res = self.fft_engine.compute_zoom_fft(
                                arr,
                                fs,
                                f_start=0.0,
                                f_stop=min(fs * 0.5, self.fft_zoom_stop_hz),
                                m_bins=2048,
                                window_name=self.fft_window,
                                compute_metrics=False,
                            )
                        else:
                            res = self.fft_engine.compute(
                                arr,
                                fs,
                                window_name=self.fft_window,
                                pad_to=pad_n,
                                compute_metrics=False,
                            )
                        if res:
                            if self.fft_avg_count > 1:
                                alpha = np.float32(1.0 / float(self.fft_avg_count))
                                prev = self._fft_accum.get(ch_name)
                                if prev is None or prev.shape != res["dbv"].shape:
                                    self._fft_accum[ch_name] = res["dbv"].copy()
                                else:
                                    self._fft_accum[ch_name] = (np.float32(1.0) - alpha) * prev + alpha * res["dbv"]
                                res["dbv"] = self._fft_accum[ch_name]
                            if self.fft_trace_mode in ("max_hold", "min_hold"):
                                prev_h = self._fft_hold.get(ch_name)
                                if prev_h is None or prev_h.shape != res["dbv"].shape:
                                    self._fft_hold[ch_name] = res["dbv"].copy()
                                elif self.fft_trace_mode == "max_hold":
                                    np.maximum(prev_h, res["dbv"], out=prev_h)
                                else:
                                    np.minimum(prev_h, res["dbv"], out=prev_h)
                                res["dbv"] = self._fft_hold[ch_name]
                            # Compute peak & spectral metrics on the exact displayed (averaged/held) dbv array
                            vrms_disp = np.power(10.0, res["dbv"] / 20.0).astype(np.float32)
                            res["metrics"] = self.fft_engine._compute_spectral_metrics(res["freqs"], res["dbv"], vrms_disp)
                            self._last_metrics[ch_name] = res["metrics"]
                            if self.fft_goertzel_enabled and do_telemetry:
                                res["goertzel"] = self.fft_engine.compute_goertzel(
                                    arr,
                                    fs,
                                    self.fft_goertzel_freq_hz,
                                    window_name=self.fft_window,
                                )
                        fft_res[ch_name] = res

            raw["meas"] = self._last_meas
            raw["fft"] = fft_res
            raw["fps"] = self.fps
            raw["telemetry_updated"] = do_telemetry
            self.frame_ready.emit(raw)

            if single_shot or self.driver.scope_cfg.trigger.mode == "Single":
                self.stop_acquisition()
                self.status_changed.emit("单次触发采集完成 (SINGLE DONE)")
        except DwfError as exc:
            self.stop_acquisition()
            self.error_occurred.emit(str(exc))
            self.status_changed.emit("硬件采集异常 · " + str(exc))
        except Exception as exc:
            self.error_occurred.emit(str(exc))
