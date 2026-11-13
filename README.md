# WaveDesk (`~/enterprises/wavedesk`)

Next-generation GPU-accelerated Oscilloscope, Real-Time FFT Spectrum Analyzer, and Waveform Generator for **Digilent Analog Discovery 3 (AD3)**.

## Key Features (v0.1.0)

1. **Cross-Vendor GPU FFT Engine (`src/wavedesk/gpu_fft.py`)**
   - **OpenCL GPU Backend**: Built-in Radix-2 Cooley-Tukey OpenCL kernel + fused windowing & dBV/phase calculation. Works out of the box on **AMD** (e.g., Radeon 780M `gfx1103`), **Intel**, and **NVIDIA** GPUs without proprietary SDK dependencies.
   - **CUDA Backend**: Optional `CuPy` support for NVIDIA CUDA / AMD ROCm.
   - **CPU Fallback**: Multi-threaded SIMD `scipy.fft` fallback.
   - **Automated Spectral Metrics**: Real-time sub-bin parabolic peak frequency, amplitude (`dBV` / `Vrms`), `THD` (dB & %), `SNR`, and `SFDR`.

2. **Next-Gen OpenGL SDF & DPO Waveform Engine (`src/wavedesk/gpu_view.py`)**
   - **GPU SDF Thick-Ribbon Anti-Aliasing**: Computes screen-space normals and renders distance-field smoothed thick curves with outer bloom in a single draw call (overcoming OpenGL Core Profile's `1.0px` `glLineWidth` limit and multi-pass redraw overhead).
   - **True Digital Phosphor Oscilloscope (DPO) Thermal Grading**: Ping-pong offscreen FBO persistence with a thermal color LUT mapping hit density from cool base color to hot white core.
   - **Peak-Detect Min/Max Envelope Decimation**: Preserves sub-pixel high-frequency glitches when rendering deep 32k buffers.

3. **AD3 Hardware SDK Driver (`src/wavedesk/driver.py`)**
   - Direct `ctypes` binding to `libdwf.so` (`WaveForms Runtime`).
   - Opens AD3 in Configuration `1` (`2 x 32,768` deep scope buffer at up to `125 MS/s`).
   - Dual-channel Oscilloscope (`CH1`/`CH2`) + Dual-channel Function Generator (`W1`/`W2`) + built-in `--demo` simulator.

## Quick Start

```bash
cd ~/enterprises/wavedesk
./run.sh          # Connect to real AD3 hardware
./run.sh --demo   # Run with built-in signal simulator
uv run pytest     # Run unit & GPU verification tests
```
