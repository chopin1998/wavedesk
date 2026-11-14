"""Next-Gen OpenGL Waveform & Spectrum Rendering Engine (Beyond SpectraDesk).

Upgrades over SpectraDesk's `gpu_curve.py`:
1. True GPU SDF Thick-Segment Anti-Aliasing (instanced/expanded triangle strip in vertex+fragment shader,
   avoiding OpenGL core profile's 1px `glLineWidth` limit and 3-pass CPU redraw loops) with twist-free vertex normals.
2. True Digital Phosphor Oscilloscope (DPO) persistence with thermal/intensity color grading
   (accumulating hit density in an offscreen FBO and mapping intensity through a GPU thermal LUT).
3. Hardware Peak-Detect Min/Max Envelope decimation when sample count >> screen width, ensuring
   sub-pixel high-frequency glitches are never lost during zooming out.
"""
from __future__ import annotations
import ctypes as C
import math
import time
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtOpenGL
from pyqtgraph.Qt import OpenGLHelpers, OpenGLConstants as GL
from pyqtgraph.graphicsItems.PlotCurveItem import OpenGLState

try:
    _native_gl = C.CDLL("libGL.so.1")
    _native_gl.glGetIntegerv.argtypes = [C.c_uint, C.POINTER(C.c_int)]
    _native_gl.glGetIntegerv.restype = None
    _native_gl.glReadPixels.argtypes = [
        C.c_int,
        C.c_int,
        C.c_int,
        C.c_int,
        C.c_uint,
        C.c_uint,
        C.c_void_p,
    ]
    _native_gl.glReadPixels.restype = None
except OSError:
    _native_gl = None


def framebuffer_binding() -> int:
    if _native_gl is None:
        return 0
    value = C.c_int()
    _native_gl.glGetIntegerv(0x8CA6, C.byref(value))
    return value.value


GL_FRAMEBUFFER = 0x8D40
GL_TEXTURE0 = 0x84C0
GL_TEXTURE_2D = 0x0DE1

# Vertex shader for SDF Anti-Aliased Thick Segments
# Each vertex in the strip carries: (x, y, nx, ny, edge_coord)
# where (nx, ny) is screen-space normal offset and edge_coord is in [-1, 1] for SDF feathering
SDF_LINE_VERT = """#version 140
in vec4 a_pos_normal;
in float a_edge;
uniform mat4 u_mvp;
uniform vec2 u_viewport;
uniform float u_half_width;
out float v_edge;
out float v_y;
void main() {
    vec4 clip = u_mvp * vec4(a_pos_normal.xy, 0.0, 1.0);
    // Note: u_mvp uses QWidget top-down coordinates (Y inverted in NDC),
    // so negate ny to keep screen-space normal perpendicular to the tangent!
    vec2 offset_px = vec2(a_pos_normal.z, -a_pos_normal.w) * u_half_width;
    clip.xy += (offset_px / u_viewport) * 2.0 * clip.w;
    gl_Position = clip;
    v_edge = a_edge;
    v_y = a_pos_normal.y;
}
"""

SDF_LINE_FRAG = """#version 140
in float v_edge;
in float v_y;
uniform vec4 u_color;
uniform int u_glow_mode;
out vec4 fragColor;
void main() {
    float d = abs(v_edge);
    // Smooth SDF core + subtle outer bloom
    float core = 1.0 - smoothstep(0.0, 0.75, d);
    if (u_glow_mode == 1) {
        float halo = exp(-6.0 * d * d);
        vec3 col = mix(u_color.rgb, vec3(1.0, 1.0, 1.0), core * 0.25);
        float alpha = u_color.a * clamp(core * 0.94 + halo * 0.22, 0.0, 1.0);
        fragColor = vec4(col, alpha);
    } else {
        fragColor = vec4(u_color.rgb, u_color.a * core);
    }
}
"""

QUAD_VERT = """#version 140
in vec4 a_position;
out vec2 uv;
void main() {
    gl_Position = a_position;
    uv = a_position.xy * 0.5 + 0.5;
}
"""

# Decay pass + DPO thermal grading shader
QUAD_FRAG = """#version 140
uniform sampler2D u_texture;
uniform float u_decay;
uniform int u_thermal_lut;
uniform vec3 u_base_color;
in vec2 uv;
out vec4 fragColor;

vec3 dpo_thermal(float intensity, vec3 base) {
    // Keep the channel's signature color prominent, with a bright core on high persistence density
    float t = clamp(intensity, 0.0, 1.0);
    vec3 low = base * 0.65;
    vec3 mid = base;
    vec3 high = mix(base, vec3(1.0, 1.0, 1.0), 0.45);
    if (t < 0.60) {
        return mix(low, mid, t / 0.60);
    }
    return mix(mid, high, (t - 0.60) / 0.40);
}

void main() {
    vec4 tex = texture(u_texture, uv) * u_decay;
    if (u_thermal_lut == 1) {
        float lum = clamp(tex.a, 0.0, 1.0);
        if (lum < 0.01) {
            fragColor = vec4(0.0);
        } else {
            vec3 graded = dpo_thermal(lum, u_base_color);
            float out_a = clamp(lum * 1.10, 0.0, 0.95);
            fragColor = vec4(graded * out_a, out_a);
        }
    } else {
        fragColor = min(tex, vec4(1.0));
    }
}
"""


def _adapt_glsl(context, vert: str, frag: str) -> tuple[str, str]:
    if not context.isOpenGLES() and context.format().version() >= (3, 1):
        return vert, frag
    es = context.isOpenGLES()
    version = "#version 100\nprecision highp float;\nprecision highp int;" if es else "#version 120"
    v = (
        vert.replace("#version 140", version)
        .replace("in vec4 ", "attribute vec4 ")
        .replace("in float ", "attribute float ")
        .replace("out float ", "varying float ")
        .replace("out vec2 ", "varying vec2 ")
    )
    f = (
        frag.replace("#version 140", version)
        .replace("in float ", "varying float ")
        .replace("in vec2 ", "varying vec2 ")
        .replace("out vec4 fragColor;", "")
        .replace("fragColor =", "gl_FragColor =")
        .replace("texture(", "texture2D(")
    )
    return v, f


def compile_program(widget, key: str, vert: str, frag: str, attrs: list[tuple[int, str]]):
    program = widget.retrieveProgram(key)
    if program is None:
        program = QtOpenGL.QOpenGLShaderProgram()
        for kind, src in [
            (QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex, vert),
            (QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment, frag),
        ]:
            if not program.addShaderFromSourceCode(kind, src):
                raise RuntimeError(program.log())
        for idx, name in attrs:
            program.bindAttributeLocation(name, idx)
        if not program.link():
            raise RuntimeError(program.log())
        widget.storeProgram(key, program)
    return program


def build_sdf_ribbon_vertices(
    x: np.ndarray,
    y: np.ndarray,
    view_range: tuple[tuple[float, float], tuple[float, float]],
    viewport_px: tuple[float, float],
    max_segments: int = 2048,
) -> np.ndarray:
    """Build twist-free GPU triangle-strip ribbon with screen-space normals for SDF anti-aliasing & peak envelope."""
    n = x.size
    if n < 2:
        return np.empty((0, 5), dtype=np.float32)

    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)

    (x0, x1), (y0, y1) = view_range
    if x1 < x0:
        x0, x1 = x1, x0

    # Clip to visible horizontal view range first so zoomed-in views only process visible samples
    if x[0] < x0 or x[-1] > x1:
        i0 = max(0, int(np.searchsorted(x, x0, side="left")) - 2)
        i1 = min(n, int(np.searchsorted(x, x1, side="right")) + 2)
        if i1 - i0 >= 2:
            x = x[i0:i1]
            y = y[i0:i1]
            n = x.size

    # Hardware-style Min/Max Peak-Detect decimation when sample count exceeds 2x pixel width
    if n > max_segments * 2:
        step = max(2, (n + max_segments - 1) // max_segments)
        buckets = n // step
        usable = buckets * step
        xr = x[:usable].reshape(buckets, step)
        yr = y[:usable].reshape(buckets, step)
        idx_min = np.argmin(yr, axis=1)
        idx_max = np.argmax(yr, axis=1)
        rows = np.arange(buckets)
        first_min = idx_min <= idx_max

        x_min = xr[rows, idx_min]
        y_min = yr[rows, idx_min]
        x_max = xr[rows, idx_max]
        y_max = yr[rows, idx_max]

        rem = n - usable
        extra = 2 if rem >= 2 else (1 if rem == 1 else 0)
        x_out = np.empty(buckets * 2 + extra, dtype=np.float32)
        y_out = np.empty(buckets * 2 + extra, dtype=np.float32)
        x_out[: buckets * 2 : 2] = np.where(first_min, x_min, x_max)
        y_out[: buckets * 2 : 2] = np.where(first_min, y_min, y_max)
        x_out[1 : buckets * 2 : 2] = np.where(first_min, x_max, x_min)
        y_out[1 : buckets * 2 : 2] = np.where(first_min, y_max, y_min)
        if rem >= 2:
            xr_t, yr_t = x[usable:], y[usable:]
            i_min, i_max = int(np.argmin(yr_t)), int(np.argmax(yr_t))
            if i_min <= i_max:
                x_out[-2], y_out[-2] = xr_t[i_min], yr_t[i_min]
                x_out[-1], y_out[-1] = xr_t[i_max], yr_t[i_max]
            else:
                x_out[-2], y_out[-2] = xr_t[i_max], yr_t[i_max]
                x_out[-1], y_out[-1] = xr_t[i_min], yr_t[i_min]
        elif rem == 1:
            x_out[-1], y_out[-1] = x[-1], y[-1]
        x, y = x_out, y_out
        n = x.size

    span_x = np.float32(max(abs(x1 - x0), 1e-12))
    span_y = np.float32(max(abs(y1 - y0), 1e-12))
    scale_x = np.float32(max(viewport_px[0], 1.0)) / span_x
    scale_y = np.float32(max(viewport_px[1], 1.0)) / span_y

    # Segment vectors in screen pixels
    dx_seg = (x[1:] - x[:-1]) * scale_x
    dy_seg = (y[1:] - y[:-1]) * scale_y
    seg_len = np.maximum(np.hypot(dx_seg, dy_seg), np.float32(1e-6))

    # For dense/sub-pixel horizontal steps (dx_seg < 2.5 px), keep nx non-negative so vertical oscillations
    # and min/max envelopes never flip normal orientation (-1 -> +1) and twist the triangle strip into a bowtie!
    nx_seg = np.where(dx_seg < np.float32(2.5), np.abs(dy_seg) / seg_len, -dy_seg / seg_len)
    ny_seg = dx_seg / seg_len

    # Smooth vertex normals by averaging incoming and outgoing segment normals
    nx = np.empty(n, dtype=np.float32)
    ny = np.empty(n, dtype=np.float32)
    nx[0], ny[0] = nx_seg[0], ny_seg[0]
    nx[-1], ny[-1] = nx_seg[-1], ny_seg[-1]
    if n > 2:
        nx[1:-1] = np.float32(0.5) * (nx_seg[:-1] + nx_seg[1:])
        ny[1:-1] = np.float32(0.5) * (ny_seg[:-1] + ny_seg[1:])

    v_len = np.hypot(nx, ny)
    small = v_len < np.float32(0.35)
    if np.any(small):
        idx_s = np.minimum(np.nonzero(small)[0], n - 2)
        nx[small] = np.abs(nx_seg[idx_s])
        ny[small] = np.maximum(ny_seg[idx_s], np.float32(0.25))
        v_len = np.hypot(nx, ny)

    inv_vlen = np.float32(1.0) / np.maximum(v_len, np.float32(1e-6))
    nx *= inv_vlen
    ny *= inv_vlen

    # Expand each point into 2 vertices (+normal with edge=+1, -normal with edge=-1)
    verts = np.empty((n * 2, 5), dtype=np.float32)
    verts[0::2, 0] = x
    verts[0::2, 1] = y
    verts[0::2, 2] = nx
    verts[0::2, 3] = ny
    verts[0::2, 4] = 1.0

    verts[1::2, 0] = x
    verts[1::2, 1] = y
    verts[1::2, 2] = -nx
    verts[1::2, 3] = -ny
    verts[1::2, 4] = -1.0
    return verts


class DpoShaderState(OpenGLState):
    def __init__(self, widget):
        super().__init__(widget)
        self.targets: list[QtOpenGL.QOpenGLFramebufferObject] = []
        self.history_key = None
        self.last_time = None
        self.serial = -1
        self.ribbon_vao = QtOpenGL.QOpenGLVertexArrayObject(widget)
        self.ribbon_vbo = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.quad_vao = QtOpenGL.QOpenGLVertexArrayObject(widget)
        self.quad_vbo = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.vertex_count = 0

    def setup(self, context):
        if self.context is context:
            return
        if self.context is not None:
            try:
                self.context.aboutToBeDestroyed.disconnect(self.cleanup)
            except Exception:
                pass
            self.cleanup()

        widget = self.parent()
        l_vert, l_frag = _adapt_glsl(context, SDF_LINE_VERT, SDF_LINE_FRAG)
        q_vert, q_frag = _adapt_glsl(context, QUAD_VERT, QUAD_FRAG)

        line_prog = compile_program(
            widget, "WaveDeskSdfLine", l_vert, l_frag, [(0, "a_pos_normal"), (1, "a_edge")]
        )
        quad_prog = compile_program(
            widget, "WaveDeskDpoQuad", q_vert, q_frag, [(0, "a_position")]
        )

        self.ribbon_vao.create()
        self.ribbon_vbo.create()
        self.ribbon_vao.bind()
        self.ribbon_vbo.bind()
        stride = 5 * 4
        line_prog.enableAttributeArray(0)
        line_prog.setAttributeBuffer(0, GL.GL_FLOAT, 0, 4, stride)
        line_prog.enableAttributeArray(1)
        line_prog.setAttributeBuffer(1, GL.GL_FLOAT, 4 * 4, 1, stride)
        self.ribbon_vbo.release()
        self.ribbon_vao.release()

        self.quad_vao.create()
        self.quad_vbo.create()
        self.quad_vao.bind()
        self.quad_vbo.bind()
        quad_verts = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32)
        self.quad_vbo.allocate(quad_verts, quad_verts.nbytes)
        quad_prog.enableAttributeArray(0)
        quad_prog.setAttributeBuffer(0, GL.GL_FLOAT, 0, 2)
        self.quad_vbo.release()
        self.quad_vao.release()

        self.context = context
        context.aboutToBeDestroyed.connect(self.cleanup)

    def cleanup(self):
        widget = self.parent()
        if widget is not None:
            widget.makeCurrent()
        self.targets = []
        self.history_key = None
        self.last_time = None
        self.serial = -1
        self.ribbon_vao.destroy()
        self.ribbon_vbo.destroy()
        self.quad_vao.destroy()
        self.quad_vbo.destroy()
        super().cleanup()


class GpuWaveformCurve(pg.PlotCurveItem):
    """High-performance OpenGL SDF Anti-Aliased & Digital Phosphor (DPO) Waveform Curve."""

    def __init__(self, color: str = "#00f2fe", width: float = 2.2, dpo_enabled: bool = False, **kwargs):
        pen = pg.mkPen(color, width=width)
        super().__init__(pen=pen, **kwargs)
        self.curve = self
        self.base_color = QtGui.QColor(color)
        self.line_width = float(width)
        self.dpo_enabled = dpo_enabled
        self.dpo_decay_tau = 0.22  # Crisp persistence decay constant in seconds
        self.glow_enabled = True
        self.serial = 0
        self.last_error = None
        self._fast_bounds: QtCore.QRectF | None = None

    def set_color(self, color: str):
        self.base_color = QtGui.QColor(color)
        self.setPen(pg.mkPen(self.base_color, width=self.line_width))

    def set_dpo(self, enabled: bool, decay_tau: float = 0.22):
        self.dpo_enabled = bool(enabled)
        self.dpo_decay_tau = max(0.06, float(decay_tau))
        self.update()

    def set_fast_data(self, x: np.ndarray, y: np.ndarray):
        """Low-overhead waveform buffer update for OpenGL/Software streaming with accurate dynamic bounds."""
        self.xData = np.asarray(x, dtype=np.float32)
        self.yData = np.asarray(y, dtype=np.float32)
        self.path = None
        self.fillPath = None
        self._fillPathList = None
        self._mouseShape = None
        self._lineSegmentsRendered = False
        self._boundsCache = [None, None]
        self.serial = getattr(self, "serial", 0) + 1
        if self.xData.size >= 2:
            x0, x1 = float(self.xData[0]), float(self.xData[-1])
            ymin = float(np.min(self.yData))
            ymax = float(np.max(self.yData))
            if not np.isfinite(ymin) or not np.isfinite(ymax):
                ymin, ymax = -10.0, 10.0
            pad_y = max(abs(ymax - ymin) * 0.15, 0.5)
            self._fast_bounds = QtCore.QRectF(
                min(x0, x1),
                ymin - pad_y,
                max(abs(x1 - x0), 1e-9),
                max((ymax - ymin) + 2.0 * pad_y, 1.0),
            )
        self.update()

    def boundingRect(self) -> QtCore.QRectF:
        if self._fast_bounds is not None:
            return self._fast_bounds
        if (view := self.getViewBox()) is not None:
            vr = view.viewRange()
            return QtCore.QRectF(
                vr[0][0],
                vr[1][0],
                max(vr[0][1] - vr[0][0], 1e-9),
                max(vr[1][1] - vr[1][0], 1e-9),
            )
        return super().boundingRect()

    def setData(self, *args, **kwargs):
        self._fast_bounds = None
        super().setData(*args, **kwargs)
        self.serial = getattr(self, "serial", 0) + 1

    def capture_viewport(self, widget) -> QtGui.QImage:
        """Read the last painted Qt FBO for diagnostic verification."""
        if self.glstate is None or _native_gl is None:
            return QtGui.QImage()
        widget.makeCurrent()
        gl = widget.getFunctions()
        previous = framebuffer_binding()
        try:
            gl.glBindFramebuffer(GL_FRAMEBUFFER, self.glstate.output_fbo)
            ratio = widget.devicePixelRatioF()
            w = round(widget.width() * ratio)
            h = round(widget.height() * ratio)
            pixels = np.empty((h, w, 4), dtype=np.uint8)
            _native_gl.glReadPixels(0, 0, w, h, 0x1908, 0x1401, pixels.ctypes.data)
            pixels = pixels[::-1].copy()
            image = QtGui.QImage(pixels.data, w, h, w * 4, QtGui.QImage.Format.Format_RGBA8888).copy()
            image.setDevicePixelRatio(ratio)
            return image
        finally:
            gl.glBindFramebuffer(GL_FRAMEBUFFER, previous)
            widget.doneCurrent()

    def paint(self, painter, option, widget):
        if self.xData is None or len(self.xData) < 2:
            return
        if not isinstance(widget, OpenGLHelpers.GraphicsViewGLWidget):
            return super().paint(painter, option, widget)
        if self.glstate is None or not isinstance(self.glstate, DpoShaderState):
            self.glstate = DpoShaderState(widget)
            self.sigPlotChanged.connect(self.glstate.verticesChanged)

        painter.beginNativePainting()
        self.glstate.output_fbo = framebuffer_binding()
        try:
            self.paintGL(widget)
            self.last_error = None
        except Exception as err:
            if self.last_error is None:
                import traceback
                traceback.print_exc()
            self.last_error = str(err)
        finally:
            widget.getFunctions().glBindFramebuffer(GL_FRAMEBUFFER, self.glstate.output_fbo)
            painter.endNativePainting()

    def _upload_and_draw_sdf_line(self, widget, view, viewport_size: QtCore.QSize, alpha_scale: float = 1.0):
        state: DpoShaderState = self.glstate
        gl = widget.getFunctions()
        prog = widget.retrieveProgram("WaveDeskSdfLine")
        prog.bind()

        center = self.boundingRect().center()
        xc, yc = float(center.x()), float(center.y())

        # Exact pyqtgraph sceneTransform -> widget ortho projection (with float64 origin shift)
        proj = QtGui.QMatrix4x4()
        proj.ortho(widget.rect())
        tr = self.sceneTransform()
        tr.translate(xc, yc)
        mvp = proj * QtGui.QMatrix4x4(tr)

        vr = view.viewRange()
        vb_rect = view.sceneBoundingRect()
        vw_logical = float(max(16.0, vb_rect.width()))
        vh_logical = float(max(16.0, vb_rect.height()))

        x_rel = np.asarray(self.xData, dtype=np.float32) - np.float32(xc)
        y_rel = np.asarray(self.yData, dtype=np.float32) - np.float32(yc)

        verts = build_sdf_ribbon_vertices(
            x_rel,
            y_rel,
            ((vr[0][0] - xc, vr[0][1] - xc), (vr[1][0] - yc, vr[1][1] - yc)),
            (vw_logical, vh_logical),
            max_segments=max(1024, int(vw_logical * 1.5)),
        )
        if verts.shape[0] < 4:
            prog.release()
            return

        state.ribbon_vao.bind()
        state.ribbon_vbo.bind()
        state.ribbon_vbo.allocate(verts, verts.nbytes)

        w_logical = float(max(1.0, widget.width()))
        h_logical = float(max(1.0, widget.height()))
        OpenGLHelpers.setUniformValue(prog, "u_mvp", mvp)
        OpenGLHelpers.setUniformValue(prog, "u_viewport", QtGui.QVector2D(w_logical, h_logical))
        half_w = max(1.3, self.line_width * (1.45 if self.glow_enabled else 1.05))
        gl.glUniform1f(prog.uniformLocation("u_half_width"), float(half_w))
        c = self.base_color
        OpenGLHelpers.setUniformValue(
            prog,
            "u_color",
            QtGui.QVector4D(c.redF(), c.greenF(), c.blueF(), c.alphaF() * alpha_scale),
        )
        gl.glUniform1i(prog.uniformLocation("u_glow_mode"), int(1 if self.glow_enabled else 0))

        gl.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, verts.shape[0])
        state.ribbon_vbo.release()
        state.ribbon_vao.release()
        prog.release()

    def _draw_quad(self, widget, texture_id: int, decay: float, thermal: bool):
        state: DpoShaderState = self.glstate
        gl = widget.getFunctions()
        prog = widget.retrieveProgram("WaveDeskDpoQuad")
        prog.bind()
        gl.glUniform1i(prog.uniformLocation("u_texture"), 7)
        gl.glUniform1f(prog.uniformLocation("u_decay"), float(decay))
        gl.glUniform1i(prog.uniformLocation("u_thermal_lut"), int(1 if thermal else 0))
        c = self.base_color
        OpenGLHelpers.setUniformValue(prog, "u_base_color", QtGui.QVector3D(c.redF(), c.greenF(), c.blueF()))

        gl.glActiveTexture(GL_TEXTURE0 + 7)
        gl.glBindTexture(GL_TEXTURE_2D, texture_id)
        if hasattr(gl, "glBindSampler"):
            gl.glBindSampler(7, 0)
        gl.glTexParameteri(GL_TEXTURE_2D, 0x2801, 0x2601)
        gl.glTexParameteri(GL_TEXTURE_2D, 0x2800, 0x2601)
        gl.glTexParameteri(GL_TEXTURE_2D, 0x2802, 0x812F)
        gl.glTexParameteri(GL_TEXTURE_2D, 0x2803, 0x812F)

        state.quad_vao.bind()
        gl.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        state.quad_vao.release()

        gl.glBindTexture(GL_TEXTURE_2D, 0)
        gl.glActiveTexture(GL_TEXTURE0)
        prog.release()

    def paintGL(self, widget):
        if (view := self.getViewBox()) is None:
            return
        state: DpoShaderState = self.glstate
        state.setup(widget.context())
        gl = widget.getFunctions()
        dpr = widget.devicePixelRatioF()
        size = QtCore.QSize(round(widget.width() * dpr), round(widget.height() * dpr))

        if not self.dpo_enabled:
            state.history_key = None
            gl.glViewport(0, 0, size.width(), size.height())
            widget.setViewboxClip(view)
            gl.glEnable(GL.GL_BLEND)
            gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            self._upload_and_draw_sdf_line(widget, view, size, alpha_scale=1.0)
            gl.glDisable(GL.GL_BLEND)
            return

        key = (
            size.width(),
            size.height(),
            tuple(view.viewRange()[0]),
            tuple(view.viewRange()[1]),
            view.sceneBoundingRect(),
        )
        changed = key != state.history_key
        if not state.targets or state.targets[0].size() != size:
            state.targets = [QtOpenGL.QOpenGLFramebufferObject(size) for _ in range(2)]
            if not all(f.isValid() for f in state.targets):
                raise RuntimeError("Cannot allocate DPO framebuffer")
            changed = True

        if changed or state.serial != self.serial:
            now = time.monotonic()
            previous, target = state.targets
            target.bind()
            gl.glViewport(0, 0, size.width(), size.height())
            gl.glDisable(GL.GL_SCISSOR_TEST)
            gl.glDisable(GL.GL_BLEND)
            gl.glClearColor(0.0, 0.0, 0.0, 0.0)
            gl.glClear(GL.GL_COLOR_BUFFER_BIT)

            if not changed and state.last_time is not None:
                dt = max(0.001, now - state.last_time)
                decay = math.exp(-dt / self.dpo_decay_tau)
                self._draw_quad(widget, previous.texture(), decay, thermal=False)

            # Accumulate new trace into DPO FBO with additive blending
            gl.glEnable(GL.GL_BLEND)
            gl.glBlendFunc(GL.GL_SRC_ALPHA, 1)
            self._upload_and_draw_sdf_line(widget, view, size, alpha_scale=0.55)
            gl.glDisable(GL.GL_BLEND)

            state.targets = [target, previous]
            state.serial = self.serial
            state.history_key = key
            state.last_time = now

        # Composite DPO FBO onto output framebuffer with thermal intensity grading
        gl.glBindFramebuffer(GL_FRAMEBUFFER, state.output_fbo)
        gl.glViewport(0, 0, size.width(), size.height())
        widget.setViewboxClip(view)
        gl.glEnable(GL.GL_BLEND)
        gl.glBlendFunc(1, GL.GL_ONE_MINUS_SRC_ALPHA)
        self._draw_quad(widget, state.targets[0].texture(), 1.0, thermal=True)
        gl.glDisable(GL.GL_BLEND)
