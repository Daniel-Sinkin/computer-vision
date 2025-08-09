import math
import sys
from pathlib import Path
from time import perf_counter

import glfw
import imgui
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
from OpenGL import GL as gl

# ---------------------------
# Helpers: math (no extra deps)
# ---------------------------


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = normalize(center - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def perspective(
    fovy_radians: float, aspect: float, znear: float, zfar: float
) -> np.ndarray:
    f = 1.0 / math.tan(fovy_radians / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def rotation_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


# ---------------------------
# Camera
# ---------------------------


class Camera:
    def __init__(
        self,
        position: np.ndarray,
        target: np.ndarray,
        fov_deg: float = 50.0,
        znear: float = 0.1,
        zfar: float = 100.0,
    ):
        self.position = position.astype(np.float32)
        self.target = target.astype(np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.fov_deg = fov_deg
        self.znear = znear
        self.zfar = zfar

    def view(self) -> np.ndarray:
        return look_at(self.position, self.target, self.up)

    def proj(self, aspect: float) -> np.ndarray:
        return perspective(math.radians(self.fov_deg), aspect, self.znear, self.zfar)


# ---------------------------
# Shader utils
# ---------------------------


def compile_shader(src: str, shader_type) -> int:
    sid = gl.glCreateShader(shader_type)
    gl.glShaderSource(sid, src)
    gl.glCompileShader(sid)
    if gl.glGetShaderiv(sid, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
        log = gl.glGetShaderInfoLog(sid).decode("utf-8")
        gl.glDeleteShader(sid)
        raise RuntimeError(f"Shader compile error:\n{log}")
    return sid


def link_program(vsid: int, fsid: int) -> int:
    pid = gl.glCreateProgram()
    gl.glAttachShader(pid, vsid)
    gl.glAttachShader(pid, fsid)
    gl.glLinkProgram(pid)
    if gl.glGetProgramiv(pid, gl.GL_LINK_STATUS) != gl.GL_TRUE:
        log = gl.glGetProgramInfoLog(pid).decode("utf-8")
        gl.glDeleteProgram(pid)
        raise RuntimeError(f"Program link error:\n{log}")
    gl.glDetachShader(pid, vsid)
    gl.glDetachShader(pid, fsid)
    gl.glDeleteShader(vsid)
    gl.glDeleteShader(fsid)
    return pid


# ---------------------------
# Geometry: cube with per-face color & normals
# ---------------------------


def create_cube_vao():
    # Interleaved: position (x,y,z), normal (nx,ny,nz), color (r,g,b)
    # 6 faces * 4 verts each = 24 verts. Indices are 6 faces * 2 tris = 12 tris.
    # fmt: off
    c_front  = [0.90, 0.30, 0.30]
    c_back   = [0.30, 0.60, 0.90]
    c_left   = [0.30, 0.90, 0.50]
    c_right  = [0.90, 0.70, 0.30]
    c_top    = [0.80, 0.80, 0.80]
    c_bottom = [0.50, 0.50, 0.50]

    vertices = np.array([
        # Front (z+), normal (0,0,1)
        -1.0, -1.0,  1.0,   0.0, 0.0,  1.0,  *c_front,
         1.0, -1.0,  1.0,   0.0, 0.0,  1.0,  *c_front,
         1.0,  1.0,  1.0,   0.0, 0.0,  1.0,  *c_front,
        -1.0,  1.0,  1.0,   0.0, 0.0,  1.0,  *c_front,

        # Back (z-), normal (0,0,-1)
         1.0, -1.0, -1.0,   0.0, 0.0, -1.0,  *c_back,
        -1.0, -1.0, -1.0,   0.0, 0.0, -1.0,  *c_back,
        -1.0,  1.0, -1.0,   0.0, 0.0, -1.0,  *c_back,
         1.0,  1.0, -1.0,   0.0, 0.0, -1.0,  *c_back,

        # Left (x-), normal (-1,0,0)
        -1.0, -1.0, -1.0,  -1.0, 0.0,  0.0,  *c_left,
        -1.0, -1.0,  1.0,  -1.0, 0.0,  0.0,  *c_left,
        -1.0,  1.0,  1.0,  -1.0, 0.0,  0.0,  *c_left,
        -1.0,  1.0, -1.0,  -1.0, 0.0,  0.0,  *c_left,

        # Right (x+), normal (1,0,0)
         1.0, -1.0,  1.0,   1.0, 0.0,  0.0,  *c_right,
         1.0, -1.0, -1.0,   1.0, 0.0,  0.0,  *c_right,
         1.0,  1.0, -1.0,   1.0, 0.0,  0.0,  *c_right,
         1.0,  1.0,  1.0,   1.0, 0.0,  0.0,  *c_right,

        # Top (y+), normal (0,1,0)
        -1.0,  1.0,  1.0,   0.0, 1.0,  0.0,  *c_top,
         1.0,  1.0,  1.0,   0.0, 1.0,  0.0,  *c_top,
         1.0,  1.0, -1.0,   0.0, 1.0,  0.0,  *c_top,
        -1.0,  1.0, -1.0,   0.0, 1.0,  0.0,  *c_top,

        # Bottom (y-), normal (0,-1,0)
        -1.0, -1.0, -1.0,   0.0,-1.0,  0.0,  *c_bottom,
         1.0, -1.0, -1.0,   0.0,-1.0,  0.0,  *c_bottom,
         1.0, -1.0,  1.0,   0.0,-1.0,  0.0,  *c_bottom,
        -1.0, -1.0,  1.0,   0.0,-1.0,  0.0,  *c_bottom,
    ], dtype=np.float32)

    indices = np.array([
        # front
        0, 1, 2,  2, 3, 0,
        # back
        4, 5, 6,  6, 7, 4,
        # left
        8, 9,10, 10,11, 8,
        # right
       12,13,14, 14,15,12,
        # top
       16,17,18, 18,19,16,
        # bottom
       20,21,22, 22,23,20,
    ], dtype=np.uint32)
    # fmt: on

    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    ebo = gl.glGenBuffers(1)

    gl.glBindVertexArray(vao)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(
        gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW
    )

    stride = 9 * 4  # 9 floats * 4 bytes each
    # positions (location = 0)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(
        0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(0)
    )
    # normals (location = 1), offset 12 bytes
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(
        1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(12)
    )
    # colors (location = 2), offset 24 bytes
    gl.glEnableVertexAttribArray(2)
    gl.glVertexAttribPointer(
        2, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(24)
    )

    gl.glBindVertexArray(0)

    return vao, vbo, ebo, indices.size


# ---------------------------
# Framebuffer (off-screen) for each camera
# ---------------------------


class OffscreenView:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.fbo = None
        self.color_tex = None
        self.depth_rb = None
        self._alloc()

    def _alloc(self):
        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)

        self.color_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.color_tex)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            self.width,
            self.height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            self.color_tex,
            0,
        )

        self.depth_rb = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.depth_rb)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH24_STENCIL8, self.width, self.height
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER,
            gl.GL_DEPTH_STENCIL_ATTACHMENT,
            gl.GL_RENDERBUFFER,
            self.depth_rb,
        )

        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer incomplete: 0x{status:X}")

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def resize(self, width: int, height: int):
        if width == self.width and height == self.height:
            return
        self.destroy()
        self.width, self.height = width, height
        self._alloc()

    def bind(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, self.width, self.height)

    def unbind(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def destroy(self):
        try:
            if self.depth_rb:
                gl.glDeleteRenderbuffers(1, [self.depth_rb])
            if self.color_tex:
                gl.glDeleteTextures(1, [self.color_tex])
            if self.fbo:
                gl.glDeleteFramebuffers(1, [self.fbo])
        finally:
            self.fbo = None
            self.color_tex = None
            self.depth_rb = None


# ---------------------------
# Shaders (GL 3.3 core)
# ---------------------------

VERT_SRC = """
#version 330 core
layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec3 in_color;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 v_color;
out vec3 v_normal_ws;

void main() {
    v_color = in_color;
    // transform normal by model (no non-uniform scaling here)
    v_normal_ws = mat3(u_model) * in_normal;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

FRAG_SRC = """
#version 330 core
in vec3 v_color;
in vec3 v_normal_ws;
out vec4 frag_color;

uniform vec3 u_light_dir; // direction FROM light TOWARD scene, world space (normalized)

void main() {
    vec3 N = normalize(v_normal_ws);
    float ndl = max(dot(N, normalize(u_light_dir)), 0.0);
    float ambient = 0.25;
    float lighting = ambient + (1.0 - ambient) * ndl;
    vec3 color = v_color * lighting;
    frag_color = vec4(color, 1.0);
}
"""


# ---------------------------
# App
# ---------------------------


def main():
    if not glfw.init():
        print("Failed to initialize GLFW.", file=sys.stderr)
        sys.exit(1)

    # GL 3.3 core profile
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    try:
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)  # type: ignore[attr-defined]
    except Exception:
        pass
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(1280, 800, "Epipolar Geometry Sandbox", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create window.", file=sys.stderr)
        sys.exit(1)

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Dear ImGui
    imgui.create_context()
    io = imgui.get_io()
    io.ini_file_name = None  # keep things clean
    impl = GlfwRenderer(window, attach_callbacks=False)

    # GL state
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_CULL_FACE)  # optional but nice once indices are correct
    gl.glCullFace(gl.GL_BACK)
    gl.glFrontFace(gl.GL_CCW)
    gl.glClearColor(0.07, 0.07, 0.08, 1.0)

    # Geometry & shaders
    vao, vbo, ebo, index_count = create_cube_vao()
    vs = compile_shader(VERT_SRC, gl.GL_VERTEX_SHADER)
    fs = compile_shader(FRAG_SRC, gl.GL_FRAGMENT_SHADER)
    program = link_program(vs, fs)
    u_mvp = gl.glGetUniformLocation(program, "u_mvp")
    u_model = gl.glGetUniformLocation(program, "u_model")
    u_light_dir = gl.glGetUniformLocation(program, "u_light_dir")

    # Two cameras
    cam_left = Camera(
        position=np.array([-3.0, 2.0, 5.0], dtype=np.float32),
        target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    cam_right = Camera(
        position=np.array([+3.0, 2.0, 5.0], dtype=np.float32),
        target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )

    # Off-screen views (we’ll resize to fit available ImGui content region later)
    off_left = OffscreenView(640, 640)
    off_right = OffscreenView(640, 640)

    start = perf_counter()

    # simple directional light (world space)
    light_dir = normalize(np.array([0.6, 0.8, 0.4], dtype=np.float32))

    def render_scene(view: np.ndarray, proj: np.ndarray, theta: float):
        model = rotation_y(theta).astype(np.float32)
        mvp = (proj @ view @ model).astype(np.float32)
        gl.glUseProgram(program)
        # IMPORTANT: upload transpose so GLSL (column-major) sees the right matrices
        gl.glUniformMatrix4fv(u_mvp, 1, gl.GL_FALSE, mvp.T)
        gl.glUniformMatrix4fv(u_model, 1, gl.GL_FALSE, model.T)
        gl.glUniform3fv(u_light_dir, 1, light_dir)
        gl.glBindVertexArray(vao)
        gl.glDrawElements(gl.GL_TRIANGLES, index_count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

    # Minimal keyboard: tweak baseline parallax by moving right camera in X, move both Z
    def handle_input():
        speed = 0.05
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            cam_left.position[0] -= speed
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            cam_left.position[0] += speed
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            cam_right.position[0] -= speed
        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            cam_right.position[0] += speed
        # dolly in/out together
        if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
            cam_left.position[2] -= speed
            cam_right.position[2] -= speed
        if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
            cam_left.position[2] += speed
            cam_right.position[2] += speed

    while not glfw.window_should_close(window):
        glfw.poll_events()
        handle_input()
        impl.process_inputs()

        now = perf_counter()
        theta = (now - start) * 0.7

        # Decide render sizes from available ImGui content region (we’ll display scaled if needed)
        imgui.new_frame()
        imgui.set_next_window_position(10, 10, imgui.ONCE)
        imgui.set_next_window_size(1260, 780, imgui.ONCE)
        imgui.begin("Epipolar Geometry Sandbox", True, imgui.WINDOW_NO_MOVE)

        avail_w, avail_h = imgui.get_content_region_available()
        # Put two images side-by-side with a small gap
        gap = 8.0
        target_h = max(200.0, avail_h - 8.0)
        target_w_each = max(200.0, (avail_w - gap) * 0.5)

        # Resize FBOs only if the available size meaningfully changed (round to integers)
        new_w = max(64, int(target_w_each))
        new_h = max(64, int(target_h))
        off_left.resize(new_w, new_h)
        off_right.resize(new_w, new_h)

        # Render left camera
        off_left.bind()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        view_l = cam_left.view()
        proj_l = cam_left.proj(aspect=off_left.width / float(off_left.height))
        render_scene(view_l, proj_l, theta)
        off_left.unbind()

        # Render right camera
        off_right.bind()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        view_r = cam_right.view()
        proj_r = cam_right.proj(aspect=off_right.width / float(off_right.height))
        render_scene(view_r, proj_r, theta)
        off_right.unbind()

        # UI: left image
        imgui.text("Left Camera")
        imgui.image(
            off_left.color_tex, target_w_each, target_h, uv0=(0, 1), uv1=(1, 0)
        )  # flip Y for GL->ImGui
        imgui.same_line()
        # UI: right image
        imgui.begin_group()
        imgui.text("Right Camera")
        imgui.image(
            off_right.color_tex, target_w_each, target_h, uv0=(0, 1), uv1=(1, 0)
        )
        imgui.end_group()

        imgui.separator()
        imgui.text_wrapped(
            "Controls: A/D move left camera in X. LEFT/RIGHT move right camera in X. "
            "UP/DOWN dolly both cameras in Z. Adjust window size to rescale views."
        )

        # Example readouts (handy later for epipolar math)
        baseline = float(abs(cam_right.position[0] - cam_left.position[0]))
        imgui.text(f"Baseline: {baseline:.3f}")
        imgui.text(f"Left pos: {tuple(float(x) for x in cam_left.position)}")
        imgui.text(f"Right pos: {tuple(float(x) for x in cam_right.position)}")

        imgui.end()

        # Draw ImGui
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        w, h = glfw.get_framebuffer_size(window)
        gl.glViewport(0, 0, w, h)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        imgui.render()
        impl.render(imgui.get_draw_data())

        glfw.swap_buffers(window)

    # Cleanup
    off_left.destroy()
    off_right.destroy()
    gl.glDeleteVertexArrays(1, [vao])
    gl.glDeleteBuffers(1, [vbo])
    gl.glDeleteBuffers(1, [ebo])
    gl.glDeleteProgram(program)

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
