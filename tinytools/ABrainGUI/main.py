import sys
import os
from typing import *

import glfw
import glm
import numpy as np
from gui import *
from OpenGL.GL import *
from shader import *
from status import *
from utils import *
from neuron import *

with open(
    os.path.join(os.path.dirname(__file__), "vertex.vert"),
    "r",
) as f:
    vertex_source = f.read()
vertex_ptrs = ["screen_size", "screen_plane", "eye", "V2W"]
with open(
    os.path.join(os.path.dirname(__file__), "fragment.frag"),
    "r",
) as f:
    fragment_source = f.read()
fragment_ptrs = [
    "step",
    "alpha",
    "pix_min",
    "pix_max",
    "eye",
    "cube_a",
    "cube_b",
    "W2M",
    "volume",
]


class Render(object):
    def __init__(self) -> None:
        self.init_status()
        self.init_glfw()
        self.init_gui()

        self.create_buffer()
        self.create_shader()
        self.create_texture()

    def __del__(self):
        pass

    def init_status(self):
        self.state = Status()

    def init_gui(self):
        self.gui = GUI(self.window, self.state)

    def create_shader(self):
        self.shader = load_shader_from_text(None, vertex_source, fragment_source)
        symbols = set(vertex_ptrs + fragment_ptrs)
        self.ptrs = shader_ptrs(self.shader, symbols)
        print(self.ptrs)

    def create_buffer(self):
        vertices = np.array(
            [[-0.5, -0.5], [-0.5, +0.5], [+0.5, -0.5], [+0.5, +0.5]],
            dtype=np.float32,
        )
        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        with VAO(self.vao):
            with Buffer(vbo, GL_ARRAY_BUFFER):
                glBufferData(GL_ARRAY_BUFFER, None, vertices, GL_STATIC_DRAW)
                glVertexAttribPointer(
                    0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), ctypes.c_void_p(0)
                )
                glEnableVertexAttribArray(0)

    def load_volume_texture(self, img: np.ndarray):
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        bg_img = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, bg_img)
        glTexImage3D(
            GL_TEXTURE_3D,
            0,
            GL_R32F,
            *img.shape,
            0,
            GL_RED,
            GL_FLOAT,
            img.ctypes.data_as(GLvoidp)
        )
        glGenerateMipmap(GL_TEXTURE_3D)

    def create_texture(
        self, img: Optional[np.ndarray] = None, seg: Optional[np.ndarray] = None
    ):
        if img is None and seg is None:
            self.texture_img = 0
            self.texture_seg = 0
        else:
            if self.texture_img != 0:
                glDeleteTextures(1, self.texture_img)
            if self.texture_seg != 0:
                glDeleteTextures(1, self.texture_seg)
            self.texture_img = glGenTextures(1)
            self.texture_seg = glGenTextures(1)
            with Texture(self.texture_img, GL_TEXTURE_3D):
                self.load_volume_texture(img)
            with Texture(self.texture_seg, GL_TEXTURE_3D):
                self.load_volume_texture(seg)

    def init_glfw(self):
        if glfw.init() == 0:
            exit()
        glfw.default_window_hints()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        if sys.platform == "darwin":
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        width, height = self.state.screen_size
        window = glfw.create_window(width, height, "ABrain", None, None)
        if not window:
            exit()
        self.window = window
        glfw.make_context_current(window)

    def prepare_glfw(self):
        glfw.swap_interval(1)

    def prepare_opengl(self):
        glClearColor(0.1, 0.1, 0.2, 1)

    def update_shader(self):
        glUniform2f(
            self.ptrs["screen_size"], self.state.viewport[1], self.state.viewport[3]
        )
        glUniform3f(
            self.ptrs["screen_plane"],
            self.state.view_radians,
            self.state.view_near,
            self.state.view_far,
        )
        view_i = glm.inverse(self.state.camera_lookat())
        glUniformMatrix4fv(
            self.ptrs["V2W"],
            1,
            GL_FALSE,
            glm.value_ptr(view_i),
        )
        glUniform3f(self.ptrs["eye"], *self.state.camera_origin)
        glUniform1f(self.ptrs["step"], self.state.ray_step)
        glUniform1f(self.ptrs["alpha"], self.state.ray_alpha)
        glUniform1f(self.ptrs["pix_min"], self.state.voxel_min)
        glUniform1f(self.ptrs["pix_max"], self.state.voxel_max)
        bbox = self.state.cube_bounding()
        glUniform3fv(self.ptrs["cube_a"], 1, GL_FALSE, glm.value_ptr(bbox[0]))
        glUniform3fv(self.ptrs["cube_b"], 1, GL_FALSE, glm.value_ptr(bbox[1]))
        # only need load once
        glUniformMatrix4fv(
            self.ptrs["W2M"], 1, GL_FALSE, glm.value_ptr(self.state.mat_W2M())
        )

    def update_texture(self):
        if self.state.need_reload_file():
            print("Reload texture")
            img, spacing = get_img(self.state.input_file)
            seg = get_seg(self.state.input_file)
            print(img.shape, spacing)
            self.state.update_img_meta(spacing, img.shape, "CT")
            self.create_texture(img, seg)
            self.state.update_file_history()
            self.state.reset_camera()

    def update_screen_size(self):
        glViewport(*self.state.viewport)

    def check_and_update(self):
        self.update_screen_size()
        self.update_texture()
        self.update_shader()

    def do_render(self):
        glClear(GL_COLOR_BUFFER_BIT)
        with Program(self.shader):
            self.check_and_update()
            if self.state.file_opened:
                with VAO(self.vao):
                    with Texture(self.texture_img, GL_TEXTURE_3D):
                        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    def render_loop(self):
        glfw.poll_events()
        with self.gui:
            self.do_render()
        glfw.swap_buffers(self.window)

    def run(self):
        self.prepare_glfw()
        self.prepare_opengl()
        while glfw.window_should_close(self.window) == 0:
            self.render_loop()
        glfw.terminate()


if __name__ == "__main__":
    render = Render()
    render.run()
