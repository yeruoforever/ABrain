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

DEBUG = True


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
        self.shader_plane = load_shader_from_text("plane")
        self.shader_3d = load_shader_from_text("3d")
        symbols = [
            "plane",
            "screen",
            "focus",
            "slice",
            "range",
            "hu_range",
            "color_bg",
            "color_1",
            "color_2",
            "img",
            "seg",
            "mix_rate",
        ]
        self.ptrs_plane = shader_ptrs(self.shader_plane, symbols)
        symbols = [
            "fov",
            "screen",
            "eye",
            "V2W",
            "step",
            "alpha",
            "vox_min",
            "vox_max",
            "cube_a",
            "cube_b",
            "W2M",
            "img",
            "seg",
            "color_bg",
            "color_1",
            "color_2",
            "mix_rate",
        ]
        self.ptrs_3d = shader_ptrs(self.shader_3d, symbols)
        print(self.ptrs_plane, self.ptrs_3d)

    def create_buffer(self):
        vertices = np.array(
            [[-1.0, -1.0], [-1.0, +1.0], [+1.0, -1.0], [+1.0, +1.0]],
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
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        bg_img = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, bg_img)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
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
        # glGenerateMipmap(GL_TEXTURE_3D)

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
            with Texture(GL_TEXTURE0):
                glBindTexture(GL_TEXTURE_3D, self.texture_img)
                self.load_volume_texture(img)
            with Texture(GL_TEXTURE1):
                glBindTexture(GL_TEXTURE_3D, self.texture_seg)
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
        with Program(self.shader_plane):
            W, H = self.state.viewport[2], self.state.viewport[3]
            W, H = W * self.state.plane_scale, H * self.state.plane_scale
            glUniform2f(self.ptrs_plane["screen"], float(W), float(H))
            glUniform3f(self.ptrs_plane["focus"], *self.state.plane_focus)
            glUniform3f(self.ptrs_plane["slice"], *self.state.plane_slice)
            glUniform3f(self.ptrs_plane["range"], *self.state.img_region)
            glUniform2f(
                self.ptrs_plane["hu_range"], self.state.voxel_min, self.state.voxel_max
            )

            glUniform3f(self.ptrs_plane["color_bg"], *self.state.color_background)
            glUniform3f(self.ptrs_plane["color_1"], *self.state.color_target_1)
            glUniform3f(self.ptrs_plane["color_2"], *self.state.color_target_2)
            glUniform1f(self.ptrs_plane["mix_rate"], self.state.color_overlap)

            glUniform1i(self.ptrs_plane["img"], 0)
            glUniform1i(self.ptrs_plane["seg"], 1)

            # IMG,SEG

        with Program(self.shader_3d):
            glUniform2f(
                self.ptrs_3d["screen"],
                float(self.state.viewport[2]),
                float(self.state.viewport[3]),
            )
            # glUniform1d()
            glUniform1f(self.ptrs_3d["fov"], self.state.view_radians)
            glUniform3f(self.ptrs_3d["eye"], *self.state.camera_origin)
            view_i = self.state.V2W
            glUniformMatrix4fv(
                self.ptrs_3d["V2W"],
                1,
                GL_FALSE,
                glm.value_ptr(view_i),
            )
            glUniform1f(self.ptrs_3d["step"], self.state.ray_step)
            glUniform1f(self.ptrs_3d["alpha"], self.state.ray_alpha)
            glUniform1f(self.ptrs_3d["vox_min"], self.state.voxel_min)
            glUniform1f(self.ptrs_3d["vox_max"], self.state.voxel_max)
            bbox = self.state.cube_bounding()
            glUniform3fv(self.ptrs_3d["cube_a"], 1, glm.value_ptr(bbox[0]))
            glUniform3fv(self.ptrs_3d["cube_b"], 1, glm.value_ptr(bbox[1]))
            # only need load once
            w2m = self.state.W2M
            glUniformMatrix4fv(self.ptrs_3d["W2M"], 1, GL_FALSE, glm.value_ptr(w2m))

            glUniform3f(self.ptrs_3d["color_bg"], *self.state.color_background)
            glUniform3f(self.ptrs_3d["color_1"], *self.state.color_target_1)
            glUniform3f(self.ptrs_3d["color_2"], *self.state.color_target_2)
            glUniform1f(self.ptrs_3d["mix_rate"], self.state.color_overlap)

            glUniform1i(self.ptrs_3d["img"], 0)
            glUniform1i(self.ptrs_3d["seg"], 1)

    def update_texture(self):
        if self.state.need_reload_file():
            print("Reload texture")
            img, spacing, directions = get_img(self.state.input_file)
            seg = get_seg(self.state.input_file)
            print(img.shape, spacing, directions)
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

    def draw_panel(self, plane_type):
        if plane_type == PLANE_3D:
            shader = self.shader_3d
            with Program(shader):
                with VAO(self.vao):
                    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        else:
            shader = self.shader_plane
            with Program(shader):
                glUniform1i(self.ptrs_plane["plane"], plane_type)
                with VAO(self.vao):
                    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    def do_render(self):
        glClear(GL_COLOR_BUFFER_BIT)
        self.check_and_update()
        if not self.state.file_opened:
            return
        if self.state.view_type == PlANE_ALL:
            for i, tp in enumerate(
                [PLANE_CORONAL, PLANE_3D, PLANE_CROSS, PLANE_SAGITTAL]
            ):
                w, h = self.state.viewport[2] // 2, self.state.viewport[3] // 2
                a, b = i % 2, i // 2
                glViewport(a * w, b * h, w, h)
                self.draw_panel(tp)
        else:
            glViewport(0, 0, self.state.viewport[2], self.state.viewport[3])
            self.draw_panel(self.state.view_type)

    def render_loop(self):
        glfw.poll_events()
        with self.gui:
            self.do_render()
        glfw.swap_buffers(self.window)

    def run(self):
        self.prepare_glfw()
        self.prepare_opengl()
        if DEBUG:
            self.state.input_file = (
                "/Users/yeruo/WorkSpace/CTCSF/5.0mm/img/20230102000345.nii.gz"
                # "/Users/yeruo/WorkSpace/3月颅脑CT及标记数据数据/3月颅脑导数据/标记数据/img/1.25mm/20230327007282.nii.gz"
            )
            self.update_texture()
        while glfw.window_should_close(self.window) == 0:
            self.render_loop()
        glfw.terminate()


if __name__ == "__main__":
    render = Render()
    render.run()
