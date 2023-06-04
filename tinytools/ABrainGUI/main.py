import sys
from typing import *

import glfw
import glm
import numpy as np
from gui import *
from neuron import *
from OpenGL.GL import *
from shader import *
from status import *
from utils import *

DEBUG = True


TEXTURE_CROSS = GL_TEXTURE0 + PLANE_CROSS  # 0
TEXTURE_CORONAL = GL_TEXTURE0 + PLANE_CORONAL  # 1
TEXTURE_SAGITTAL = GL_TEXTURE0 + PLANE_SAGITTAL  # 2
TEXTURE_3D = GL_TEXTURE0 + PLANE_3D  # 3
TEXTURE_IMG = GL_TEXTURE4  # 4
TEXTURE_SEG = GL_TEXTURE5  # 5


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
        self.shader_foreground = load_shader_from_text("page")

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
        self.ptrs_foreground = shader_ptrs(self.shader_foreground, ["sense"])

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
            img.ctypes.data_as(GLvoidp),
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
            with Texture(TEXTURE_IMG):
                glBindTexture(GL_TEXTURE_3D, self.texture_img)
                self.load_volume_texture(img)
            with Texture(TEXTURE_SEG):
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
        self.prepare_framebuff()

    def prepare_framebuff(self):
        W, H = self.state.viewport[2], self.state.viewport[3]
        self.framebuffers = glGenFramebuffers(PlANE_ALL)
        self.textures = glGenTextures(PlANE_ALL)
        for fid, tid in zip(self.framebuffers, self.textures):
            glBindFramebuffer(GL_FRAMEBUFFER, fid)
            glBindTexture(GL_TEXTURE_2D, tid)
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, None
            )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glFramebufferTexture2D(
                GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tid, 0
            )
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
        for i, tid in enumerate(self.textures):
            with Texture(GL_TEXTURE0 + i):
                glBindTexture(GL_TEXTURE_2D, tid)
        self.state.set_refresh_all()

    def update_shader_3d(self):
        glUniform2f(
            self.ptrs_3d["screen"],
            float(self.state.viewport[2]),
            float(self.state.viewport[3]),
        )
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

        glUniform1i(self.ptrs_3d["img"], 4)
        glUniform1i(self.ptrs_3d["seg"], 5)

    def update_shader_plane(self):
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

        glUniform1i(self.ptrs_plane["img"], 4)
        glUniform1i(self.ptrs_plane["seg"], 5)

    def update_texture(self):
        if self.state.need_reload_file():
            img, spacing, directions = get_img(self.state.input_file)
            seg = get_seg(self.state.input_file)
            self.state.update_img_meta(spacing, img.shape, "CT")
            self.create_texture(img, seg)
            self.state.update_file_history()
            self.state.reset_camera()
            self.state.set_refresh_all()

    def update_screen_size(self):
        if self.state.flag_screen_size:
            self.prepare_framebuff()
            self.state.flag_screen_size = False

    def check_and_update(self):
        self.update_texture()
        self.update_screen_size()
        self.refresh()

    def refresh(self):
        for plane, refresh in enumerate(self.state.need_refresh):
            if not refresh:
                continue
            glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffers[plane])
            glViewport(0, 0, self.state.viewport[2], self.state.viewport[3])
            if plane == PLANE_3D:
                shader = self.shader_3d
                with Program(shader):
                    self.update_shader_3d()
                    with VAO(self.vao):
                        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            else:
                shader = self.shader_plane
                with Program(shader):
                    self.update_shader_plane()
                    glUniform1i(self.ptrs_plane["plane"], plane)
                    with VAO(self.vao):
                        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            self.state.need_refresh[plane] = False

    def draw_panel(self, plane_type):
        with Program(self.shader_foreground):
            glUniform1i(self.ptrs_foreground["sense"], plane_type)
            with VAO(self.vao):
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    def do_render(self):
        self.check_and_update()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClear(GL_COLOR_BUFFER_BIT)
        if not self.state.file_opened:
            return
        w = self.state.viewport[2]
        h = self.state.viewport[3]
        if self.state.view_type == PlANE_ALL:
            for i, tp in enumerate(
                [PLANE_CORONAL, PLANE_3D, PLANE_CROSS, PLANE_SAGITTAL]
            ):
                a, b = i % 2, i // 2
                glViewport(a * w // 2, b * h // 2, w // 2, h // 2)
                self.draw_panel(tp)
        else:
            glViewport(0, 0, w, h)
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
                "/Users/yeruo/WorkSpace/Datasets/CTCSF/1.25mm/img/20230109000675.nii.gz"
                # "/Users/yeruo/WorkSpace/CTCSF/5.0mm/img/20230102000345.nii.gz"
            )
            self.update_texture()
        while glfw.window_should_close(self.window) == 0:
            self.render_loop()
        glfw.terminate()


if __name__ == "__main__":
    render = Render()
    render.run()
