import sys
from collections import namedtuple

import glfw
import imgui
import numpy as np
from buffer import *
from gui.plane import *
from imgui.integrations.glfw import GlfwRenderer, ProgrammablePipelineRenderer
from niiio import *
from OpenGL.GL import *
from shader import *
from texture import *

ProgramTuple = namedtuple("ProgramTuple", ["prog", "ptr"])


MSG_WIDTH = 330


class ABrainApp(object):
    def __init__(self) -> None:
        self.W = 1280
        self.H = 720
        self.viewport_W = self.W - MSG_WIDTH
        self.viewport_H = self.H
        self.changed = False
        self.init_glfw()

        self.texture_img = -1
        self.texture_seg = -1
        self.program_plane = None
        self.program_3d = None
        self.plane_VAO = None

        self.mouse_button_state = [False, False]
        self.mouse_position = [0.0, 0.0]
        self.mouse_location = PLANE_CORONAL

        self.init_glfw()

    def adjust_viewport_size(self, W, H):
        self.W = W
        self.H = H
        self.viewport_W = W - MSG_WIDTH
        self.viewport_H = H

    def init_glfw(self):
        if glfw.init() == 0:
            exit()

        glfw.default_window_hints()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
        if sys.platform == "darwin":
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

    def change_window_size(self, window, width, height):
        self.changed = True
        # self.W, self.H = glfw.get_framebuffer_size(window)
        self.adjust_viewport_size(width, height)
        self.conf.screen = [self.W * self.conf.scale, self.H * self.conf.scale]

    def mouse_butten_callback(self, window, button, action, mod):
        if button == glfw.MOUSE_BUTTON_LEFT:
            button_id = 0
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            button_id = 1
        else:
            return
        if action == glfw.PRESS:
            self.mouse_button_state[button_id] = True
        elif action == glfw.RELEASE:
            self.mouse_button_state[button_id] = False
        print(self.mouse_button_state)

    def mouse_postion_callback(self, window, xpos, ypos):
        print(self.mouse_position)
        self.mouse_position[0] = xpos
        self.mouse_position[1] = ypos
        if self.conf.plane_type == PlANE_ALL:
            w, h = self.viewport_W, self.viewport_H
            hw, hh = w / 2, h / 2
            if xpos > 0 and xpos < hw:
                if ypos > 0 and ypos < hh:
                    self.mouse_location = PLANE_CROSS
                elif ypos > hh and ypos < h:
                    self.mouse_location = PLANE_CORONAL
                else:
                    self.mouse_location = PlANE_ALL
            elif xpos > hw and xpos < w:
                if ypos > 0 and ypos < hh:
                    self.mouse_location = PLANE_SAGITTAL
                elif ypos > hh and ypos < h:
                    self.mouse_location = PLANE_3D
                else:
                    self.mouse_location = PlANE_ALL
            else:
                self.mouse_location = PlANE_ALL
            self.mouse_location = PlANE_ALL
        else:
            self.mouse_location = self.conf.plane_type

    def set_glfw(self, window):
        glfw.make_context_current(window)
        glfw.set_framebuffer_size_callback(window, self.change_window_size)
        glfw.set_mouse_button_callback(window, self.mouse_butten_callback)
        glfw.set_cursor_pos_callback(window, self.mouse_postion_callback)
        glfw.swap_interval(1)

    def set_opengl(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def get_img_texture(self):
        img, spacing = load_nii(self.conf.input_file)
        texture = volume_texture(img)
        self.conf.shape = list(img.shape)
        self.texture_img = texture
        self.conf.spacing = list(spacing)
        region = np.array(img.shape) * np.array(spacing)
        self.conf.region = region.tolist()
        self.conf.center = (region / 2).tolist()

    def do_diff(self):
        if self.conf.need_to_load():
            print("open", self.conf.input_file)
            self.drop_texture()
            self.get_img_texture()
            self.conf.swap_current_file()
            # TODO  加载Seg
        self.pass_paramter()

    def pass_paramter(self):
        glUseProgram(self.program_plane.prog)
        glUniform1i(self.program_plane.ptr["plane"], self.conf.plane_type)
        glUniform2f(self.program_plane.ptr["hu_range"], *self.conf.hu_range)
        glUniform3f(self.program_plane.ptr["center"], *self.conf.center)
        W, H = self.viewport_W, self.viewport_H
        WH = [W * self.conf.scale, H * self.conf.scale]
        glUniform2f(self.program_plane.ptr["WH"], *WH)
        glUniform3f(self.program_plane.ptr["ABC"], *self.conf.region)

    def create_window(self):
        window = glfw.create_window(self.W, self.H, "ABrain", None, None)
        glfw.make_context_current(window)
        self.window = window
        return window

    def init_imgui(self, window):
        imgui.create_context()
        # io = imgui.get_io()
        # io.display_size = 640, 480
        # io.config_flags |= imgui
        impl = GlfwRenderer(window)
        # return impl, io
        return impl

    def input_events(self):
        glfw.poll_events()
        self.gui.process_inputs()

    def prepare_rander(self):
        imgui.new_frame()
        glClear(GL_COLOR_BUFFER_BIT)

    def load_programs(self):
        program, ptrs = plane_shader()
        self.program_plane = ProgramTuple(prog=program, ptr=ptrs)
        program, ptrs = volume_shader()
        self.program_3d = ProgramTuple(prog=program, ptr=ptrs)

    def prepare_vertex(self):
        self.plane_VAO = plane_buffers()
        pass

    def draw_plane(self, plane_type):
        glBindVertexArray(self.plane_VAO.vao)
        glUseProgram(self.program_plane.prog)
        glBindTexture(GL_TEXTURE_3D, self.texture_img)
        glUniform1i(self.program_plane.ptr["plane"], plane_type)
        glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_INT, ctypes.c_void_p(0))

    def draw_3d(self):
        pass

    def draw_panes(self):
        if self.conf.current_file == "":
            return
        w, h = self.viewport_W, self.viewport_H
        if self.conf.plane_type == PlANE_ALL:
            glViewport(0, 0, w // 2, h // 2)
            self.draw_plane(PLANE_CROSS)
            glViewport(0, h // 2, w // 2, h // 2)
            self.draw_plane(PLANE_CORONAL)
            glViewport(w // 2, 0, w // 2, h // 2)
            self.draw_plane(PLANE_SAGITTAL)
            glViewport(w // 2, h // w, w // 2, h // 2)
            self.draw_3d()
        elif self.conf.plane_type == PLANE_3D:
            glViewport(0, 0, w, h)
            self.draw_3d()
        else:
            glViewport(0, 0, w, h)
            self.draw_plane(self.conf.plane_type)

    def do_render(self):
        imgui.render()
        self.gui.render(imgui.get_draw_data())
        glfw.swap_buffers(self.window)

    def run(self):
        window = self.create_window()
        self.conf = WindowConfig(window)
        self.gui = self.init_imgui(window)

        self.set_glfw(window)
        self.set_opengl()

        self.load_programs()
        self.prepare_vertex()

        self.changed = True
        while glfw.window_should_close(window) == 0:
            self.render_loop()

        self.delete_buffers()
        self.drop_texture()
        glfw.terminate()

    def render_loop(self):
        if self.changed:
            self.do_diff()
            self.changed = False
        self.input_events()
        self.prepare_rander()
        self.draw_panes()
        self.changed = some_settings(self.conf)
        self.do_render()

    def drop_texture(self):
        glDeleteTextures(1, self.texture_img)
        self.texture_img = -1
        glDeleteBuffers(1, self.texture_seg)
        self.texture_seg = -1

    def delete_buffers(self):
        delete_vertex_array(self.plane_VAO)


if __name__ == "__main__":
    app = ABrainApp()
    app.run()
