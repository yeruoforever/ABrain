import os
import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
from tkinter import Tk
import tkinter.filedialog as filedialog

from status import *

__all__ = ["GUI"]

tk_root = Tk()
tk_root.withdraw()


# debug
import numpy as np
import glm

MSG_BOX = 285


class GUI(object):
    def __init__(self, window, status: Status) -> None:
        self.window = window
        imgui.create_context()
        self.impl = GlfwRenderer(window)
        self.state = status
        self.mouse_pos_last = None
        glfw.set_mouse_button_callback(window, self.mouse_viewport_click)
        glfw.set_cursor_pos_callback(window, self.mouse_viewport_move)
        glfw.set_framebuffer_size_callback(window, self.resize_window)
        glfw.set_scroll_callback(window, self.mouse_scroll)
        win_w, win_h = glfw.get_window_size(window)
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        font_scaling_factor = max(float(fb_w) / win_w, float(fb_h) / win_h)
        self.io = imgui.get_io()
        self.font_characters = self.io.fonts.add_font_from_file_ttf(
            os.path.join(os.path.dirname(__file__), "fonts", "楷体_GB2312.ttf"),
            13 * font_scaling_factor,
            None,
            self.io.fonts.get_glyph_ranges_chinese_full(),
        )
        self.impl.refresh_font_texture()
        self.io.font_global_scale /= font_scaling_factor

    def __enter__(self):
        self.impl.process_inputs()

    def __exit__(self, exc_type, exc_value, traceback):
        imgui.new_frame()
        self.draw_and_update_status()
        imgui.render()
        self.impl.render(imgui.get_draw_data())

    def resize_window(self, window, width, height):
        self.state.screen_size[0] = width
        self.state.screen_size[1] = height
        self.state.viewport[2] = width - MSG_BOX
        self.state.viewport[3] = height

    def mouse_location(self):
        if self.state.view_type == PlANE_ALL:
            x, y = self.state.mouse_pos
            y = self.state.viewport[3] - y
            W, H = self.state.viewport[2], self.state.viewport[3]
            hW, hH = W // 2, H // 2
            if x < hW:
                if y < hH:
                    return PLANE_CORONAL
                elif y < H:
                    return PLANE_CROSS
                return PLANE_UNKNOWN
            elif x < W:
                if y < hH:
                    return PLANE_3D
                elif y < H:
                    return PLANE_SAGITTAL
                return PLANE_UNKNOWN
        else:
            return self.state.view_type

    def mouse_viewport_click(self, window, button, action, mode):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.state.mouse_activate = self.mouse_location()
                self.mouse_pos_last = [self.state.mouse_pos[0], self.state.mouse_pos[1]]
                self.state.mouse_state[0] = True
            elif action == glfw.RELEASE:
                self.state.mouse_state[0] = False
        if button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                self.state.mouse_state[1] = True
            elif action == glfw.RELEASE:
                self.state.mouse_state[1] = False

    def mouse_viewport_move(self, window, xpos, ypos):
        if self.state.view_type == PlANE_ALL:
            scale = 2
        else:
            scale = 1
        if self.state.mouse_state[0]:
            delta_x = xpos - self.state.mouse_pos[0]
            delta_y = ypos - self.state.mouse_pos[1]
            delta_x *= self.state.plane_scale * scale
            delta_y *= self.state.plane_scale * scale
            if self.state.mouse_activate == PLANE_CROSS:
                self.state.plane_focus[0] -= delta_x
                self.state.plane_focus[1] += delta_y
            elif self.state.mouse_activate == PLANE_CORONAL:
                self.state.plane_focus[0] -= delta_x
                self.state.plane_focus[2] += delta_y
            elif self.state.mouse_activate == PLANE_SAGITTAL:
                self.state.plane_focus[1] -= delta_x
                self.state.plane_focus[2] += delta_y

        self.state.mouse_pos[0] = xpos
        self.state.mouse_pos[1] = ypos

    def mouse_scroll(self, window, xoffset, yoffset):
        self.state.plane_scale += self.state.mouse_scroll_speed * yoffset
        self.state.plane_scale = min(self.state.plane_scale, 3)
        self.state.plane_scale = max(self.state.plane_scale, 0.01)

    def main_menu_bar(self):
        changed = False
        with imgui.begin_main_menu_bar() as main_bar:
            if main_bar.opened:
                with imgui.begin_menu("文件", True) as file_menu:
                    if file_menu.opened:
                        clicked, state = imgui.menu_item("打开", "Ctrl+O", False, True)
                        if clicked:
                            self.state.input_file = filedialog.askopenfilename()
                        if imgui.menu_item(f"关闭", "Cmd+Q", False, True)[0]:
                            glfw.set_window_should_close(self.window, 1)

    def patient_message(self):
        with imgui.begin(
            "Patient",
            flags=imgui.WINDOW_NO_TITLE_BAR
            # | imgui.WINDOW_NO_RESIZE
            # | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            # | imgui.WINDOW_NO_MOVE,
        ):
            t = self.state.frame_time_step() + 1e-4
            msg = "%3.2f" % (1 / t)
            imgui.text(
                f"{self.state.patient_id}\t{self.state.patient_name}\t{self.state.patient_gender}\t{self.state.patient_age}"
            )
            imgui.text(f"体重    \t:\t{self.state.patient_weight}")
            imgui.text(f"扫描区域\t:\t{self.state.img_body_range}")
            imgui.text(f"图像模态\t:\t{self.state.img_modalty}")
            imgui.text(f"图像大小 (pixel):\t{self.state.img_shape}")
            imgui.text(f"图像范围 (mm):\t{self.state.img_region}")
            imgui.text(f"像素间距 (mm):\t{self.state.img_spacing}")
            imgui.text(f"渲染帧率 (fps):\t{msg.rjust(10)}")

    def camera_control(self):
        with imgui.begin(
            "Camera",
            flags=imgui.WINDOW_NO_TITLE_BAR
            # | imgui.WINDOW_NO_RESIZE
            # | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            # | imgui.WINDOW_NO_MOVE,
        ):
            has_changed = False

            # changed, self.state.view_near = imgui.slider_float(
            #     "Near",
            #     self.state.view_near,
            #     min_value=-1000,
            #     max_value=1000,
            #     format="%.2f",
            # )
            # has_changed = has_changed or changed
            # changed, self.state.view_far = imgui.slider_float(
            #     "Far",
            #     self.state.view_far,
            #     min_value=-1000,
            #     max_value=1000,
            #     format="%.2f",
            # )
            # has_changed = has_changed or changed
            clicked, current = imgui.combo(
                "",
                self.state.view_type,
                ["速览", "横断面", "冠状面", "矢状面", "三维"],
            )
            if clicked and current != self.state.view_type:
                self.state.view_type = current
            imgui.same_line(spacing=1)
            if imgui.button("重设3D相机"):
                self.state.reset_camera()
            changed, self.state.view_radians = imgui.slider_angle(
                "视野大小", self.state.view_radians, 0.0, 180.0
            )
            has_changed = has_changed or changed

            for i, axis in enumerate(["X", "Y", "Z"]):
                changed, self.state.camera_origin[i] = imgui.slider_float(
                    f"视点位置{axis}",
                    self.state.camera_origin[i],
                    min_value=-1000,
                    max_value=1000,
                    format="%.2f",
                )
                has_changed = has_changed or changed

            L = max(*self.state.img_region) / 2
            changed, values = imgui.slider_float3("焦点", *self.state.plane_focus, -L, L)
            if changed:
                self.state.plane_focus[0] = values[0]
                self.state.plane_focus[1] = values[1]
                self.state.plane_focus[2] = values[2]

            for i, axis in enumerate(["右", "前", "上"]):
                changed, self.state.plane_slice[i] = imgui.slider_float(
                    f"切片位置({axis})",
                    self.state.plane_slice[i],
                    min_value=-self.state.img_region[i] / 2,
                    max_value=self.state.img_region[i] / 2,
                    format="%.2f",
                )
                has_changed = has_changed or changed
            changed, self.state.plane_scale = imgui.slider_float(
                "缩放",
                self.state.plane_scale,
                0.0,
                1.5,
            )

    def voxel_setting(self):
        with imgui.begin("voxel"):
            has_changed = False
            changed, values = imgui.slider_float2(
                "体素值域",
                self.state.voxel_min,
                self.state.voxel_max,
                -2048.0,
                +2048.0,
            )
            if changed:
                self.state.voxel_min = values[0]
                self.state.voxel_max = values[1]

    def image_color_setting(self):
        with imgui.begin(
            "Camera",
            flags=imgui.WINDOW_NO_TITLE_BAR
            # | imgui.WINDOW_NO_RESIZE
            # | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            # | imgui.WINDOW_NO_MOVE,
        ):
            has_changed = False

    def test_plane_coord(self, pos_screen, screen, focus, slice, range):
        range = range + glm.vec3(0.000001)
        cro = (2 * focus.xy + screen * pos_screen) / (2 * range.xy) + 0.5
        cro = glm.vec3(cro.x, cro.y, slice.z / range.z + 0.5)
        con = (2 * focus.xz + screen * pos_screen) / (2 * range.xz) + 0.5
        con = glm.vec3(con.x, slice.y / range.y + 0.5, con.y)
        sag = (2 * focus.yz + screen * pos_screen) / (2 * range.yz) + 0.5
        sag = glm.vec3(slice.x / range.x + 0.5, sag.x, sag.y)
        imgui.text(f"横断：{cro}")
        imgui.text(f"冠状：{con}")
        imgui.text(f"矢状：{sag}")

    def debug_plane(self):
        with imgui.begin("plane"):
            W, H = self.state.viewport[2], self.state.viewport[3]
            ndc = [
                (self.state.mouse_pos[0] / W) * 2 - 1,
                1 - (self.state.mouse_pos[1] / H) * 2,
            ]
            imgui.text(f"标准屏幕坐标:  {ndc}")
            self.test_plane_coord(
                glm.vec2(*ndc),
                glm.vec2(W * self.state.plane_scale, H * self.state.plane_scale),
                glm.vec3(*self.state.plane_focus),
                glm.vec3(*self.state.plane_slice),
                glm.vec3(*self.state.img_region),
            )
        pass

    def debug_3d(self):
        with imgui.begin("vertex"):
            W, H = self.state.viewport[2], self.state.viewport[3]
            V = self.state.mat_W2V()
            Vi = self.state.mat_V2W()
            aspect = W / H
            ndc = [
                (self.state.mouse_pos[0] / W) * 2 - 1,
                1 - (self.state.mouse_pos[1] / H) * 2,
            ]
            imgui.text(f"标准屏幕坐标:  {ndc}")
            X = aspect * ndc[0]
            Y = ndc[1]
            dis = glm.tan(self.state.view_radians / 2)
            ray = -Vi * glm.vec4(X, Y, -1 / dis, 1.0)
            ray = glm.normalize(ray.xyz)
            imgui.text(f"屏幕大小: {self.state.screen_size}")
            imgui.text(f"视口大小:  {self.state.viewport}")
            imgui.text(f"纵横比:  {aspect}")
            imgui.text(f"视野宽度:  {self.state.view_radians*180/np.pi}")
            # imgui.text(f"Plane:  Near({self.state.view_near})")
            # imgui.text(f"Plane:  Far({self.state.view_far})")
            imgui.text(f"相机位置:  {self.state.camera_origin}")
            imgui.text(f"相机视野坐标:  {V*glm.vec3(*self.state.camera_origin)}")
            imgui.text(f"世界原点视野坐标:  {V*glm.vec3(0)}")
            imgui.text(f"转换矩阵（世界到视野）:\n{V}")
            imgui.text(f"转换矩阵（视野到世界）:\n{Vi}")
            imgui.text(f"光线方向: {ray}")
        with imgui.begin("fragment"):
            imgui.text(f"光线步长:  {self.state.ray_step}")
            imgui.text(f"光线不透明度:  {self.state.ray_alpha}")
            pix_min, pix_max = self.state.voxel_min, self.state.voxel_max
            pix_window = pix_max - pix_min
            imgui.text(f"体素亮度范围:  {pix_min,pix_max}")
            imgui.text(f"体素亮度窗口大小:  {pix_window}")
            imgui.text(f"转换矩阵（模型到世界）\n{self.state.mat_M2W()}")
            imgui.text(f"转换矩阵（世界到模型）\n{self.state.mat_W2M()}")
            bbox = self.state.cube_bounding()
            eye = glm.vec3(*self.state.camera_origin)
            imgui.text(f"图像边界1:  {bbox[0]}")
            imgui.text(f"图像边界2:  {bbox[1]}")
            slab = is_insersect(eye, ray, bbox[0], bbox[1])
            W2M = self.state.mat_W2M()
            imgui.text(f"纹理边界1: {W2M*bbox[0]}")
            imgui.text(f"纹理边界2: {W2M*bbox[1]}")
            flag, t_min, t_max = slab
            step_min = t_min * ray
            step_max = t_max * ray
            imgui.text(f"光线起止:  {t_max-t_min}")
            imgui.text(f"光线与模型相交:  {flag}")
            imgui.text(f"光线起点t:  {t_min}")
            imgui.text(f"光线终点t:  {t_max}")
            imgui.text(f"光线起点坐标:  {eye+step_min}")
            imgui.text(f"光线终点坐标:  {eye+step_max}")

    def draw_and_update_status(self):
        with imgui.font(self.font_characters):
            self.main_menu_bar()
            self.patient_message()
            self.camera_control()
            self.image_color_setting()
            self.voxel_setting()
            self.debug_plane()
            self.debug_3d()


def print_mat4(mat):
    pass


def is_insersect(origin, light, cube_a, cube_b):
    a = (cube_a - origin) / light
    b = (cube_b - origin) / light

    t_min = max(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z))
    t_max = min(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z))

    flag = t_min >= 0 and t_min <= t_max

    return flag, t_min, t_max
