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


def clamp(x, mi, mx):
    return min(max(mi, x), mx)


class GUI(object):
    def __init__(self, window, status: Status) -> None:
        self.window = window
        imgui.create_context()
        self.impl = GlfwRenderer(window)
        self.state = status
        self.mouse_pos_last = glm.vec2(0.0, 0.0)
        self.monitor_scale = glfw.get_monitor_content_scale(glfw.get_primary_monitor())
        self.MSG_BOX = MSG_BOX
        glfw.set_mouse_button_callback(window, self.mouse_viewport_click)
        glfw.set_cursor_pos_callback(window, self.mouse_viewport_move)
        glfw.set_framebuffer_size_callback(window, self.resize_window)
        glfw.set_scroll_callback(window, self.mouse_scroll)
        glfw.set_key_callback(window, self.key_tracking)
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
        screen = glfw.get_primary_monitor()
        sw, sh = glfw.get_monitor_content_scale(screen)
        self.state.screen_size[0] = width
        self.state.screen_size[1] = height
        self.state.viewport[2] = width - self.MSG_BOX * sw
        self.state.viewport[3] = height
        self.state.flag_screen_size = True

    def key_tracking(self, window, key, scancode, action, mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            self.state.key_status[key] = True
        elif action == glfw.RELEASE:
            self.state.key_status[key] = False
        if (
            self.state.key_status[glfw.KEY_LEFT_SHIFT]
            or self.state.key_status[glfw.KEY_RIGHT_SHIFT]
        ):
            self.mouse_pos_last[0] = self.state.mouse_pos[0]
            self.mouse_pos_last[1] = self.state.mouse_pos[1]

    def mouse_location(self):
        x, y = self.state.mouse_pos
        x *= self.monitor_scale[0]
        y *= self.monitor_scale[1]
        y = self.state.viewport[3] - y
        W, H = self.state.viewport[2], self.state.viewport[3]
        if self.state.view_type == PlANE_ALL:
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
            if 0 < x and x < W and 0 < y and y < H:
                return self.state.view_type
            else:
                return PLANE_UNKNOWN

        return PLANE_UNKNOWN

    def mouse_viewport_click(self, window, button, action, mode):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
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
        self.state.mouse_pos[0] = xpos
        self.state.mouse_pos[1] = ypos
        self.state.mouse_activate = self.mouse_location()
        if self.state.mouse_state[0]:
            if self.state.view_type == PlANE_ALL:
                scale = 2
            else:
                scale = 1
            delta_x = xpos - self.mouse_pos_last[0]
            delta_y = ypos - self.mouse_pos_last[1]
            if self.state.mouse_activate == PLANE_UNKNOWN:
                return

            if self.state.mouse_activate == PLANE_CROSS:
                delta_x *= self.state.plane_scale * scale
                delta_y *= self.state.plane_scale * scale
                self.state.plane_focus[0] -= delta_x
                self.state.plane_focus[1] += delta_y
                self.state.set_refresh_plane()
            elif self.state.mouse_activate == PLANE_CORONAL:
                delta_x *= self.state.plane_scale * scale
                delta_y *= self.state.plane_scale * scale
                self.state.plane_focus[0] -= delta_x
                self.state.plane_focus[2] += delta_y
                self.state.set_refresh_plane()

            elif self.state.mouse_activate == PLANE_SAGITTAL:
                delta_x *= self.state.plane_scale * scale
                delta_y *= self.state.plane_scale * scale
                self.state.plane_focus[1] -= delta_x
                self.state.plane_focus[2] += delta_y
                self.state.set_refresh_plane()

            elif self.state.mouse_activate == PLANE_3D:
                mat = self.state.camera_lookat
                rx = glm.rotate(delta_x * 0.03, glm.row(mat, 1).xyz)
                ry = glm.rotate(delta_y * 0.03, glm.row(mat, 0).xyz)
                r = rx * ry
                self.state.camera_up = r * self.state.camera_up
                self.state.camera_origin = r * self.state.camera_origin
                self.state.camera_update_lookat()
                self.state.check_and_set_refresh(PLANE_3D, True)

            self.mouse_pos_last[0] = xpos
            self.mouse_pos_last[1] = ypos

    def mouse_scroll(self, window, xoffset, yoffset):
        if self.state.mouse_activate == PLANE_UNKNOWN:
            return
        if (
            self.state.key_status[glfw.KEY_LEFT_SHIFT]
            or self.state.key_status[glfw.KEY_RIGHT_SHIFT]
        ):
            self.state.plane_scale += self.state.mouse_scroll_speed * yoffset
            self.state.plane_scale = min(self.state.plane_scale, 3)
            self.state.plane_scale = max(self.state.plane_scale, 0.01)
            self.state.set_refresh_plane()
        else:
            self.state.check_and_set_refresh(self.state.mouse_activate, True)
            if self.state.mouse_activate == PLANE_CROSS:
                delta = yoffset * 0.3 + self.state.plane_slice[2]
                bound = self.state.img_region[2] / 2
                self.state.plane_slice[2] = clamp(delta, -bound, bound)
            elif self.state.mouse_activate == PLANE_CORONAL:
                delta = yoffset * 0.3 + self.state.plane_slice[1]
                bound = self.state.img_region[1] / 2
                self.state.plane_slice[1] = clamp(delta, -bound, bound)
            elif self.state.mouse_activate == PLANE_SAGITTAL:
                delta = yoffset * 0.3 + self.state.plane_slice[0]
                bound = self.state.img_region[0] / 2
                self.state.plane_slice[0] = clamp(delta, -bound, bound)

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
                ["横断面", "冠状面", "矢状面", "三维", "速览"],
            )
            if clicked and current != self.state.view_type:
                self.state.view_type = current
            imgui.same_line(spacing=1)
            if imgui.button("重设3D相机"):
                self.state.reset_camera()
                self.state.check_and_set_refresh(PLANE_3D, True)
            changed, self.state.view_radians = imgui.slider_angle(
                "视野大小", self.state.view_radians, 20.0, 160.0
            )
            self.state.check_and_set_refresh(PLANE_3D, changed)

            for i, axis in enumerate(["X", "Y", "Z"]):
                changed, self.state.camera_origin[i] = imgui.slider_float(
                    f"视点位置{axis}",
                    self.state.camera_origin[i],
                    min_value=-1000,
                    max_value=1000,
                    format="%.2f",
                )
                if changed:
                    self.state.camera_update_lookat()
                    self.state.check_and_set_refresh(PLANE_3D, True)

            L = max(*self.state.img_region) / 2
            changed, values = imgui.slider_float3("焦点", *self.state.plane_focus, -L, L)
            if changed:
                self.state.plane_focus[0] = values[0]
                self.state.plane_focus[1] = values[1]
                self.state.plane_focus[2] = values[2]
                self.state.set_refresh_plane()

            for i, axis in enumerate(["右", "前", "上"]):
                changed, self.state.plane_slice[i] = imgui.slider_float(
                    f"切片位置({axis})",
                    self.state.plane_slice[i],
                    min_value=-self.state.img_region[i] / 2,
                    max_value=self.state.img_region[i] / 2,
                    format="%.2f",
                )
                if axis == "右":
                    plane = PLANE_SAGITTAL
                elif axis == "前":
                    plane = PLANE_CORONAL
                else:
                    plane = PLANE_CROSS

                self.state.check_and_set_refresh(plane, changed)

            changed, self.state.plane_scale = imgui.slider_float(
                "缩放",
                self.state.plane_scale,
                0.0,
                1.5,
            )
            if changed:
                self.state.set_refresh_plane()
            changed, self.state.ray_step = imgui.slider_float(
                "光线步长", self.state.ray_step, 0.001, 0.01, format="%.4f"
            )
            self.state.check_and_set_refresh(PLANE_3D, changed)
            changed, self.state.ray_alpha = imgui.slider_float(
                "透明度",
                self.state.ray_alpha,
                0.0001,
                1.0,
            )
            self.state.check_and_set_refresh(PLANE_3D, changed)

    def voxel_setting(self):
        with imgui.begin("voxel", flags=imgui.WINDOW_NO_TITLE_BAR):
            changed, self.state.color_overlap = imgui.slider_float(
                "标注透明度",
                self.state.color_overlap,
                0.0,
                1.0,
            )
            if changed:
                self.state.set_refresh_all()
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
                self.state.set_refresh_all()

            changed, self.state.color_target_1 = imgui.color_edit3(
                "侧脑室", *self.state.color_target_1
            )
            if changed:
                self.state.set_refresh_all()
            changed, self.state.color_target_2 = imgui.color_edit3(
                "三脑室", *self.state.color_target_2
            )
            if changed:
                self.state.set_refresh_all()

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
            V = self.state.W2V
            Vi = self.state.V2W
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
            ray_color = ray * 0.5 + 0.5
            imgui.color_edit3("光线", ray_color.x, ray_color.y, ray_color.z)
        with imgui.begin("fragment"):
            imgui.text(f"光线步长:  {self.state.ray_step}")
            imgui.text(f"光线不透明度:  {self.state.ray_alpha}")
            pix_min, pix_max = self.state.voxel_min, self.state.voxel_max
            pix_window = pix_max - pix_min
            imgui.text(f"体素亮度范围:  {pix_min,pix_max}")
            imgui.text(f"体素亮度窗口大小:  {pix_window}")
            imgui.text(f"转换矩阵（模型到世界）\n{self.state.M2W}")
            imgui.text(f"转换矩阵（世界到模型）\n{self.state.W2M}")
            bbox = self.state.cube_bounding()
            eye = glm.vec3(*self.state.camera_origin)
            imgui.text(f"图像边界1:  {bbox[0]}")
            imgui.text(f"图像边界2:  {bbox[1]}")
            slab = is_insersect(eye, ray, bbox[0], bbox[1])
            W2M = self.state.W2M
            imgui.text(f"纹理边界1: {W2M*bbox[0]}")
            imgui.text(f"纹理边界2: {W2M*bbox[1]}")
            imgui.text(f"纹理边界3： {W2M*glm.vec4(0)}")
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
            imgui.set_next_window_size(MSG_BOX, 150)
            imgui.set_next_window_position(
                self.state.viewport[2] / self.monitor_scale[0],
                20,
            )
            self.patient_message()
            imgui.set_next_window_size(MSG_BOX, 295)
            imgui.set_next_window_position(
                self.state.viewport[2] / self.monitor_scale[0],
                20 + 150,
            )
            self.camera_control()
            imgui.set_next_window_size(MSG_BOX, 105)
            imgui.set_next_window_position(
                self.state.viewport[2] / self.monitor_scale[0],
                20 + 150 + 295,
            )
            self.voxel_setting()
            self.image_color_setting()
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
