import os
import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
from tkinter import Tk
import tkinter.filedialog as filedialog
import glm

from status import *

__all__ = ["GUI"]

tk_root = Tk()
tk_root.withdraw()

DEBUG = False

if DEBUG:
    # debug
    from debug import *


MSG_BOX = 285


def clamp(x, mi, mx):
    return min(max(mi, x), mx)


def print3i(t: tuple):
    return "%.0f, %.0f, %.0f" % (t[0], t[1], t[2])


def print3f(t: tuple):
    return "%.2f, %.2f, %.2f" % (t[0], t[1], t[2])


class GUI(object):
    def __init__(self, window, status: Status) -> None:
        self.window = window
        imgui.create_context()
        self.impl = GlfwRenderer(window)
        self.state = status
        self.MSG_BOX = MSG_BOX
        self.mouse_pos_last = glm.vec2(0.0, 0.0)
        self.monitor_scale = glfw.get_window_content_scale(window)
        w, h = glfw.get_window_size(self.window)
        self.state.screen_size[0] = w
        self.state.screen_size[1] = h
        self.state.viewport[2] = (w - self.MSG_BOX) * self.monitor_scale[0]
        self.state.viewport[3] = h * self.monitor_scale[1]
        glfw.set_mouse_button_callback(window, self.mouse_viewport_click)
        glfw.set_cursor_pos_callback(window, self.mouse_viewport_move)
        glfw.set_framebuffer_size_callback(window, self.resize_window)
        glfw.set_scroll_callback(window, self.mouse_scroll)
        glfw.set_key_callback(window, self.key_tracking)
        font_scaling_factor = max(*self.monitor_scale)
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
        self.monitor_scale = glfw.get_window_content_scale(window)
        self.state.screen_size[0] = width
        self.state.screen_size[1] = height
        self.state.viewport[2] = width - self.MSG_BOX * self.monitor_scale[0]
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
        if self.state.mouse_activate == PLANE_UNKNOWN:
            return
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
            if self.state.mouse_activate == PLANE_3D:
                delta = self.state.view_radians - yoffset * 0.03
                self.state.view_radians = clamp(delta, 3.1415 / 6, 3.1415 / 2)
                self.state.check_and_set_refresh(PLANE_3D, True)
            else:
                self.state.plane_scale += self.state.mouse_scroll_speed * yoffset
                self.state.plane_scale = min(self.state.plane_scale, 0.8)
                self.state.plane_scale = max(self.state.plane_scale, 0.01)
                self.state.set_refresh_plane()
        elif (
            self.state.key_status[glfw.KEY_LEFT_CONTROL]
            or self.state.key_status[glfw.KEY_RIGHT_CONTROL]
        ):
            if self.state.mouse_activate == PLANE_3D:
                delta = yoffset * 0.0001 + self.state.ray_step
                self.state.ray_step = clamp(delta, 0.0001, 0.03)
                self.state.check_and_set_refresh(PLANE_3D, True)
        else:
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
            elif self.state.mouse_activate == PLANE_3D:
                delta = yoffset * 0.001 + self.state.ray_alpha
                self.state.ray_alpha = clamp(delta, 0.0001, 1)
            self.state.check_and_set_refresh(self.state.mouse_activate, True)

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
            flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE
            # | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            # | imgui.WINDOW_NO_MOVE,
        ):
            imgui.columns(4)
            imgui.text(f"{self.state.patient_id}")
            imgui.next_column()
            imgui.text(f"{self.state.patient_name}")
            imgui.next_column()
            imgui.text(f"{self.state.patient_gender}")
            imgui.next_column()
            imgui.text(f"{self.state.patient_age}岁")
            imgui.columns(1)

            imgui.text(f"体重    \t:\t{self.state.patient_weight}")
            imgui.text(f"扫描区域\t:\t{self.state.img_body_range}")
            imgui.text(f"图像模态\t:\t{self.state.img_modalty}")
            imgui.text(f"图像大小 (pixel):\t{print3i(self.state.img_shape)}")
            imgui.text(f"图像范围 (mm):\t{print3i(self.state.img_region)}")
            imgui.text(f"像素间距 (mm):\t{print3f(self.state.img_spacing)}")

    def camera_control(self):
        with imgui.begin(
            "Camera",
            flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE
            # | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            # | imgui.WINDOW_NO_MOVE,
        ):
            clicked, current = imgui.combo(
                "",
                self.state.view_type,
                ["横断面", "冠状面", "矢状面", "三维", "速览"],
            )
            if clicked and current != self.state.view_type:
                self.state.view_type = current
            if imgui.button("重置透视"):
                self.state.reset_camera()
                self.state.check_and_set_refresh(PLANE_3D, True)
            imgui.same_line(spacing=2)

            if imgui.button("切片居中"):
                self.state.reset_focus()
                self.state.set_refresh_plane()
            imgui.same_line(spacing=2)

            if imgui.button("重置三视"):
                self.state.reset_camera()
                self.state.reset_slice()
                self.state.set_refresh_plane()

            changed, self.state.color_overlap = imgui.slider_float(
                "标注透明度",
                self.state.color_overlap,
                0.0,
                1.0,
            )
            if changed:
                self.state.set_refresh_all()
            changed, self.state.ray_alpha = imgui.slider_float(
                "组织透明度",
                self.state.ray_alpha,
                0.0001,
                1.0,
            )
            self.state.check_and_set_refresh(PLANE_3D, changed)
            changed, self.state.ray_step = imgui.slider_float(
                "光线步长", self.state.ray_step, 0.001, 0.01, format="%.4f"
            )
            self.state.check_and_set_refresh(PLANE_3D, changed)
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

    def segmentation(self):
        with imgui.begin(
            "voxel", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE
        ):
            changed, self.state.color_target_1 = imgui.color_edit3(
                "双侧脑室", *self.state.color_target_1
            )
            if changed:
                self.state.set_refresh_all()
            changed, self.state.color_target_2 = imgui.color_edit3(
                "第三脑室", *self.state.color_target_2
            )
            if changed:
                self.state.set_refresh_all()
            percentage = float(self.state.segment_progress.value)
            imgui.progress_bar(
                percentage,
                (self.MSG_BOX - 123, 20),
                "%.2f" % (percentage * 100.0) + "%",
            )
            imgui.same_line(spacing=2)
            if self.state.csf_volume >= 0:
                imgui.text("测量完成")
            else:
                imgui.text("等待测量")
            if imgui.button("自动测量"):
                self.state.csf_volume = -1.0
                self.state.segment_need_start.set()
            imgui.same_line(spacing=3)
            if imgui.button("加载"):
                seg_file = filedialog.askopenfilename()
                if seg_file != "":
                    self.state.segment_queue.put(seg_file)
                    self.state.segment_finished.set()
            imgui.same_line(spacing=3)
            if imgui.button("保存结果"):
                saveas = filedialog.asksaveasfilename()
                # TODO 保存结果
                print(saveas)

    def measuring_result(self):
        with imgui.begin(
            "result",
            flags=imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_NO_BACKGROUND
            | imgui.WINDOW_NO_RESIZE
            # | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            | imgui.WINDOW_NO_MOVE,
        ):
            has_changed = False
            msg = (
                "侧脑室脑脊液： 等待测量"
                if self.state.csf_volume < 0
                else "侧脑室脑脊液： %.2f mL" % self.state.csf_volume
            )
            imgui.text(msg)

    def draw_and_update_status(self):
        with imgui.font(self.font_characters):
            self.main_menu_bar()
            imgui.set_next_window_size(MSG_BOX, 150)
            imgui.set_next_window_position(
                self.state.viewport[2] / self.monitor_scale[0],
                20,
            )
            self.patient_message()
            imgui.set_next_window_size(MSG_BOX, 150)
            imgui.set_next_window_position(
                self.state.viewport[2] / self.monitor_scale[0],
                20 + 150,
            )
            self.camera_control()
            imgui.set_next_window_size(MSG_BOX, 105)
            imgui.set_next_window_position(
                self.state.viewport[2] / self.monitor_scale[0],
                20 + 150 + 150,
            )
            self.segmentation()
            imgui.set_next_window_size(160, 30)
            imgui.set_next_window_position(
                self.state.viewport[2] / self.monitor_scale[0] - 160,
                self.state.viewport[3] / self.monitor_scale[1] - 45,
            )
            self.measuring_result()
            if DEBUG:
                debug_plane(self)
                debug_3d(self)
