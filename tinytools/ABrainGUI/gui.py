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

MSG_BOX = 330


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
        self.state.viewport[1] = width - MSG_BOX
        self.state.viewport[3] = height

    def mouse_viewport_click(self, window, button, action, mode):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            self.mouse_pos_last = [self.state.mouse_pos[0], self.state.mouse_pos[1]]
        # elif button==glfw.mo

    def mouse_viewport_move(self, window, xpos, ypos):
        if self.mouse_pos_last:
            pass
        self.state.mouse_pos[0] = xpos
        self.state.mouse_pos[1] = ypos

    def main_menu_bar(self):
        changed = False
        with imgui.begin_main_menu_bar() as main_bar:
            if main_bar.opened:
                with imgui.begin_menu("File", True) as file_menu:
                    if file_menu.opened:
                        clicked, state = imgui.menu_item(
                            "Open ...", "Ctrl+O", False, True
                        )
                        if clicked:
                            self.state.input_file = filedialog.askopenfilename()
                        if imgui.menu_item(f"Close", "Cmd+Q", False, True)[0]:
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
            msg = "%3.2f fps" % (1 / t)
            imgui.text(f"Patient Name       \t:\t{self.state.patient_name}")
            imgui.text(f"Patient Age        \t:\t{self.state.patient_age}")
            imgui.text(f"Body Region        \t:\t{self.state.img_body_range}")
            imgui.text(f"Image Modality     \t:\t{self.state.img_modalty}")
            imgui.text(f"Image Shape (pixel)\t:\t{self.state.img_shape}")
            imgui.text(f"Image Range (mm)   \t:\t{self.state.img_region}")
            imgui.text(f"Image Spacing (mm) \t:\t{self.state.img_spacing}")
            imgui.text(f"Render FPS         \t:\t{msg.rjust(10)}")

    def camera_control(self):
        with imgui.begin(
            "Camera",
            flags=imgui.WINDOW_NO_TITLE_BAR
            # | imgui.WINDOW_NO_RESIZE
            # | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            # | imgui.WINDOW_NO_MOVE,
        ):
            has_changed = False

            changed, self.state.view_near = imgui.slider_float(
                "Near",
                self.state.view_near,
                min_value=-1000,
                max_value=1000,
                format="%.2f",
            )
            has_changed = has_changed or changed
            changed, self.state.view_far = imgui.slider_float(
                "Far",
                self.state.view_far,
                min_value=-1000,
                max_value=1000,
                format="%.2f",
            )
            has_changed = has_changed or changed

            changed, self.state.view_radians = imgui.slider_angle(
                "FOV", self.state.view_radians, 0.0, 180.0
            )
            has_changed = has_changed or changed

            for i, axis in enumerate(["X", "Y", "Z"]):
                changed, self.state.camera_origin[i] = imgui.slider_float(
                    f"Camera {axis}",
                    self.state.camera_origin[i],
                    min_value=-1000,
                    max_value=1000,
                    format="%.2f",
                )
                has_changed = has_changed or changed

            if imgui.button("Reset Camera"):
                self.state.reset_camera()

    def image_color_setting(self):
        with imgui.begin(
            "Camera",
            flags=imgui.WINDOW_NO_TITLE_BAR
            # | imgui.WINDOW_NO_RESIZE
            # | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            # | imgui.WINDOW_NO_MOVE,
        ):
            has_changed = False

    def debug_view_camera(self):
        with imgui.begin("vertex"):
            W, H = self.state.viewport[2], self.state.viewport[3]
            V = self.state.camera_lookat()
            Vi = glm.inverse(V)
            aspect = W / H
            ndc = [
                (self.state.mouse_pos[0] / W) * 2 - 1,
                (self.state.mouse_pos[1] / H) * 2 - 1,
            ]
            imgui.text(f"gl_Position:  {ndc}")
            Y = np.tan(self.state.view_radians / 2) * self.state.view_near
            X = aspect * Y
            ray = Vi * glm.vec4(X * ndc[0], Y * ndc[1], -self.state.view_near, 1)
            ray = glm.normalize(ray.xyz)
            imgui.text(f"Screen Size:  {self.state.screen_size}")
            imgui.text(f"Viewport:  {self.state.viewport}")
            imgui.text(f"Aspect:  {aspect}")
            imgui.text(f"Plane:  FOV({self.state.view_radians*180/np.pi})")
            imgui.text(f"Plane:  Near({self.state.view_near})")
            imgui.text(f"Plane:  Far({self.state.view_far})")
            imgui.text(f"Camera pos:  {self.state.camera_origin}")
            imgui.text("Bound box:  ")
            imgui.text(f"World to View:\n{V}")
            imgui.text(f"View to World:\n{Vi}")
            imgui.text(f"ray direction:  {ray}")
        with imgui.begin("fragment"):
            imgui.text(f"Ray Step:  {self.state.ray_step}")
            imgui.text(f"Ray alpha:  {self.state.ray_alpha}")
            pix_min, pix_max = self.state.voxel_min, self.state.voxel_max
            pix_window = pix_max - pix_min
            imgui.text(f"Voxel Value Range:  {pix_min,pix_max}")
            imgui.text(f"Voxel value Window:  {pix_window}")
            imgui.text(f"Model to World\n{self.state.mat_M2W()}")
            imgui.text(f"World to Model\n{self.state.mat_W2M()}")
            bbox = self.state.cube_bounding()
            eye = glm.vec3(*self.state.camera_origin)
            imgui.text(f"Image BBox a:  {bbox[0]}")
            imgui.text(f"Image BBox b:  {bbox[1]}")
            slab = is_insersect(eye, ray, bbox[0], bbox[1])
            imgui.text(f"Slab result:  {slab[0]}")
            imgui.text(f"Slab result:  {slab[1]*self.state.ray_step*ray}")
            imgui.text(f"Slab result:  {slab[2]*self.state.ray_step*ray}")

    def draw_and_update_status(self):
        self.main_menu_bar()
        self.patient_message()
        self.camera_control()
        self.image_color_setting()
        self.debug_view_camera()


def print_mat4(mat):
    pass


def is_insersect(origin, light, cube_a, cube_b):
    a = (cube_a - origin) / light
    b = (cube_b - origin) / light

    t_min = max(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z))
    t_max = min(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z))

    flag = t_min >= 0 and t_min <= t_max

    return flag, t_min, t_max
