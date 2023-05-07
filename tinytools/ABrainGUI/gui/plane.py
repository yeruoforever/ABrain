import imgui
import glfw
import time
import tkinter.filedialog as filedialog
import tkinter as tk
from OpenGL.GL import *
from collections import namedtuple

__all__ = [
    "some_settings",
    "WindowConfig",
    "PlANE_ALL",
    "PLANE_CROSS",
    "PLANE_CORONAL",
    "PLANE_SAGITTAL",
    "PLANE_3D",
]

tk_root = tk.Tk()
tk_root.withdraw()

PlANE_ALL = 0
PLANE_CROSS = 1
PLANE_CORONAL = 2
PLANE_SAGITTAL = 3
PLANE_3D = 4


class WindowConfig(object):
    def __init__(self, window) -> None:
        self.window = window
        self.reset()

    def reset(self):
        self.start = time.time()

        self.input_file = ""
        self.current_file = ""

        self.center = [0.0, 0.0, 0.0]
        self.hu_range = [0.0, 100.0]
        self.shape = [1, 1, 1]
        self.hu_window = 100.0
        self.scale = 0.85
        self.spacing = [0.0, 0.0, 0.0]
        self.region = [0.0, 0.0, 0.0]

        self.plane_type = PlANE_ALL

    def step(self):
        before = self.start
        self.start = time.time()
        return self.start - before

    def swap_current_file(self):
        self.current_file = self.input_file
        self.input_file = ""

    def need_to_load(self):
        return self.input_file != "" and self.input_file != self.current_file


def change_v2(target: list, values):
    target[0] = values[0]
    target[1] = values[1]


def change_v3(target: list, values):
    target[0] = values[0]
    target[1] = values[1]
    target[2] = values[2]


def change_v4(target: list, values):
    target[0] = values[0]
    target[1] = values[1]
    target[2] = values[2]
    target[3] = values[3]


def main_menu_bar(conf: WindowConfig):
    changed = False
    with imgui.begin_main_menu_bar() as main_bar:
        if main_bar.opened:
            with imgui.begin_menu("File", True) as file_menu:
                if file_menu.opened:
                    clicked, state = imgui.menu_item("Open ...", "Ctrl+O", False, True)
                    if clicked:
                        conf.input_file = filedialog.askopenfilename()
                        changed = conf.input_file != conf.current_file
                    if imgui.menu_item(f"Close", "Cmd+Q", False, True)[0]:
                        glfw.set_window_should_close(conf.window, 1)
    return changed


def mri_settings(conf: WindowConfig):
    has_changed = False
    changed, conf.center[0] = imgui.slider_float(
        "Center X", conf.center[0], min_value=0, max_value=conf.region[0], format="%.2f"
    )
    has_changed = has_changed or changed
    changed, conf.center[1] = imgui.slider_float(
        "Center Y", conf.center[1], min_value=0, max_value=conf.region[1], format="%.2f"
    )
    has_changed = has_changed or changed
    changed, conf.center[2] = imgui.slider_float(
        "Center Z", conf.center[2], min_value=0, max_value=conf.region[2], format="%.2f"
    )
    has_changed = has_changed or changed
    changed, conf.scale = imgui.slider_float(
        "Scale", conf.scale, min_value=0.01, max_value=1, format="%.2f"
    )
    has_changed = has_changed or changed

    hu_min, hu_max = conf.hu_range
    hu_window = conf.hu_window

    changed, hu_window = imgui.slider_float(
        "Window (HU)", hu_window, min_value=2, max_value=5000, format="%.0f"
    )

    changed, hu_min = imgui.slider_float(
        "Min (HU)",
        hu_min,
        min_value=-5000,
        max_value=+5000,
    )

    changed, hu_max = imgui.slider_float(
        "Max (HU)",
        hu_max,
        min_value=-5000,
        max_value=+5000,
    )

    changed = (
        hu_min != conf.hu_range[0]
        or hu_max != conf.hu_range[1]
        # or hu_window != conf.hu_window
    )
    if changed:
        hu_min = max(-5000, hu_min)
        hu_max = min(+5000, hu_max)
        if hu_max < hu_min + 1:
            hu_max = hu_min + 1
        hu_window = hu_max - hu_min
        change_v2(conf.hu_range, (hu_min, hu_max))
        conf.hu_window = hu_window
    has_changed = has_changed or changed


def overlay_msg(conf: WindowConfig):
    with imgui.begin(
        "Test",
        flags=imgui.WINDOW_NO_TITLE_BAR
        # | imgui.WINDOW_NO_RESIZE
        # | imgui.WINDOW_ALWAYS_AUTO_RESIZE
        # | imgui.WINDOW_NO_MOVE,
    ):
        t = conf.step()+1e-4
        msg = "%3.2f fps" % (1 / t)
        imgui.text(f"Patient Name       \t:\tXU KUN CAI")
        imgui.text(f"Patient Age        \t:\t38")
        imgui.text(f"Body Region        \t:\tBrain")
        imgui.text(f"Image Modality     \t:\tCT")
        imgui.text(f"Image Shape (pixel)\t:\t512   512   120")
        imgui.text(f"Image Range (mm)   \t:\t80.00 80.00 80.00")
        imgui.text(f"Image Spacing (mm) \t:\t0.625 0.625 1.250")
        imgui.text(f"Render FPS         \t:\t{msg.rjust(10)}")


def camera_msg(conf: WindowConfig):
    with imgui.begin("Camera", flags=imgui.WINDOW_NO_TITLE_BAR):
        clicked, current = imgui.combo(
            "",
            conf.plane_type,
            ["All", "Cross", "Coronal", "Sagittal", "3D"],
        )
        if clicked and current != conf.plane_type:
            conf.plane_type = current
        imgui.same_line(spacing=1)
        imgui.button("Reset Camera")
        return True


def view_select(conf: WindowConfig):
    clicked, current = imgui.combo(
        "View",
        conf.plane_type,
        ["All", "Cross", "Coronal", "Sagittal", "3D"],
    )
    if clicked and current != conf.plane_type:
        conf.plane_type = current
        return True
    return False


def some_settings(conf: WindowConfig):
    has_changed = False
    changed = main_menu_bar(conf)
    has_changed = has_changed or changed
    with imgui.begin("Settings"):
        changed = view_select(conf)
        has_changed = has_changed or changed
        changed = mri_settings(conf)
        has_changed = has_changed or changed
    overlay_msg(conf)
    camera_msg(conf)

    return has_changed, conf
