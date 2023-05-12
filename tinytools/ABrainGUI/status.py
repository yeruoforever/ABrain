import time
from queue import Queue
from typing import *

from glm import *
from glm import translate, scale
import numpy as np

__all__ = [
    "Status",
    "PlANE_ALL",
    "PLANE_CROSS",
    "PLANE_CORONAL",
    "PLANE_SAGITTAL",
    "PLANE_3D",
]

MAX_HISTORY_LENGTH = 6
PlANE_ALL = 0
PLANE_CROSS = 1
PLANE_CORONAL = 2
PLANE_SAGITTAL = 3
PLANE_3D = 4


class Status(object):
    def __init__(self) -> None:
        self.screen_size = [1280, 720]
        self.viewport = [0, 0, 995, 720]
        self.view_type = PlANE_ALL

        self.mouse_pos = [0, 0]
        self.mouse_state = [False, False]

        self.key_status = [False] * 1024

        self.camera_origin = [4.0, 4.0, 4.0]
        self.camera_target = [0.0, 0.0, 0.0]
        # self.view_far = 100.0
        # self.view_near = 0.1
        self.view_radians = 3.141592 / 4.0

        self.ray_step = 0.001
        self.ray_alpha = 0.18

        self.voxel_min = 0.0
        self.voxel_max = 100.0
        self.voxel_window = 100.0

        self.color_background = [0.0, 0.0, 0.0, 0.0]
        self.color_target_1 = [1.0, 0.0, 0.0, 1.0]
        self.color_target_2 = [0.0, 1.0, 0.0, 1.0]

        self.input_file = ""
        self.recents = Queue(MAX_HISTORY_LENGTH)
        self.file_opened = False

        self.img_spacing = [0.0, 0.0, 0.0]
        self.img_shape = [0, 0, 0]
        self.img_modalty = "CT"
        self.img_region = [0.0, 0.0, 0.0]
        self.img_body_range = "Head (Brain)"

        self.plane_focus = [0.0, 0.0, 0.0]
        self.plane_scale = 0.85
        self.plane_slice = [0.0, 0.0, 0.0]

        self.patient_id = "CT20220222"
        self.patient_name = "蔡徐坤"
        self.patient_age = "38"
        self.patient_gender = "女"
        self.patient_weight = "75 kg"

        self.frame_timestamp = time.time()

    def update_img_meta(self, spacing, shape, modalty):
        self.img_spacing = list(spacing)
        self.img_shape = list(shape)
        self.modalty = modalty
        region = np.array(spacing) * np.array(shape)
        self.img_region = region.tolist()

    def update_patient_meta(self, pid, name, age, gender, weight):
        self.patient_id = pid
        self.patient_name = name
        self.patient_age = age
        self.patient_gender = gender
        self.patient_weight = weight

    def reset_camera(self):
        self.camera_origin = [4.0, 4.0, 4.0]
        self.camera_target = [0.0, 0.0, 0.0]
        self.view_far = 100.0
        self.view_near = 0.1
        self.view_radians = 3.141592 / 4.0

    def frame_time_step(self):
        t = time.time()
        delta = t - self.frame_timestamp
        self.frame_timestamp = t
        return delta

    def need_reload_file(self):
        if len(self.input_file) == 0:
            return False
        return self.input_file != self.current_file()

    def current_file(self) -> str:
        if self.recents.empty():
            return ""
        return self.recents.queue[-1]

    def update_file_history(self):
        if self.recents.full():
            self.recents.get()
        self.recents.put(self.input_file)
        self.input_file = ""
        self.file_opened = True

    def camera_lookat(self):
        """view matrix"""
        return lookAt(
            vec3(*self.camera_origin),
            vec3(*self.camera_target),
            vec3(0.0, 1.0, 0.0),
        )

    # def projection(self):
    #     return perspective(
    #         self.view_radians,
    #         self.viewport[2] / self.viewport[3],
    #         self.view_near,
    #         self.view_far,
    #     )

    def mat_W2V(self):
        return self.camera_lookat()

    def mat_V2W(self):
        return inverse(self.camera_lookat())

    def img_aspact(self):
        L = min(*self.img_region)
        aspect = np.array(self.img_region) / L
        return aspect.tolist()

    def mat_M2W(self):
        m = mat4(1)
        t = translate(m, vec3(-0.5, -0.5, -0.5))
        s = scale(m, vec3(2, 2, 2))
        ss = scale(m, vec3(*self.img_aspact()))
        return ss * s * t

    def mat_W2M(self):
        return inverse(self.mat_M2W())

    def cube_bounding(self):
        a, b = vec4(0, 0, 0, 1), vec4(1)
        m2w = self.mat_M2W()
        return (m2w * a).xyz, (m2w * b).xyz
