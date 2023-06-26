import time
from queue import Queue
from typing import *
import nibabel as nib
import ctypes

import numpy as np
import torch
import torch.multiprocessing as mp
from glm import *
from glm import scale, translate

__all__ = [
    "Status",
    "PlANE_ALL",
    "PLANE_CROSS",
    "PLANE_CORONAL",
    "PLANE_SAGITTAL",
    "PLANE_3D",
    "PLANE_UNKNOWN",
]

MAX_HISTORY_LENGTH = 6
PLANE_CROSS = 0
PLANE_CORONAL = 1
PLANE_SAGITTAL = 2
PLANE_3D = 3
PlANE_ALL = 4
PLANE_UNKNOWN = -1


class Status(object):
    def __init__(self) -> None:
        self.segment_progress = mp.Manager().Value(ctypes.c_float, 0.0)
        self.segment_process = []
        self.segment_finished = mp.Event()
        self.segment_finished.clear()
        self.segment_queue = mp.Queue(maxsize=2)
        self.has_seg = False

        self.csf_volume = -1.0
        self.csf_cnt = mp.Manager().Value(ctypes.c_int64, 0)

        self.screen_size = ivec2(1280, 720)
        self.viewport = ivec4(0, 0, 995, 720)
        self.view_type = PlANE_ALL

        self.flag_screen_size = False
        self.need_refresh = [True] * PlANE_ALL

        self.mouse_pos = ivec2(0, 0)
        self.mouse_state = bvec2(False, False)
        self.mouse_activate = PLANE_UNKNOWN
        self.mouse_scroll_speed = 0.05

        self.key_status = [False] * 1024

        self.camera_origin = vec3(0.0, 0.0, 5.0)
        self.camera_target = vec3(0.0, 0.0, 0.0)
        self.camera_up = vec3(0.0, 1.0, 0.0)
        self.camera_update_lookat()
        self.view_radians = 3.141592 / 4.0

        self.ray_step = 0.01
        self.ray_alpha = 0.02

        self.voxel_min = 0.0
        self.voxel_max = 100.0
        self.voxel_window = 100.0

        self.color_overlap = 0.5
        self.color_background = vec3(0.0, 0.0, 0.0)
        self.color_target_1 = vec3(0.0, 1.0, 0.0)
        self.color_target_2 = vec3(0.0, 0.0, 1.0)

        self.input_file = ""
        self.recents = Queue(MAX_HISTORY_LENGTH)
        self.file_opened = False

        self.img_spacing = vec3(0.0, 0.0, 0.0)
        self.img_shape = ivec3(0, 0, 0)
        self.img_modalty = "CT"
        self.img_region = vec3(0.0, 0.0, 0.0)
        self.img_body_range = "Head (Brain)"

        self.plane_focus = vec3(0.0, 0.0, 0.0)
        self.plane_scale = 0.30
        self.plane_slice = vec3(0.0, 0.0, 0.0)

        self.patient_id = "未加载"
        self.patient_name = "--"
        self.patient_age = "--"
        self.patient_gender = "-"
        self.patient_weight = "-- kg"

        self.frame_timestamp = time.time()

        self.M2M = mat4()
        self.M2W = mat4()
        self.W2M = mat4()

    def get_seg(self):
        seg = self.segment_queue.get()
        seg = nib.load(seg)
        seg = np.array(seg.dataobj, dtype=np.float32)
        self.segment_finished.clear()
        return seg

    def set_refresh_all(self):
        for i in range(PlANE_ALL):
            self.need_refresh[i] = True

    def set_refresh_plane(self):
        for i in range(PlANE_ALL - 1):
            self.need_refresh[i] = True

    def reset_focus(self):
        self.plane_focus = vec3(0)

    def reset_slice(self):
        self.plane_slice = vec3(0)

    def check_and_set_refresh(self, plane, flag):
        self.need_refresh[plane] = self.need_refresh[plane] or flag

    def update_img_meta(self, spacing, shape, modalty):
        self.img_spacing = vec3(*spacing)
        self.img_shape = ivec3(*shape)
        self.modalty = modalty
        region = np.array(spacing) * np.array(shape)
        self.img_region = vec3(*region)
        self.update_aspact()
        self.update_M2W()

    def update_patient_meta(self, pid, name, age, gender, weight):
        self.patient_id = pid
        self.patient_name = name
        self.patient_age = age
        self.patient_gender = gender
        self.patient_weight = weight

    def reset_camera(self):
        self.camera_origin = vec3(0.0, 0.0, 5.0)
        self.camera_target = vec3(0.0, 0.0, 0.0)
        self.camera_update_lookat()
        self.view_radians = 3.141592 / 4.0

    def volume_delta(self):
        return self.img_spacing[0] * self.img_spacing[1] * self.img_spacing[2]

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

    def camera_update_lookat(self):
        """view matrix"""
        self.camera_lookat = lookAt(
            self.camera_origin,
            self.camera_target,
            self.camera_up,
        )
        self.W2V = self.camera_lookat
        self.V2W = inverse(self.W2V)

    def update_aspact(self):
        L = min(*self.img_region)
        aspect = self.img_region / L
        self.img_aspect = vec3(aspect)

    def update_M2W(self):
        m = mat4(1)
        t = translate(m, vec3(-0.5, -0.5, -0.5))
        s = scale(m, vec3(2, 2, 2))
        ss = scale(m, self.img_aspect)
        self.M2W = ss * s * t
        self.W2M = inverse(self.M2W)

    def cube_bounding(self):
        a, b = vec4(0, 0, 0, 1), vec4(1)
        return (self.M2W * a).xyz, (self.M2W * b).xyz

    def __del__(self):
        print("Closing subprocessing...")
        for each in self.segment_process:
            each.terminal()
        for each in self.segment_process:
            each.join()
