import numpy as np
import glm
import imgui
from status import *


def print_mat4(mat):
    pass


def is_insersect(origin, light, cube_a, cube_b):
    a = (cube_a - origin) / light
    b = (cube_b - origin) / light

    t_min = max(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z))
    t_max = min(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z))

    flag = t_min >= 0 and t_min <= t_max

    return flag, t_min, t_max


def test_plane_coord(pos_screen, screen, focus, slice, range):
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
        test_plane_coord(
            glm.vec2(*ndc),
            glm.vec2(W * self.state.plane_scale, H * self.state.plane_scale),
            glm.vec3(*self.state.plane_focus),
            glm.vec3(*self.state.plane_slice),
            glm.vec3(*self.state.img_region),
        )
    pass


def debug_state(self):
    changed, self.state.view_radians = imgui.slider_angle(
        "视野大小", self.state.view_radians, 30.0, 90.0
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


def debug_3d(self):
    with imgui.begin("vertex"):
        t = self.state.frame_time_step() + 1e-4
        msg = "%3.2f" % (1 / t)
        imgui.text(f"渲染帧率 (fps):\t{msg.rjust(10)}")
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
        selected = [False, False]
        _, selected[0] = imgui.selectable("1. I am selectable", selected[0])
        _, selected[1] = imgui.selectable("2. I am selectable too", selected[1])
        imgui.text("3. I am not selectable")
