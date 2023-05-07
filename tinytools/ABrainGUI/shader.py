import os
from typing import *

import OpenGL.GL.shaders as shaders
from OpenGL.GL import *


def read_from_text(path: str):
    with open(path, "r") as io:
        content = io.read()
    return content


def load_shader_from_text(
    target: str,
    vertex_source: Optional[str] = None,
    fragment_source: Optional[str] = None,
):
    if vertex_source is None:
        path = os.path.join(os.path.dirname(__file__), "shader", target, "vertex.vert")
        vertex_source = read_from_text(path)

    if fragment_source is None:
        path = os.path.join(
            os.path.dirname(__file__), "shader", target, "fragment.frag"
        )
        fragment_source = read_from_text(path)

    program = shaders.compileProgram(
        shaders.compileShader(vertex_source, GL_VERTEX_SHADER),
        shaders.compileShader(fragment_source, GL_FRAGMENT_SHADER),
        validate=False,
    )
    glLinkProgram(program)
    return program


def shader_ptrs(program, symbols: Iterable[str]):
    out: Dict[str, GLint] = {}
    for s in symbols:
        loc = glGetUniformLocation(program, s)
        out[s] = loc
    return out


def plane_shader(vertex_source=None, fragment_source=None):
    program = load_shader_from_text("plane", vertex_source, fragment_source)
    symbols = ["hu_range", "img", "WH", "ABC", "center", "plane"]
    ptrs = shader_ptrs(program, symbols)
    return program, ptrs


def font_shader(vertex_source=None, fragment_source=None):
    program = load_shader_from_text("font", vertex_source, fragment_source)
    symbols = ["text", "textColor", "projection"]
    ptrs = shader_ptrs(program, symbols)
    return program, ptrs


def volume_shader(vertex_source=None, fragment_source=None):
    program = load_shader_from_text("3D", vertex_source, fragment_source)
    symbols = [
        "volume",
        "model",
        "view",
        "projection",
        "color_1",
        "color_bg",
        "model_i",
    ]
    ptrs = shader_ptrs(program, symbols)
    return program, ptrs


__all__ = ["plane_shader", "font_shader", "volume_shader"]
