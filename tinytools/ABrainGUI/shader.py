import os
from typing import *

import OpenGL.GL.shaders as shaders
from OpenGL.GL import *

__all__ = ["load_shader_from_text", "shader_ptrs"]


def read_from_text(path: str):
    with open(path, "r",encoding="UTF8") as io:
        content = io.read()
    return content


def load_shader_from_text(
    target: str,
    vertex_source: Optional[str] = None,
    fragment_source: Optional[str] = None,
):
    if vertex_source is None:
        path = os.path.join(os.path.dirname(__file__), "shader", f"{target}.vert")
        vertex_source = read_from_text(path)

    if fragment_source is None:
        path = os.path.join(os.path.dirname(__file__), "shader", f"{target}.frag")
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
