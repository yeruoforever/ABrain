from collections import namedtuple
from typing import Iterable
import numpy as np
from OpenGL.GL import *

__all__ = [
    "VertexArray",
    "plane_buffers",
    "font_buffers",
    "delete_vertex_array",
]


VertexArray = namedtuple("VertexArray", ("vao", "vbo", "ebo"))


def plane_buffers():
    vertexs = np.array(
        [
            -1.0,
            -1.0,
            +1.0,
            -1.0,
            -1.0,
            +1.0,
            +1.0,
            +1.0,
        ],
        dtype=np.float32,
    )

    indexs = np.array([0, 1, 2, 3], dtype=np.uint32)

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, None, vertexs, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, None, indexs, GL_STATIC_DRAW)
    glVertexAttribPointer(
        0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), ctypes.c_void_p(0)
    )
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return VertexArray(vao, vbo, ebo)


def font_buffers():
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, None, GL_DYNAMIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return VertexArray(vao, vbo, 0)


def delete_vertex_array(*VAs: Iterable[VertexArray]):
    for VA in VAs:
        if VA is None:
            continue
        glDeleteBuffers(1, VA.ebo)
        glDeleteBuffers(1, VA.vbo)
        glDeleteVertexArrays(1, VA.vao)
