from typing import *
from OpenGL.GL import *

__all__ = ["VAO", "Buffer", "Texture", "Program"]


class VAO(object):
    def __init__(self, vao_id) -> None:
        self.id = vao_id

    def __enter__(self):
        glBindVertexArray(self.id)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        glBindVertexArray(0)


class Buffer(object):
    def __init__(self, buffer_id, buffe_type) -> None:
        self.id = buffer_id
        self.type = buffe_type

    def __enter__(self):
        glBindBuffer(self.type, self.id)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        glBindBuffer(self.type, 0)


class Texture(object):
    def __init__(self, texture) -> None:
        self.texture = texture

    def __enter__(self):
        if self.texture != GL_TEXTURE0:
            glActiveTexture(self.texture)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.texture != GL_TEXTURE0:
            glActiveTexture(GL_TEXTURE0)


class Program(object):
    def __init__(self, prog) -> None:
        self.prog = prog

    def __enter__(self):
        glUseProgram(self.prog)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        glUseProgram(0)
