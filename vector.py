# coding:utf-8
from __future__ import division, print_function
import struct
from functools import reduce
from math import sqrt
import numpy as np


def bytes2vec(data):
    ret = []
    for i in range(0, len(data), 2):
        x = ord(data[i + 1]) * 256 + ord(data[i])
        ret.append(x)
    return ret


def vec2bytes(vec):
    ret = bytes()
    for x in vec:
        x2 = x % 256
        x1 = x // 256
        if x1 >= 256:
            x1 = 255
        elif x1 < 0:
            x1 = 0

        try:
            ret += chr(x2) + chr(x1)
        except:
            print(x2, x1)
            raise
    return ret


def upsampling(vec, size=2):
    ret = np.zeros(len(vec)*size)
    for i, x in enumerate(vec):
        ret[size * i] = x
    return ret.tolist()


def downsampling(vec, size=2):
    return vec[::size]


def _convolution_period(vec1, vec2, k):
    ret = 0
    l = len(vec2)
    for i in range(l):
        ret += vec1[i] * vec2[(k - i) % l]
    return ret


def convolution_period(vec1, vec2, N=None):
    if N is None:
        N = len(vec1)
    ret = []
    signal = vec1[:N]
    ker = vec2[:N]
    try:
        ret = np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) )).tolist()
    except:
        print("Erro:", len(vec1), len(vec2))
        raise
    return ret


def reverse(l):
    return [l[0]] + l[1:][::-1]


def add(vec1, vec2):
    assert len(vec1) == len(vec2)
    return map(lambda x : x[0] + x[1], zip(vec1, vec2))


def sub(vec1, vec2):
    assert len(vec1) == len(vec2)
    return map(lambda x : x[0] - x[1], zip(vec1, vec2))


def norm(vec):
    return reduce(lambda x, y: x + (y ** 2), vec, 0) ** 0.5


def convolution_box(vec, g, up=0):
    N = len(g) // 2
    u = upsampling(g, size=2 ** up)
    ret = convolution_period(u, vec, N)
    return ret


def cycling(vec, N):
    ret = []
    while len(ret) < len(vec) * 2:
        ret.extend(vec[:N])
    return ret


def sum_vec(vecs):
    ret = vecs[0]
    for vec in vecs[1:]:
        for i, x in enumerate(vec):
            ret[i] += x
    return ret
