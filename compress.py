# coding:utf-8
from __future__ import division, print_function
from functools import reduce
from math import sqrt
import numpy as np
import struct
from vector import *
from wavelet import *
from util import int2bytes, bytes2int

chunk_size = 4096

def sort_energies(vec):
    tmp = []
    for i, x in enumerate(vec):
        tmp.append((x ** 2, x, i))
    return list(reversed(sorted(tmp, key=lambda x: x[0])))


def cum_energies(vec):
    data = [0]
    bef = vec[0][0] + 1
    for x in vec:
        assert x[0] < bef
        data.append(data[-1] + x[0])
    return data


def _compress(u, g_, h_, SIZE, K, threshold=0.99):
    D, A = analyze(u, g_, h_, K)
    result = serialize(D, A)

    energies = sort_energies(result)
    ces = cum_energies(energies)

    sumall = ces[-1]
    flags = [True for i in range(len(ces) - 1)]
    for i, ce in enumerate(ces[:-1]):
        if ce / sumall < threshold:
            continue
        norm, vec, idx = energies[i]
        flags[idx] = False
    ret = []
    for x, flag in zip(result, flags):
        if flag:
            ret.append(x)
    D2, A2 = deserialize(ret, K, SIZE, flags)
    return ret, flags


def _decompress(result, flags, g_, h_, SIZE, K):
    D2, A2 = deserialize(result, K, SIZE, flags)
    Q2, P2 = synthesize(D2, A2, g_, h_, K)
    ret = sum_vec(Q2 + [P2])
    return ret


def flags2bytes(flags):
    d = int(''.join(map(lambda x: str(int(x)), flags)), 2)
    return int2bytes(d, just = len(flags) // 8)


def bytes2flags(b):
    x = bytes2int(b)
    ret = []
    while x > 0:
        if x % 2 == 1:
            ret.append(True)
        else:
            ret.append(False)
        x //= 2
    return list(reversed(ret))


def compress(filename):
    with open(filename, "rb") as f:
        bytedata = f.read()
    result_b = bytes()
    bytedata = bytes2vec(bytedata)
    bytedata = map(lambda x: x / (256 * 256), bytedata)
    #for i in range(len(bytedata)//chunk_size):
    for i in range(15):
        lb = i * chunk_size
        ub = (i + 1) * chunk_size
        data = bytedata[lb:ub]
        x = 1 / sqrt(2)
        g = [x, x] + [0 for i in range(chunk_size - 2)]
        h = [x, -x] + [0 for i in range(chunk_size - 2)]

        result, flags = _compress(data, g, h, chunk_size, 12)
        result = np.array(result, dtype=np.float16).tobytes()
        d = len(result)
        result_b += int2bytes(d)
        result_b += flags2bytes(flags)
        result_b += result
    with open(filename + ".cmpd", "wb") as f:
        f.write(result_b)


def decompress(filename):
    with open(filename, "rb") as f:
        dumpdata = f.read()
    ret = []
    idx = 0
    while idx < len(dumpdata):
        length = bytes2int(dumpdata[idx:idx + 2])
        idx += 2
        lb = idx
        ub = idx + length + chunk_size // 8
        idx = ub
        _dumpdata = dumpdata[lb:ub]
        flags_b = _dumpdata[:chunk_size // 8]
        flags = bytes2flags(flags_b)
        flags = [False for i in range(chunk_size - len(flags))] + flags
        data = _dumpdata[chunk_size // 8:]
        data = np.frombuffer(data, dtype=np.float16).tolist()
        x = 1 / sqrt(2)
        g = [x, x] + [0 for i in range(chunk_size - 2)]
        h = [x, -x] + [0 for i in range(chunk_size - 2)]
        result = _decompress(data, flags, g, h, chunk_size, 12)
        ret += map(lambda x: x * 256 * 256, result)
    return ret
