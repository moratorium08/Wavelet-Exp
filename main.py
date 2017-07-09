# coding:utf-8
from __future__ import division, print_function
from functools import reduce
import sys
from math import sqrt
import numpy as np
import struct
from vector import *
from player import *
from wavelet import *
from util import int2bytes, bytes2int

import matplotlib.pyplot as plt


filename = "affection.wav"
dumpfile = "dump128blocks-raw"


#data = get_data(filename, 4096 * block, 20)
with open(dumpfile, "rb") as f:
    data = f.read()
    #f.write(data)
#data = bytes2vec(data)
#playback_data(vec2bytes(data))

SIZE = 16
K = 3
h = [0 for i in range(4096)]
h[0] = np.float16(1 / sqrt(2))
h[1] = np.float16(1 / sqrt(2))

g = [0 for i in range(4096)]
g[0] = np.float16(1 / sqrt(2))
g[1] = np.float16(-1 / sqrt(2))
g_ = g[:SIZE]
h_ = h[:SIZE]
u = [i % 2 for i in range(16)]
#u = [i % 4 for i in range(16)]

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


ret, flags = _compress(u, g_, h_, SIZE, K)
u2 = _decompress(ret, flags, g_, h_, SIZE, K)
# print(u2)
assert map(int, map(round, u2)) == u


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
    #for i in range(len(bytedata)//4096):
    for i in range(15):
        lb = i * 4096
        ub = (i + 1) * 4096
        data = bytedata[lb:ub]
        x = 1 / sqrt(2)
        g = [x, x] + [0 for i in range(4094)]
        h = [x, -x] + [0 for i in range(4094)]

        result, flags = _compress(data, g, h, 4096, 12)
        result = np.array(result, dtype=np.float16).tobytes()
        d = len(result)
        print("lend", d)
        print("lnflgas", len(flags))
        print("len(d)", len(int2bytes(d)))
        print("len(flgas2", len(flags2bytes(flags)))
        result_b += int2bytes(d)
        result_b += flags2bytes(flags)
        result_b += result
    with open(filename + ".cmpd", "wb") as f:
        f.write(result_b)

# compress(dumpfile)

def decompress(filename):
    with open(filename, "rb") as f:
        dumpdata = f.read()
    ret = []
    idx = 0
    while idx < len(dumpdata):
        length = bytes2int(dumpdata[idx:idx + 2])
        idx += 2
        lb = idx
        ub = idx + length + 4096 // 8
        idx = ub
        _dumpdata = dumpdata[lb:ub]
        flags_b = _dumpdata[:4096 // 8]
        flags = bytes2flags(flags_b)
        flags = [False for i in range(4096 - len(flags))] + flags
        data = _dumpdata[4096 // 8:]
        data = np.frombuffer(data, dtype=np.float16).tolist()
        x = 1 / sqrt(2)
        g = [x, x] + [0 for i in range(4094)]
        h = [x, -x] + [0 for i in range(4094)]
        result = _decompress(data, flags, g, h, 4096, 12)
        print("flags", len(flags))
        print("lendata", len(data))
        print(len(result))
        ret += map(lambda x: x * 256 * 256, result)
    return ret


result = decompress(dumpfile + ".cmpd")
#plt.plot(map(lambda x: x / (256 * 256), result))
result = map(lambda x: int(round(x)), result)
#plt.plot(result)
#plt.show()
playback_data(vec2bytes(map(lambda x: int(round(x)), result)))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("[Usage] python %s [filename]" % sys.argv[0])
        exit(0)
    filename = sys.argv[1]
    playback_file(filename)
