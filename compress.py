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
        assert x[0] <= bef
        bef = x[0]
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


def compress(bytedata, g=None, h=None, threshold=0.98,
        filename="dump.cmpd", verbose=0):
    assert len(bytedata) % chunk_size == 0
    #for i in range(len(bytedata)//chunk_size):
    size = len(bytedata) // chunk_size

    if g is None or h is None:
        if verbose == 1:
            print("[Warning]haar wavelet is used")
        x = 1 / sqrt(2)
        h = [x, x] + [0 for i in range(chunk_size - 2)]
        g = [x, -x] + [0 for i in range(chunk_size - 2)]
    written_bytes = 0

    with open(filename, "wb") as f:
        result_b = bytes()
        for i in range(size):
            if verbose == 1:
                print("%d/%d" % (i + 1, size))
            lb = i * chunk_size
            ub = (i + 1) * chunk_size
            data = bytedata[lb:ub]

            result, flags = _compress(data, g, h, chunk_size, 12,
                    threshold=threshold)
            result = np.array(result, dtype=np.float16).tobytes()
            d = len(result)
            result_b += int2bytes(d)
            result_b += flags2bytes(flags)
            result_b += result
        written_bytes += len(result_b)
        f.write(result_b)
    return written_bytes

def decompress(dumpdata, verbose=0, g=None, h=None):
    ret = []
    idx = 0
    cnt = 0
    if g is None or h is None:
        if verbose == 1:
            print("[Warning]haar wavelet is used")
        x = 1 / sqrt(2)
        h = [x, x] + [0 for i in range(chunk_size - 2)]
        g = [x, -x] + [0 for i in range(chunk_size - 2)]
    while idx < len(dumpdata):
        if verbose == 1:
            cnt += 1
            print("%d" % cnt)
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
        result = _decompress(data, flags, g, h, chunk_size, 12)
        ret += map(lambda x: x * 256 * 256, result)
    return ret


def read_wavelet_dump(filename):
    with open(filename, "rb") as f:
        dumpdata = f.read()
    return dumpdata


def read_wav_dump(filename):
    with open(filename, "rb") as f:
        bytedata = f.read()
    result_b = bytes()
    bytedata = bytes2vec(bytedata)
    bytedata = map(lambda x: x / (256 * 256), bytedata)
    return bytedata
