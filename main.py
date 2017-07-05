# coding:utf-8
from __future__ import division, print_function
from functools import reduce
import sys
import wave
import pyaudio
from math import sqrt
import numpy as np

filename = "affection.wav"
block = 128
sampwidth = 2
#channels = 2
channels = 1
framerate = 44100

dumpfile = "dump128blocks-raw"


def playback_file(filename):
    wf = wave.open(filename, "r")
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True)

    chunk = 4096

    data = wf.readframes(chunk)
    while data != '':
        stream.write(data)
        data = wf.readframes(chunk)

    stream.close()
    p.terminate()

# size must be multiple of 4096
def get_data(filename, size, shift=3):
    wf = wave.open(filename, "r")
    chunk = 4096
    data = wf.readframes(chunk)
    for i in range(shift - 1):
        data = wf.readframes(chunk)
    byte_data = bytes()
    for i in range(size // chunk):
        byte_data = byte_data + wf.readframes(chunk)

    ret = bytes()
    print(len(byte_data))
    for i in range(0, len(byte_data), 4):
        if i % 100000 == 0:
            print(i)
        ret = ret + byte_data[i:i + 2]
    print(len(ret))
    return ret

def playback_data(data):
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(sampwidth),
            channels=channels,
            rate=framerate,
            output=True)
    for i in range(block):
        c = data[4096 * i: 4096 * (i+1)]
        stream.write(c)
    stream.close()
    p.terminate()

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
        ret += chr(x2) + chr(x1)
    return ret


#data = get_data(filename, 4096 * block, 20)
with open(dumpfile, "rb") as f:
    data = f.read()
    #f.write(data)

assert bytes2vec(vec2bytes([1, 2, 3])) == [1, 2, 3]
#data = bytes2vec(data)
#playback_data(vec2bytes(data))

def upsampling(vec, size=2):
    ret = []
    for x in vec:
        ret.extend([x] + [0 for i in range(size - 1)])
    return ret

def downsampling(vec, size=2):
    return vec[::size]

dim = 4096
h = [0 for i in range(4096)]
h[0] = np.float16(1 / sqrt(2))
h[1] = np.float16(1 / sqrt(2))

g = [0 for i in range(4096)]
g[0] = np.float16(1 / sqrt(2))
g[1] = np.float16(-1 / sqrt(2))

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
    for k in range(N):
        ret.append(_convolution_period(vec1, vec2, k))
    return ret


vec1 = [2, 1, 2, 1]
vec2 = [1, 2, 3, 4]
assert convolution_period(vec1, vec2) == [14, 16, 14, 16]
assert downsampling(upsampling([1, 2, 3])) == [1, 2, 3]

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


u = [0, 1, 2, 3, 4, 5]
phi = h[:6]
psi = g[:6]

phi_rev = reverse(phi)
psi_rev = reverse(psi)

p = convolution_period(phi_rev, u)
p = upsampling(downsampling(p))
p = convolution_period(phi, p)

q = convolution_period(psi_rev, u)
q = upsampling(downsampling(q))
q = convolution_period(psi, q)

u2 = add(p, q)
assert norm(sub(u, u2)) < 0.01

# print(add(p, q))

def convolution_box(vec, g, up=0):
    N = len(g) //2
    #ret = upsampling(convolution_period(g, vec, N = N // (2 ** up)), size=2**up)
    #ret = upsampling(convolution_period(g, vec, N =N), size=2**up)
    ret = convolution_period(upsampling(g,size=2**up), vec, N =N)
    #print(len(ret))
    return ret

def cycling(vec, N):
    ret = []
    while len(ret) < len(vec) * 2:
        ret.extend(vec[:N])
    return ret

def analyze(u, g, h, K = 4):
    assert len(u) == len(g)
    assert len(u) == len(h)
    assert (len(u) % (2 ** K)) == 0

    Ds = []
    N = len(u)
    hjs = [cycling(h, N // (2 ** j)) for j in range(K)]
    for i in range(K):
        tmp = u
        for j in range(i):
            hj = hjs[j]
            hj_rev = reverse(hj)
            tmp = convolution_box(tmp, hj_rev, up=j)

        gi = cycling(g, N // (2 ** i))
        gi_rev = reverse(gi)
        tmp = convolution_box(tmp, gi_rev, up=i)
        tmp = downsampling(tmp, size=2 ** (i+1))
        Ds.append(tmp)

    A_vec = u
    for j in range(K):
        hj = hjs[j]
        hj_rev = reverse(hj)
        A_vec = convolution_box(A_vec, hj_rev, up=j)
    A_vec = downsampling(A_vec, size=2**K)
    return Ds, A_vec

def synthesize(D, A, g, h, K=4):
    Q = []
    tmp = upsampling(A, size=2**K)
    N = len(h)
    hjs = [cycling(h, N // (2 ** j)) for j in range(K)]
    for j in range(K-1, -1, -1):
        hj = hjs[j]
        tmp = convolution_box(tmp, hj, up=j)
    P = tmp

    for i in range(K-1, -1 ,-1):
        tmp = D[i]
        tmp = upsampling(tmp, size=2 ** (i + 1))
        gi = cycling(g, N // (2 ** i))
        tmp = convolution_box(tmp, gi, up=i)
        for j in range(i-1, -1, -1):
            hj = hjs[j]
            tmp = convolution_box(tmp, hj, up=j)

        Q.append(tmp)
    return list(reversed(Q)), P

def wavelet_transform(u, g, h, level):
    D, A = analyze(u, g, h, level)
    Q, P = synthesize(D, A, g, h, level)
    return P, Q

def sum_vec(vecs):
    ret = vecs[0]
    for vec in vecs[1:]:
        for i, x in enumerate(vec):
            ret[i] += x
    return ret

def serialize(Ds, A, flags):
    Ds = Ds[:]
    Ds.append(A)
    assert len(Ds) == len(flags)
    return reduce(lambda x, y: x + (y[0] if y[1] else []), zip(Ds, flags), [])

def deserialize(vec, level, size, flags):
    l = size
    idx = 0
    cnt = 1
    ret = []
    for i in range(level):
        if flags[i]:
            ub = idx + l // (2 ** cnt)
            ret.append(vec[idx:ub])
            idx = ub
        else:
            # dummy (garbage code)
            ret.append([0 for i in range(l // (2 ** cnt))])
        cnt += 1
    if flags[level]:
        A = vec[idx:]
    else:
        A = []
    return ret, A

flags = [True, False, True, False]
serd = serialize([range(8),range(4),range(2)], range(2), flags)
assert serd == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
assert deserialize(serd, 3, 16, flags)[0][0] == [0, 1, 2, 3, 4, 5, 6, 7]

def sort_energies(vec):
    tmp = []
    for i, x in enumerate(vec):
        tmp.append((norm(x[1]) ** 2, x[0], i))
    return list(reversed(sorted(tmp, key=lambda x: x[0])))

def cum_energies(vec):
    data = [0]
    bef = vec[0][0] + 1
    for x in vec:
        assert x[0] < bef
        data.append(data[-1] + x[0])
    return data


SIZE = 16
K = 3
g_ = g[:SIZE]
h_ = h[:SIZE]
u = [i % 2 for i in range(16)]

def _compress(u, g_, h_, SIZE, K, threshold=0.98):
    D, A = analyze(u, g_, h_, K)
    Q, P = synthesize(D, A, g_, h_, K)
    energies = sort_energies(list(zip(D + [A], Q + [P])))
    ces = cum_energies(energies)

    sumall = ces[-1]
    ret = [True for i in range(len(ces) - 1)]
    for i, ce in enumerate(ces[:-1]):
        if ce / sumall < threshold:
            continue
        norm, vec, idx = energies[i]
        ret[idx] = False
    result = serialize(D, A, ret)
    return result, ret

def _decompress(result, flags, g_, h_, SIZE, K):
    D2, A2 = deserialize(result, K, SIZE, flags)
    Q2, P2 = synthesize(D2, A2, g_, h_, K)
    return sum_vec(Q2 + [P2])

ret, flags = _compress(u, g_, h_, SIZE, K)
u2 = _decompress(ret, flags, g_, h_, SIZE, K)
assert map(int, map(round, u2)) == u

def compress(filename):
    with open(filename, "rb") as f:
        data = f.read()
    data = bytes2vec(data)[:4096]
    print(len(data))
    x = 1 / sqrt(2)
    g = [x, x] + [0 for i in range(4094)]
    h = [x, -x] + [0 for i in range(4094)]
    result = _compress(data, g, h, 4096, 4)
    print(result)


    # 出力
    pass

compress(dumpfile)

def decompress(filename):
    pass

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("[Usage] python %s [filename]" % sys.argv[0])
        exit(0)
    filename = sys.argv[1]
    playback_file(filename)
