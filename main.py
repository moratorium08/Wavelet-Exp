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
    # assert len(vec1) == len(vec2)
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
    for i in range(K):
        tmp = u
        N = len(u)
        for j in range(i):
            hj = cycling(h, N // (2 ** j))
            hj_rev = reverse(hj)
            tmp = convolution_box(tmp, hj_rev, up=j)

        gi = cycling(g, N // (2 ** i))
        gi_rev = reverse(gi)
        tmp = convolution_box(tmp, gi_rev, up=i)
        tmp = downsampling(tmp, size=2 ** (i+1))
        Ds.append(tmp)

    A_vec = u
    for j in range(K):
        hj = cycling(h, N // (2 ** j))
        hj_rev = reverse(hj)
        A_vec = convolution_box(A_vec, hj_rev, up=j)
    A_vec = downsampling(A_vec, size=2**K)
    return Ds, A_vec

def synthesize(D, A, g, h, K=4):
    Q = []
    tmp = upsampling(A, size=2**K)
    N = len(h)
    for j in range(K-1, -1, -1):
        hj = cycling(h, N // (2 ** j))
        tmp = convolution_box(tmp, hj, up=j)
    P = tmp

    for i in range(K-1, -1 ,-1):
        tmp = D[i]
        tmp = upsampling(tmp, size=2 ** (i + 1))
        gi = cycling(g, N // (2 ** i))
        tmp = convolution_box(tmp, gi, up=i)
        for j in range(i-1, -1, -1):
            hj = cycling(h, N // (2 ** j))
            tmp = convolution_box(tmp, hj, up=j)

        Q.append(tmp)
    return P, Q

def wavelet_transform(u, g, h, level):
    D, A = analyze(u, g, h, level)
    P, Q = synthesize(D, A, g, h, level)
    return P, Q

def sum_vec(vecs):
    ret = vecs[0]
    for vec in vecs[1:]:
        for i, x in enumerate(vec):
            ret[i] += x
    return ret

def serialize(Ds, A):
    Ds.append(A)
    return reduce(lambda x, y: x + y, Ds, [])

def deserialize(vec, level):
    l = len(vec)
    idx = 0
    cnt = 1
    ret = []
    for i in range(level):
        ub = idx + l // (2 ** cnt)
        ret.append(vec[idx:ub])
        idx = ub
        cnt += 1
    return ret, vec[idx:]

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
    return data[1:]



SIZE = 16
K = 3
g_ = g[:SIZE]
h_ = h[:SIZE]
u = [i % 2 for i in range(16)]
D, A = analyze(u, g_, h_, K)
P, Q = synthesize(D, A, g_, h_, K)
print(sum_vec(Q + [P]))


energies = sort_energies(list(zip(D + [A], Q + [P])))

ces = cum_energies(energies)

sumall = ces[-1]
threshold = 0.99
ret = [0 for i in range(SIZE)]
for i, ce in enumerate(ces):
    if ce / sumall > 1.1:
        break
    norm, vec, idx = energies[i]
    for j, x in enumerate(vec):
        ret[idx + j] = x
print(ret)

print(ce)
print(D, A)
#print(P, Q)

#D, A = decompose(list(range(SIZE)), g_, h_, K)
#P, Q = compose(D, A, g_, h_, K)
#print(map(round, sum_vec(Q + [P])))


def compress(filename):
    # data reading

    # Pp(u), Qp(u), Qp(u-1) ...Q1(u)のそれぞれの係数の計算

    # 係数の大きい順番に並び替え

    # エネルギーの計算

    # 出力
    pass


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("[Usage] python %s [filename]" % sys.argv[0])
        exit(0)
    filename = sys.argv[1]
    playback_file(filename)
