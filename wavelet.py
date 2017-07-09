# coding:utf-8
from vector import *


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


def serialize(Ds, A):
    Ds = Ds[:]
    Ds.append(A)
    return reduce(lambda x, y: x + y, Ds, [])


def deserialize(vec, level, size, flags):
    l = size
    cnt = 1
    ret = []

    idx = 0
    itr = 0
    for i in range(level):
        tmp = []
        for j in range(l // (2 ** cnt)):
            if flags[idx]:
                tmp.append(vec[itr])
                itr += 1
            else:
                tmp.append(0)
            idx += 1
        cnt += 1
        ret.append(tmp)
    A = []
    for j in range(l // 2 ** (cnt - 1)):
        if flags[idx]:
            A.append(vec[itr])
            itr += 1
        else:
            A.append(0)
        idx += 1
    return ret, A
