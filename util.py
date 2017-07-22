# coding:utf-8
from __future__ import division

def int2bytes(x,just=-1):
    b = ""
    while x > 0:
        tmp = x % 256
        x //= 256
        b = chr(tmp) + b
    if just == -1:
        just = len(b)

    assert len(b) <= just

    b = "\x00" * (just - len(b)) + b
    return b

def bytes2int(b):
    x = 0
    for c in b:
        x = 256 * x + ord(c)
    return x
