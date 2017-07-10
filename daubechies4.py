# coding:utf-8
from __future__ import division, print_function
from functools import reduce
import sys
from math import sqrt
import numpy as np
from player import *
import matplotlib.pyplot as plt
from compress import compress, decompress, chunk_size, read_wav_dump
from compress import read_wavelet_dump
from vector import *

h = [0] * 4096
g = [0] * 4096

h[0] = (1 + sqrt(3)) / (4 * sqrt(2))
h[1] = (3 + sqrt(3)) / (4 * sqrt(2))
h[2] = (3 - sqrt(3)) / (4 * sqrt(2))
h[3] = (1 - sqrt(3)) / (4 * sqrt(2))

g[0] = (1 - sqrt(3)) / (4 * sqrt(2))
g[1] = (-3 + sqrt(3)) / (4 * sqrt(2))
g[2] = (3 + sqrt(3)) / (4 * sqrt(2))
g[3] = (-1 - sqrt(3)) / (4 * sqrt(2))



dumpfile = "dump128blocks-raw"
size = 15

bytedata = read_wav_dump(dumpfile)[:4096 * size]
x = compress(bytedata, filename=dumpfile+".daubechies4.cmpd",
        verbose=1, threshold=0.98,
        g=g,
        h=h)

print("Compress rate:", x / 4096 / 2 / size)

original = map(lambda x: x * 256 * 256, bytedata)


dumpdata = read_wavelet_dump(dumpfile + ".daubechies4.cmpd")

result = decompress(dumpdata, verbose=0, g=g, h=h)
result = map(lambda x: int(round(x)), result)
dif = sub(original, result)

print("MSE", (norm(dif) ** 0.5) / 4096)

plt.plot(original, label="raw_data")
plt.plot(result, label="compressed")
plt.legend()
plt.show()

playback_data(vec2bytes(map(lambda x: int(round(x)), result)))
