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


filename = "affection.wav"
dumpfile = "dump128blocks-raw"

#data = get_data(filename, chunk_size * block, 20)
with open(dumpfile, "rb") as f:
    data = f.read()
    #f.write(data)
data = bytes2vec(data)
playback_data(vec2bytes(data))
raise

x = 1 / sqrt(2)
g = [x, x] + [0 for i in range(chunk_size - 2)]
h = [x, -x] + [0 for i in range(chunk_size - 2)]

size = 1
bytedata = read_wav_dump(dumpfile)[:4096 * size]
x = compress(bytedata, filename=dumpfile+".cmpd", verbose=1, threshold=0.99)
print("Compress rate:", x / 4096 / 2)

original = map(lambda x: x * 256 * 256, bytedata)


dumpdata = read_wavelet_dump(dumpfile + ".cmpd")
result = decompress(dumpdata, verbose=0)
result = map(lambda x: int(round(x)), result)
dif = sub(original, result)
print("MSE", norm(dif) ** 0.5 / 4096)
plt.plot(original, label="raw_data")
plt.plot(result, label="compressed")
plt.legend()
plt.show()
playback_data(vec2bytes(map(lambda x: int(round(x)), result)))

"""

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("[Usage] python %s [filename]" % sys.argv[0])
        exit(0)
    filename = sys.argv[1]
    playback_file(filename)
"""
