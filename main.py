# coding:utf-8
from __future__ import division, print_function
from functools import reduce
import sys
from math import sqrt
import numpy as np
from player import *
import matplotlib.pyplot as plt
from compress import compress, decompress, chunk_size
from vector import *


filename = "affection.wav"
dumpfile = "dump128blocks-raw"



#data = get_data(filename, chunk_size * block, 20)
with open(dumpfile, "rb") as f:
    data = f.read()
    #f.write(data)
#data = bytes2vec(data)
#playback_data(vec2bytes(data))

SIZE = 16
K = 3
h = [0 for i in range(chunk_size)]
h[0] = np.float16(1 / sqrt(2))
h[1] = np.float16(1 / sqrt(2))

g = [0 for i in range(chunk_size)]
g[0] = np.float16(1 / sqrt(2))
g[1] = np.float16(-1 / sqrt(2))
g_ = g[:SIZE]
h_ = h[:SIZE]
u = [i % 2 for i in range(16)]
#u = [i % 4 for i in range(16)]


# compress(dumpfile)
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
