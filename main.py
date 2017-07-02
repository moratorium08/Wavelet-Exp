# coding:utf-8
import sys
import wave
import pyaudio

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

def convolution(vec1, vec2):
    pass

assert downsampling(upsampling([1, 2, 3])) == [1, 2, 3]


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

