# coding:utf-8
import sys
import wave
import pyaudio

filename = "affection.wav"
block = 128
sampwidth = 2
channels = 2
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
        print(len(data))
    ret = bytes()
    for i in range(size // chunk):
        ret = ret + wf.readframes(chunk)
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

# data = get_data(filename, 4096 * block, 20)
with open(dumpfile, "rb") as f:
    data = f.read()
playback_data(data)



# scaling vector
def get_s_vector(p, shift):
    pass

# wavelet vector
def get_w_vector(p, shift):
    pass

def inner_product(v1, v2):
    pass

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

