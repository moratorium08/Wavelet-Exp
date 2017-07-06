# coding:utf-8
import wave
import pyaudio


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
    for i in range(0, len(byte_data), 4):
        if i % 100000 == 0:
            print(i)
        ret = ret + byte_data[i:i + 2]
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

if __name__ == '__main__':
    pass

