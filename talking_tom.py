import cv2
import sounddevice as sd
import numpy as np
import time
import librosa
import argparse


parser = argparse.ArgumentParser(
    "Talking Tom", description="Inspired from the mobile app Talking Tom")
parser.add_argument("--sample-rate", "-s", type=int, default=48_000)
parser.add_argument("--block-size", "-b", type=int, default=24_000)
parser.add_argument("--pitch-shift", "-p", type=int, default=8)
parser.add_argument("--thresh-voice", type=float, default=0.1)
parser.add_argument("--thresh-silence", type=float, default=0.05)

args = parser.parse_args()

sample_rate = args.sample_rate
block_size = args.block_size
pitch_shift = args.pitch_shift
thresh_voice = args.thresh_voice
thresh_silence = args.thresh_silence


# Setting up animation frames
f1 = cv2.imread("f1.png")
f2 = cv2.imread("f2.png")
frame_number = 0


output = []
# States
LISTENING = 0
RECORDING = 1
PLAYBACK = 2
state = LISTENING


playback_starttime = None
playback_duration = None

cv2.namedWindow("Talking Tom", cv2.WINDOW_NORMAL)


def callback(indata, frames, time_, status):
    global state, playback_starttime, playback_duration, output

    if state == PLAYBACK:
        return
    elif state == LISTENING:
        if np.max(np.abs(indata[:, 0])) > thresh_voice:
            state = RECORDING
            output.extend(indata[:, 0])
    elif state == RECORDING:
        output.extend(indata[:, 0])
        if np.max(np.abs(indata[:, 0])) < thresh_silence:
            playback_starttime = time.time()
            playback_duration = len(output)/sample_rate
            state = PLAYBACK
            output = librosa.effects.pitch_shift(
                np.array(output), sr=sample_rate, n_steps=pitch_shift).tolist()
            sd.play(output)


input_stream = sd.InputStream(
    channels=1, dtype='float32', callback=callback, blocksize=block_size, samplerate=sample_rate)
input_stream.start()

last_time = time.time()
while True:
    if state == PLAYBACK:
        if frame_number % 2 == 0:
            cv2.imshow("Talking Tom", f1)
        else:
            cv2.imshow("Talking Tom", f2)
    else:
        cv2.imshow("Talking Tom", f1)
    if time.time() > last_time + 0.2:
        last_time = time.time()
        frame_number += 1
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    if state == PLAYBACK:
        if time.time() > playback_starttime + playback_duration + 0.05:
            output = []
            state = LISTENING

input_stream.stop()
input_stream.close()
