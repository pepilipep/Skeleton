import subprocess
import numpy as np
import subprocess
import os
import sys
import cv2
import math
from scipy.io import wavfile
import argparse
import librosa

# parse arguments

parser = argparse.ArgumentParser(description='Run silent and loud parts of a video at different speeds', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-i', '--input', type=str, help='input video file', required=True)
parser.add_argument('-d', '--temp_dir', type=str, default='temp/', help='name of temporary directory to save temporary files (no need to already exist)')
parser.add_argument('-m', '--min_sound', type=float, default=0.06, help='a number between 0 and 1. the threshold for classifying as silent/loud')
parser.add_argument('-t', '--min_time', type=float, default=0.15, help='sequences of frames shorter than this will be considered of the opposite type')
parser.add_argument('-f', '--fps', type=float, help='fps of the video. default is the program figuring it out itself')
parser.add_argument('-q', '--frame_quality', type=int, default=3, help='an integer between 1 and 31. 1 is highest quality, 31 is lowest')
parser.add_argument('-s', '--silent_speed', type=float, default=2, help='new speed of the silent parts')
parser.add_argument('-l', '--loud_speed', type=float, default=1, help='new speed of the loud parts')

args = parser.parse_args()

#VIDEO INFO

FILE_PATH = './'
FILE_NAME = ''
FILE_EXT = ''

videofile = args.input

TEMP_DIR = args.temp_dir

if TEMP_DIR[-1] != '/':
    TEMP_DIR += '/'

for i in range(len(videofile) - 1, -1, -1):
    if videofile[i] == '.':
        FILE_EXT = videofile[i:]
        FILE_NAME = videofile[:i]
        break

for i in range(len(FILE_NAME) - 1, -1, -1):
    if FILE_NAME[i] == '/':
        FILE_PATH = FILE_NAME[:i]
        FILE_NAME = FILE_NAME[i:]
        break

print(FILE_PATH, FILE_NAME, FILE_EXT)

#MAGICS

MIN_SOUND = args.min_sound

MIN_TIME = args.min_time

SILENT_SPEED = args.silent_speed

LOUD_SPEED = args.loud_speed

# make temporary directory

subprocess.call(
    ['mkdir', FILE_PATH + TEMP_DIR]
)

#get info

cv2video = cv2.VideoCapture(FILE_PATH + FILE_NAME + FILE_EXT)
FPS = cv2video.get(cv2.CAP_PROP_FPS)

if args.fps:
    FPS = args.fps

num_frames = int(cv2video.get(cv2.CAP_PROP_FRAME_COUNT))
time_video = num_frames / FPS

print(FPS, num_frames, time_video)

#get audio and audio info

subprocess.call(
    ['ffmpeg', '-i', FILE_PATH + FILE_NAME + FILE_EXT, '-codec:a', 'pcm_s16le', '-ac', '1', FILE_PATH + TEMP_DIR + FILE_NAME + '.wav'])
print('file saved')

FS = wavfile.read(FILE_PATH + TEMP_DIR + FILE_NAME + '.wav')[0]
data = wavfile.read(FILE_PATH + TEMP_DIR + FILE_NAME + '.wav')[1].copy()
maxaudio_volume = np.max(data)

sounds_per_frame = math.ceil(data.shape[0] / num_frames)

#get frames as images

subprocess.call(
    ['ffmpeg', '-i', FILE_PATH + FILE_NAME + FILE_EXT, '-qscale:v', str(args.frame_quality), FILE_PATH + TEMP_DIR + '$old_frames%06d.jpg', '-hide_banner']
)

#ffmpeg -i file.mpg -qscale:v 3 $filename%03d.jpg -hide_banner

#algorithm

gone = np.zeros(num_frames)

data.resize((sounds_per_frame + 1) * num_frames)

sound_split_by_frame = np.absolute(np.reshape(data, (num_frames, sounds_per_frame + 1)))

total_sound_by_frames = np.max(sound_split_by_frame, axis=-1)

wanted = total_sound_by_frames >= MIN_SOUND * maxaudio_volume

print('loud frames: ', np.sum(wanted))

cuts = []
if wanted[0]:
    cuts.append([0, 0])

for i in range(len(wanted[1:])):
    if wanted[i] and wanted[i - 1]:
        cuts[-1][1] = i
    elif wanted[i]:
        cuts.append([i, i])

mid_cuts = []
final_cuts = []

mid_cuts.append(cuts[0])

for i in range(len(cuts) - 1):
    if cuts[i + 1][0] - mid_cuts[-1][1] < MIN_TIME * FPS:
        mid_cuts[-1][1] = cuts[i + 1][1]
    else:
        mid_cuts.append(cuts[i + 1])

for cut in mid_cuts:
    if cut[1] - cut[0] >= MIN_TIME * FPS:
        final_cuts.append(cut)

# find wanted frames

silent_audio = np.full(num_frames, True)
loud_audio = np.full(num_frames, False)

for cut in final_cuts:
    for frame in range(cut[0], cut[1] + 1):
        loud_audio[frame] = True
        silent_audio[frame] = False

silent_frames = np.extract(silent_audio, range(num_frames))

# some useful info

appr_time = (silent_frames.shape[0] / SILENT_SPEED + (num_frames - silent_frames.shape[0]) / LOUD_SPEED ) / FPS
print('approximate length of new video: %d minutes, %d seconds' % (appr_time / 60, appr_time % 60))

# build new audio

print('building new audio...')

loud_audio_sound_frames = np.repeat(loud_audio, sounds_per_frame + 1)

next_audio_frame = np.zeros(data.shape[0], np.int32)
next_audio_frame[-1] = data.shape[0]
for i in range(data.shape[0] - 2, -1, -1):
    if loud_audio_sound_frames[i] ^ loud_audio_sound_frames[i + 1]:
        next_audio_frame[i] = int(i + 1)
    else:
        next_audio_frame[i] = next_audio_frame[i + 1]

prev_frame = -1

new_audio_frames = []

i = 0
while i < data.shape[0]:
    curr_speed = LOUD_SPEED if loud_audio_sound_frames[i] else SILENT_SPEED
    new_audio_frames.extend(librosa.effects.time_stretch(data[i:next_audio_frame[i]].astype(np.float), curr_speed).astype(np.int16))
    i = next_audio_frame[i]

newAudio = np.asarray(new_audio_frames)

# write final audio

wavfile.write(FILE_PATH + TEMP_DIR + 'newAudio.wav', FS, newAudio)

# free memory

new_audio_frames = None
next_audio_frame = None
newAudio = None
data = None
loud_audio_sound_frames = None
sound_split_by_frame = None

# build new video frames

print('building frames for the new video...')

next_frame = np.zeros(num_frames)
next_frame[-1] = num_frames
for i in range(num_frames - 2, -1, -1):
    if loud_audio[i] ^ loud_audio[i + 1]:
        next_frame[i] = i + 1
    else:
        next_frame[i] = next_frame[i + 1]

prev_frame = -1
frame_id = 1

new_frames = []

i = 0
curr_speed = 0
remainders = [0,0]

while i < num_frames:
    curr_frame = math.floor(i)
    curr_speed = SILENT_SPEED if silent_audio[curr_frame] else LOUD_SPEED
    curr_idx = 0 if silent_audio[curr_frame] else 1
    
    if math.floor(i + curr_speed - remainders[curr_idx]) >= next_frame[curr_frame]:
        remainders[curr_idx] += (next_frame[curr_frame] - i)
        i = next_frame[curr_frame]
    elif math.floor(i + curr_speed) >= next_frame[curr_frame]:
        remainders[curr_idx] -= (i + curr_speed - next_frame[curr_frame])
        new_frames.append(curr_frame)
        i = next_frame[curr_frame]
    else:
        new_frames.append(curr_frame)
        i += curr_speed

for new_id, old_id in enumerate(new_frames):
    subprocess.call(
        ['cp', FILE_PATH + TEMP_DIR + '$old_frames%06d.jpg' % (old_id + 1), FILE_PATH + TEMP_DIR + '$new_frames%06d.jpg' % (new_id + 1)]
    )
    
# build remastered video

subprocess.call(
    ['ffmpeg', '-framerate', str(FPS), '-i', FILE_PATH + TEMP_DIR + '$new_frames%06d.jpg', '-i', FILE_PATH + TEMP_DIR + 'newAudio.wav', '-strict', '-2', FILE_PATH + FILE_NAME + '_remastered' + '.mp4']
)

# ffmpeg -framerate fps -i temp/$new_frames%06d.jpg -i temp/newAudio.wav -strict -2 "blank_remastered.mp4"

#remove useless files

print('do you want to remove the temporary files?')
subprocess.call(
    ['rm', '-r', '-I', FILE_PATH + TEMP_DIR]
)
