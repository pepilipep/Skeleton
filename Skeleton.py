import subprocess

import numpy as np

import subprocess
import os
import sys

import cv2

import math

from scipy.io import wavfile

import argparse

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

videofile_path = './'
videofile_name = ''
videofile_ext = ''

videofile = args.input

temp_dir = args.temp_dir
if temp_dir[-1] != '/':
    temp_dir += '/'

for i in range(len(videofile) - 1, -1, -1):
    if videofile[i] == '.':
        videofile_ext = videofile[i:]
        videofile_name = videofile[:i]
        break

for i in range(len(videofile_name) - 1, -1, -1):
    if videofile_name[i] == '/':
        videofile_path = videofile_name[:i]
        videofile_name = videofile_name[i:]
        break

print(videofile_path, videofile_name, videofile_ext)

#MAGICS

min_sound = args.min_sound

min_time = args.min_time

silent_speed = args.silent_speed

loud_speed = args.loud_speed

#INIT

subprocess.call(
    ['mkdir', videofile_path + temp_dir]
)

#get info


cv2video = cv2.VideoCapture(videofile_path + videofile_name + videofile_ext)
fps = cv2video.get(cv2.CAP_PROP_FPS)
if args.fps:
    fps = args.fps
num_frames = int(cv2video.get(cv2.CAP_PROP_FRAME_COUNT))
time_video = num_frames / fps

print(fps, num_frames, time_video)

#get audio

subprocess.call(
    ['ffmpeg', '-i', videofile_path + videofile_name + videofile_ext, '-codec:a', 'pcm_s16le', '-ac', '1', videofile_path + temp_dir + videofile_name + '.wav'])
print('file saved')

fs = wavfile.read(videofile_path + temp_dir + videofile_name + '.wav')[0]
data = wavfile.read(videofile_path + temp_dir + videofile_name + '.wav')[1].copy()
maxaudio_volume = np.max(data)

sounds_per_frame = math.ceil(data.shape[0] / num_frames)

#get frames as images

subprocess.call(
    ['ffmpeg', '-i', videofile_path + videofile_name + videofile_ext, '-qscale:v', str(args.frame_quality), videofile_path + temp_dir + '$old_frames%06d.jpg', '-hide_banner']
)

#ffmpeg -i file.mpg -r 1/1 $filename%03d.jpg

#algorithm

gone = np.zeros(num_frames)

data.resize((sounds_per_frame + 1) * num_frames)

sound_split_by_frame = np.absolute(np.reshape(data, (num_frames, sounds_per_frame + 1)))

total_sound_by_frames = np.max(sound_split_by_frame, axis=-1)

wanted = total_sound_by_frames >= min_sound * maxaudio_volume

print(np.sum(wanted))

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
    if cuts[i + 1][0] - mid_cuts[-1][1] < min_time * fps:
        mid_cuts[-1][1] = cuts[i + 1][1]
    else:
        mid_cuts.append(cuts[i + 1])

for cut in mid_cuts:
    if cut[1] - cut[0] >= min_time * fps:
        final_cuts.append(cut)

# find wanted frames

silent_audio = np.full(num_frames, True)
loud_audio = np.full(num_frames, False)

for cut in final_cuts:
    for frame in range(cut[0], cut[1] + 1):
        loud_audio[frame] = True
        silent_audio[frame] = False

next_frame = np.zeros(num_frames)
next_frame[-1] = num_frames
for i in range(num_frames - 2, -1, -1):
    if loud_audio[i] ^ loud_audio[i + 1]:
        next_frame[i] = i + 1
    else:
        next_frame[i] = next_frame[i + 1]

silent_frames = np.extract(silent_audio, range(num_frames))

# build new audio frames

loud_audio_sound_frames = np.repeat(loud_audio, sounds_per_frame + 1)

next_audio_frame = np.zeros(data.shape[0])
next_audio_frame[-1] = data.shape[0]
for i in range(data.shape[0] - 2, -1, -1):
    if loud_audio_sound_frames[i] ^ loud_audio_sound_frames[i + 1]:
        next_audio_frame[i] = i + 1
    else:
        next_audio_frame[i] = next_audio_frame[i + 1]

prev_frame = -1

new_audio_frames = []

i = 0
curr_speed = 0
while i < data.shape[0]:
    curr_frame = math.floor(i)
    curr_speed = loud_speed if loud_audio_sound_frames[curr_frame] else silent_speed
    if math.floor(i + curr_speed) > next_audio_frame[curr_frame]:
        i = next_audio_frame[curr_frame]
    else:
        new_audio_frames.append(data[curr_frame])
        i += curr_speed

newAudio = np.asarray(new_audio_frames)

wavfile.write(videofile_path + temp_dir + 'newAudio.wav', fs, newAudio)

# build new video frames
prev_frame = -1
frame_id = 1

new_frames = []

i = 0
curr_speed = 0
while i < num_frames:
    curr_frame = math.floor(i)
    curr_speed = silent_speed if silent_audio[curr_frame] else loud_speed
    if math.floor(i + curr_speed) > next_frame[curr_frame]:
        i = next_frame[curr_frame]
    else:
        new_frames.append(curr_frame)
        i += curr_speed

print('building frames for the new video...')

for new_id, old_id in enumerate(new_frames):
    subprocess.call(
        ['cp', videofile_path + temp_dir + '$old_frames%06d.jpg' % (old_id + 1), videofile_path + temp_dir + '$new_frames%06d.jpg' % (new_id + 1)]
    )
    
# build remastered video

subprocess.call(
    ['ffmpeg', '-framerate', str(fps), '-i', videofile_path + temp_dir + '$new_frames%06d.jpg', '-i', videofile_path + temp_dir + 'newAudio.wav', '-strict', '-2', videofile_path + videofile_name + '_remastered' + videofile_ext]
)

# ffmpeg -framerate fps -i temp/$new_frames%06d.jpg -i temp/newAudio.wav -strict -2 "blank_remastered.mp4"

#remove useless files

print('do you want to remove the temporary files?')
subprocess.call(
    ['rm', '-r', '-I', videofile_path + temp_dir]
)
