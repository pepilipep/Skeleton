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

parser = argparse.ArgumentParser(description='Run silent and loud parts of a video at different speeds')
parser.add_argument('-i', '--input', type=str, help='input video file', required=True)
parser.add_argument('-d', '--temp_dir', type=str, default='temp/', help='name of temporary directory to save temporary files (no need to already exist)')
parser.add_argument('-m', '--min_sound', type=float, default=0.06, help='a number between 0 and 1. the threshold for classifying as silent/loud')
parser.add_argument('-t', '--min_time', type=float, default=0.15, help='sequences of frames shorter than this will be considered of the opposite type')
parser.add_argument('-f', '--fps', type=float, help='fps of the video. default is the program figuring it out itself')
parser.add_argument('-q', '--frame_quality', type=int, default=3, help='an integer between 1 and 31. 1 is highest quality, 31 is lowest')

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

normal_speed_video = np.full(num_frames, False)
high_speed_video = np.full(num_frames, True)

for cut in final_cuts:
    for frame in range(cut[0], cut[1] + 1):
        normal_speed_video[frame] = True
        high_speed_video[frame] = False

unwanted_frames = np.extract(high_speed_video, range(num_frames))

# build new audio frames

newAudio = np.extract(np.repeat(normal_speed_video, sounds_per_frame), data)

wavfile.write(videofile_path + temp_dir + 'newAudio.wav', fs, newAudio)

# build new video frames

for old_id in unwanted_frames:
    subprocess.call(
        ['rm', videofile_path + temp_dir + '$old_frames%06d.jpg' %(old_id + 1)]
    )

# build remastered video

subprocess.call(
    ['ffmpeg', '-framerate', str(fps), '-pattern_type', 'glob', '-i', videofile_path + temp_dir + '$old_frames*.jpg', '-i', videofile_path + temp_dir + 'newAudio.wav', '-strict', '-2', videofile_path + videofile_name + '_remastered' + videofile_ext]
)

# ffmpeg -framerate fps -i temp/$new_frames%06d.jpg -i temp/newAudio.wav -strict -2 "blank_remastered.mp4"

#remove useless files

subprocess.call(
    ['rm', '-r', '-I', videofile_path + temp_dir]
)
