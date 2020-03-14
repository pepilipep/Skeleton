import subprocess

import numpy as np

import subprocess
import os
import sys

import cv2

import math

from scipy.io import wavfile

#VIDEO INFO

videofile_path = './'
videofile_name = 'L5a_bash_expansions_22.03.2018'
videofile_ext = '.webm'

temp_dir = 'temp/'

videofile = sys.argv[1]
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

min_sound = 0.055

min_time = 0.15

if len(sys.argv) >= 3:
    min_sound = int(sys.argv[2])

#INIT

subprocess.call(
    ['mkdir', videofile_path + temp_dir]
)

#get info


cv2video = cv2.VideoCapture(videofile_path + videofile_name + videofile_ext)
fps = cv2video.get(cv2.CAP_PROP_FPS)
num_frames = int(cv2video.get(cv2.CAP_PROP_FRAME_COUNT))
time_video = num_frames / fps

print(fps, num_frames, time_video)

#get audio

subprocess.call(
    ['ffmpeg', '-i', videofile_path + videofile_name + videofile_ext, '-codec:a', 'pcm_s16le', '-ac', '1', videofile_path + temp_dir + videofile_name + '.wav'])
print('file saved')

fs, data = wavfile.read(videofile_path + temp_dir + videofile_name + '.wav')
maxaudio_volume = np.max(data)

sounds_per_frame = math.ceil(data.shape[0] / num_frames)

#get frames as images

subprocess.call(
    ['ffmpeg', '-i', videofile_path + videofile_name + videofile_ext, '-qscale:v', '3', videofile_path + temp_dir + '$old_frames%06d.jpg', '-hide_banner']
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
