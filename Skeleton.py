import ffmpeg
import subprocess

import numpy as np

import subprocess
import os
import sys

import cv2

import math

from scipy.io import wavfile

#VIDEO INFO

videofile_path = '/home/user/Documents/Skeleton/'
videofile_name = 'L5a_bash_expansions_22.03.2018'
videofile_ext = '.webm'

#MAGICS

min_sound = 2300

min_time = 0.4

adjust_time = 2.3

#INIT

#make the file in mst


try:
    f = open(videofile_path + videofile_name + '.MTS')
    f.close()
    print('mts file already exists')
except IOError:
    subprocess.call(
        ['ffmpeg', '-i', videofile_path + videofile_name + videofile_ext, '-q', '0', videofile_path + videofile_name + '.MTS']
    )



#get info


cv2video = cv2.VideoCapture(videofile_path + videofile_name + videofile_ext)
fps = cv2video.get(cv2.CAP_PROP_FPS)
num_frames = int(cv2video.get(cv2.CAP_PROP_FRAME_COUNT))
time_video = num_frames / fps

print(fps, num_frames, time_video)

#get audio

subprocess.call(
    ['ffmpeg', '-i', videofile_path + videofile_name + '.MTS', '-codec:a', 'pcm_s16le', '-ac', '1', videofile_path + videofile_name + '.wav'])
print('file saved')

fs, data = wavfile.read(videofile_path + videofile_name + '.wav')
sounds_per_frame = math.ceil(data.shape[0] / num_frames)

#algorithm

gone = np.zeros(num_frames)

data.resize((sounds_per_frame + 1) * num_frames)

sound_split_by_frame = np.absolute(np.reshape(data, (num_frames, sounds_per_frame + 1)))

total_sound_by_frames = np.max(sound_split_by_frame, axis=-1)

gone = total_sound_by_frames < min_sound

print(np.sum(gone))

cuts = []
if not gone[0]:
    cuts.append([0, 0])

for i in range(len(gone[1:])):
    if not gone[i] and not gone[i - 1]:
        cuts[-1][1] = (i + 1) * time_video / num_frames
    elif not gone[i]:
        cuts.append([i * time_video / num_frames, (i + 1) * time_video / num_frames])

mid_cuts = []
final_cuts = []

mid_cuts.append(cuts[0])

for i in range(len(cuts) - 1):
    if cuts[i + 1][0] - mid_cuts[-1][1] < min_time + 0.2:
        mid_cuts[-1][1] = cuts[i + 1][1]
    else:
        mid_cuts.append(cuts[i + 1])

for cut in mid_cuts:
    if cut[1] - cut[0] >= min_time:
        final_cuts.append(cut)

#write cuts in file

f = open(videofile_path + videofile_name + '_specs.txt', 'w+')
for cut in final_cuts:
    f.write('file \'' + videofile_name + '.MTS' + '\'\n')
    f.write('inpoint %.2f\n' % (cut[0] - min_time / 4))
    f.write('outpoint %.2f\n' % (cut[1] + min_time / 4))

f.close()

#cut and concatenate the segments

subprocess.call(
    ['ffmpeg', '-f', 'concat', '-i', videofile_path + videofile_name + '_specs.txt', '-c', 'copy', videofile_path + videofile_name + '_remastered.MTS']
)


#ffmpeg -f concat -i input.txt -c copy -fflags +genpts -avoid_negative_ts make_zero output.mp4

#remove useless files

subprocess.call(
    ['rm', videofile_path + videofile_name + '.wav']
)

subprocess.call(
   ['rm', videofile_path + videofile_name + '_specs.txt']
)



# cuts_string = "select=\'"
# for i in range(len(cuts)):
#     cuts_string += 'between(t,' + str(round(cuts[i][0], 2)) + ',' + str(round(cuts[i][1], 2)) + ')'
#     if i < len(cuts) - 1:
#         cuts_string += '+'

# cuts_string += '\''

#subprocess.call(
#    ['ffmpeg', '-i', videofile_path + videofile_name + videofile_ext, '-vf', cuts_string + ',setpts=N/FRAME_RATE/TB', '-af', 'a' + cuts_string + ',asetpts=N/SR/TB', videofile_path + videofile_name + "_remastered.mp4"]
#)

#ffmpeg -i video -vf "select='between(t,4,6.5)+between(t,17,26)+between(t,74,91)',setpts=N/FRAME_RATE/TB" 
# -af "aselect='between(t,4,6.5)+between(t,17,26)+between(t,74,91)',asetpts=N/SR/TB" out.mp4

