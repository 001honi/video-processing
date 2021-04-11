# Title : Implementation of Horn and Schunck. (1981). 'Determining Optical Flow'
# Author: Selahaddin HONI | 001honi@github
# April, 2021
# --------------------------------------------------------------------------------------------
# Credits: Tom Runia (2018) flow_wis :: Flow field colorization functions | tomrunia@github
# ============================================================================================
from video import Video
from horn_schunck import HornSchunck
from tqdm import tqdm
import numpy as np
import cv2
import tqdm
import time

# Initialize Optical Flow Object (Horn & Schunck) 
# ============================================================================================
# Horn & Schunck Parameters
alpha = 10 ; maxIter = 20 ; applyGauss = True ; GaussKernel = (5,5)

optFlow = HornSchunck(alpha=alpha,maxIter=maxIter)
optFlow.applyGauss  = applyGauss
optFlow.GaussKernel = GaussKernel


# Import Video Sequence
# ============================================================================================
path_inp = "videos/original.mp4"

video = Video(path_inp)         
video.read_frames(gray=True,rescale=False)
(H,W) = video.shape

title = f"alpha{alpha}"
title = title + f"-GaussKernel{GaussKernel[0]}" if applyGauss else title
path_out = f"videos/{title}.mp4"
print(title)

# Demo 
# ============================================================================================
a = 0 ; t = 2

start_time = time.time()
anchor = video.frames_inp[a]
target = video.frames_inp[t]

(u,v) = optFlow.estimate(anchor,target,plotErr=False)

flowField = optFlow.getFlowField()
anchorP   = optFlow.predict()
collage   = optFlow.getCollage(a,t)

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} secs")

cv2.imshow("Demo",collage)
cv2.waitKey(0)
cv2.imwrite("demo.png",collage)


# Video Process
# ============================================================================================
# start_time = time.time()

# for f in tqdm.tqdm(range(video.total_frame-1)):

#     a = f; t = f+1
#     anchor = video.frames_inp[a]
#     target = video.frames_inp[t]

#     (u,v) = optFlow.estimate(anchor,target)

#     flowField = optFlow.getFlowField()
#     anchorP   = optFlow.predict(prevent_black=False)
#     collage   = optFlow.getCollage(a,t)

#     video.frames_out.append(collage)

# elapsed_time = time.time() - start_time
# print(f"Elapsed time: {elapsed_time:.3f} secs")
# video.write(path_out,fps=30,gray2bgr=False)