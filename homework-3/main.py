# Title : Affine Motion Parameters Estimation
# Author: Selahaddin HONI | 001honi@github
# April, 2021
# --------------------------------------------------------------------------------------------
# Credits: Tom Runia (2018) flow_wis :: Flow field colorization functions | tomrunia@github
# ============================================================================================
from video import Video
from horn_schunck import HornSchunck
from motion_params import AffineMotionParams
from tqdm import tqdm
import numpy as np
import cv2
import tqdm
import time

# Initialize Optical Flow Object (Horn & Schunck) 
# ============================================================================================
# Horn & Schunck Parameters
alpha = 1 ; maxIter = 20 ; applyGauss = False ; GaussKernel = (5,5)

optFlow = HornSchunck(alpha=alpha,maxIter=maxIter)
optFlow.applyGauss  = applyGauss
optFlow.GaussKernel = GaussKernel


# Import Video Sequence
# ============================================================================================
path_inp = "videos/coast_guard.mp4"

video = Video(path_inp)    ; video2 = Video(path_inp)         
video.read_frames(gray=True,rescale=False)
(H,W) = video.shape

title = f"AffineMotion"
path_out = f"videos/{title}.mp4"
print(title)

# Demo 
# ============================================================================================
# a = 80 ; t = 81

# start_time = time.time()
# anchor = video.frames_inp[a]
# target = video.frames_inp[t]

# motionParam = AffineMotionParams()

# (u,v) = optFlow.estimate(anchor,target,plotErr=False)

# # select the first 80 rows of pixels in flow field
# # 'the woody region in this specific video sample' 
# u_ = u[:80,:] ; v_ = v[:80,:]
# # convert to polar coordinates 
# (mag,ang) = cv2.cartToPolar(u_,v_)
# # pick the pixel coordinates having greater magnitude 
# idx = np.where(mag>0.02)
# # make a tuple from these pixels as 
# # pxs=((px1_x,px1_y), (px2_x,px2_y), ...)
# pxs = tuple(zip(idx[1],idx[0]))

# motionParam.Set(u,v)
# for px in pxs:
#     motionParam.AppendPixel(px)
# motionParam.Estimate()

# (uu,vv) = motionParam.Compensate()
# visual_params = motionParam.VisualParams()

# flow1 = optFlow.getFlowField(flow_norm=0.7)
# pred1 = optFlow.predict(prevent_black=False)

# flow2 = optFlow.getFlowField(uu,vv,flow_norm=0.7)
# pred2 = optFlow.predict(uu,vv,prevent_black=False)

# # collage = optFlow.getCollage2(visual_params)
# collage = motionParam.getCollage(flow1,flow2,pred1,pred2)

# elapsed_time = time.time() - start_time
# print(f"Elapsed time: {elapsed_time:.3f} secs")

# cv2.imshow("Demo",collage)
# cv2.waitKey(0)
# cv2.imwrite("demo.png",collage)


# Video Process
# ============================================================================================
start_time = time.time()

motionParam = AffineMotionParams()

for f in tqdm.tqdm(range(video.total_frame-1)):
# for f in range(video.total_frame-1):
    if f < 50 or f > 150:
        continue

    a = f; t = f+1
    anchor = video.frames_inp[a]
    target = video.frames_inp[t]

    (u,v) = optFlow.estimate(anchor,target)

    # select the first 100 rows of pixels in flow field
    # 'the woody region in this specific video sample' 
    u_ = u[:100,:] ; v_ = v[:100,:]
    # convert to polar coordinates 
    (mag,ang) = cv2.cartToPolar(u_,v_)
    # pick the pixel coordinates having greater magnitude 
    idx = np.where(mag>0.1)
    # make a tuple from these pixels as 
    # pxs=((px1_x,px1_y), (px2_x,px2_y), ...)
    pxs = tuple(zip(idx[1],idx[0]))

    motionParam.Set(u,v)
    for px in pxs:
        motionParam.AppendPixel(px)
    motionParam.Estimate()
    (uu,vv) = motionParam.Compensate()
    visual_params = motionParam.VisualParams()  

    flow1 = optFlow.getFlowField(flow_norm=1)
    pred1 = optFlow.predict(prevent_black=False)

    collage2 = optFlow.getCollage2(visual_params)

    # flow2 = optFlow.getFlowField(uu,vv,flow_norm=1)
    # pred2 = optFlow.predict(uu,vv,prevent_black=False)

    # collage = motionParam.getCollage(flow1,flow2,pred1,pred2)


    # video.frames_out.append(collage)
    video2.frames_out.append(collage2)

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} secs")
# video.write("videos/compens_collage.mp4",fps=30,gray2bgr=False)
# video.write("videos/compens_collage_slow.mp4",fps=10,gray2bgr=False)
video2.write("videos/motion_type.mp4",fps=20,gray2bgr=False)
