from video import Video
from block_matching import BlockMatching
from tqdm import tqdm
import numpy as np
import cv2
import tqdm
import time

# Block Matching Parameters
# ============================================================================================
dfd=1 ; blockSize=(16,16) ; searchMethod=0 ; searchRange=15

bm = BlockMatching(dfd=dfd,
            blockSize=blockSize,
            searchMethod=searchMethod,
            searchRange=searchRange,
            motionIntensity=True)

# Title and Parameters Info    
dfd = "MSE" if dfd else "MAD"
method = "Exhaustive" if not searchMethod else "ThreeStep"
text = ["Block Matching Algorithm","DFD: {} | {} | {} Search Range: {}".format(
    dfd,blockSize,method,searchRange)]
print(text)

# Import Video Sequence
# ============================================================================================
gray = True ; predict_from_prev = False 

predict = "prev" if predict_from_prev else "orig"
path_inp = "videos/foreman-orig.avi"
path_out = "videos/{}-Size{}-{}-{}-{}.mp4".format(dfd,blockSize[0],method,searchRange,predict)

video = Video(path_inp)         
video.read_frames(gray=gray)
(H,W) = video.shape

# Demo 
# ============================================================================================
anchor = video.frames_inp[72]
target = video.frames_inp[78]

bm.step(anchor,target)

anchorP = bm.anchorP
motionField = bm.motionField

out = video.visualize(anchor,target,motionField,anchorP,text)
cv2.imshow("Demo", out)
cv2.waitKey(0)

# Video Process
# ============================================================================================

start_time = time.time()

prev_prediction = None
for f in tqdm.tqdm(range(video.total_frame-1)):

    if predict_from_prev:
        anchor = video.frames_inp[f] if f==0 else prev_prediction
    else:
        anchor = video.frames_inp[f]
    target = video.frames_inp[f+1]

    bm.step(anchor,target)

    anchorP = bm.anchorP
    motionField = bm.motionField

    out = video.visualize(anchor,target,motionField,anchorP,text)
    video.frames_out.append(out)

    if predict_from_prev:
        prev_prediction = anchorP

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} secs")
video.write(path_out,fps=30)
print(path_out)

