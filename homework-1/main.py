from video import Video
from block_matching import BlockMatching
from tqdm import tqdm
import numpy as np
import cv2
import tqdm
import time

# Block Matching Parameters
dfd="MSE" ; blockSize=(16,16) ; searchMethod=0 ; searchRange=15

bm = BlockMatching(dfd=dfd,
            blockSize=blockSize,
            searchMethod=searchMethod,
            searchRange=searchRange)

# Title and Parameters Info    
method = "Exhaustive Search" if not searchMethod else "Three-Step Search"
text = ["Block Matching Algorithm",
"DFD: {} | Block Size {} | {} | Search Range: {}".format(
    dfd,blockSize,method,searchRange)]

# Video Sequence
gray = True
path_inp = "foreman-orig.avi"
path_out = "{}-size{}-{}-{}.avi".format(dfd,blockSize[0],method,searchRange)

video = Video(path_inp)         
video.read_frames(gray=gray)
(H,W) = video.shape

start_time = time.time()
for f in tqdm.tqdm(range(video.total_frame-1)):

    target = video.frames_inp[f]
    anchor = video.frames_inp[f+1]

    bm.step(anchor,target)

    anchorP = bm.anchorP
    motionField = bm.motionField

    gui = video.visualize(target,anchor,motionField,anchorP,text)
    video.frames_out.append(gui)

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} secs")
video.write(path_out,fps=12,gray=gray)


