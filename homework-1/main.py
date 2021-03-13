from video import Video
from block_matching import BlockMatching
from tqdm import tqdm
import numpy as np
import cv2
import tqdm

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
path_out = "{}-{}-{}-{}.avi".format(dfd,blockSize,method,searchRange)

video = Video(path_inp)         
video.read_frames(gray=gray)
(H,W) = video.shape
video.total_frame = 50
for f in tqdm.tqdm(range(video.total_frame-1)):

    target = video.frames_inp[f]
    anchor = video.frames_inp[f+1]

    bm.step(anchor,target)

    anchorP = bm.anchorP
    motionField = bm.motionField

    gui = video.visualize(target,anchor,motionField,anchorP,text)
    video.frames_out.append(gui)

video.write(path_out,fps=30,gray=gray)


