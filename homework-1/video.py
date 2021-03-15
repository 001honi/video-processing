# Author: Selahaddin HONI
# March, 2021
# ============================================================================================
import numpy as np
import cv2

class Video():
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    BLUE = (255,0,0) 
    GREEN= (0,255,0)
    RED  = (0,0,255)

    def __init__(self,path):
        self.path = path
        self.frames_inp = []
        self.frames_out = []
        self.total_frame = None
        self.shape = (None,None) # H,W
        self.gray = False

    def read_frames(self,gray=False):
        """
        stores all the frames in the given video source in 
            self.frames_inp (list) as [frame0, frame1, ...]
        where frame# is numpy array
        """
        try: 
            source = cv2.VideoCapture(self.path)
            prop = cv2.CAP_PROP_FRAME_COUNT 
            self.total_frame = int(source.get(prop)) 
        except:
            print("Error in Path or Frame Count")
            exit()

        for i in range(self.total_frame):
            ret, frame = source.read()
            if not (ret or frame):
                print("Error in Frame Read")
                break
            if gray:
                self.gray = True
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not self.shape[0]:
                self.shape = frame.shape[:2]
                       
            self.frames_inp.append(frame)
        
        print("[INFO] Video Import Completed")

    def visualize(self,anchor,target,motionField,anchorP,text):
        """Put 4 frames together to show gui."""

        h = 70 ; w = 10
        H,W = anchor.shape
        HH,WW = h+2*H+20, 2*(W+w)
        frame = np.ones((HH,WW), dtype="uint8")*255

        cv2.putText(frame, text[0], (w, 23), Video.FONT, 0.5, 0, 1)
        # cv2.line(frame, (w, 27), (WW-w, 27),0)

        cv2.putText(frame, text[1], (w, 40), Video.FONT, 0.4, 0, 1)
        cv2.line(frame, (w, 46), (WW-w, 46),0)
        # cv2.line(frame, (w, h+2*H+20), (WW-w, h+2*H+20),0)

        cv2.putText(frame, "anchor", (w, h-4), Video.FONT, 0.4, 0, 1)
        cv2.putText(frame, "target", (w+W, h-4), Video.FONT, 0.4, 0, 1)
        cv2.putText(frame, "motion field", (w, h+2*H+10), Video.FONT, 0.4, 0, 1)
        cv2.putText(frame, "predicted anchor", (w+W, h+2*H+10), Video.FONT, 0.4, 0, 1)

        frame[h:h+H, w:w+W] = anchor 
        frame[h:h+H, w+W:w+2*W] = target 
        frame[h+H:h+2*H, w:w+W] = motionField 
        frame[h+H:h+2*H, w+W:w+2*W] = anchorP 

        return frame

    def write(self, path="out.avi",fps=30):            
        H,W = self.frames_out[0].shape
        # fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        writer = cv2.VideoWriter(path,-1,fps,(W,H),True)
        for frame in self.frames_out:
            if self.gray:
                frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
            writer.write(frame)
        print("[INFO] Video Export Completed")
        