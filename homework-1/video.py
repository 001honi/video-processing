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
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not self.shape[0]:
                self.shape = frame.shape[:2]
                       
            self.frames_inp.append(frame)
        
        print("[INFO] Video Import Completed")

    def visualize(self,target,anchor,motionField,anchorP,text):
        h = 85 ; w = 25
        H,W = anchor.shape
        HH,WW = 3*H-20, 2*W+35
        frame = np.ones((HH,WW), dtype="uint8")*255

        cv2.putText(frame, text[0], (10, 25), Video.FONT, 0.7, 0, 1)
        cv2.putText(frame, text[1], (10, 45), Video.FONT, 0.3, 0, 1)
        cv2.line(frame, (10, 52),(WW-10,52),0)

        cv2.putText(frame, "target", (10, h-10), Video.FONT, 0.4, 0, 1)
        cv2.putText(frame, "anchor", (W+w, h-10), Video.FONT, 0.4, 0, 1)
        cv2.putText(frame, "motion field", (10, h+H+15), Video.FONT, 0.4, 0, 1)
        cv2.putText(frame, "predicted anchor", (W+w, h+H+15), Video.FONT, 0.4, 0, 1)

        frame[h:h+H, 10:10+W] = target 
        frame[h:h+H, W+w:2*W+w] = anchor 
        frame[h+H+25:h+2*H+25, 10:10+W] = motionField 
        frame[h+H+25:h+2*H+25, W+w:2*W+w] = anchorP 

        return frame

    def write(self, path="out.avi",fps=30,gray=False):            
        H,W = self.frames_out[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path,fourcc,fps,(W,H),True)
        for frame in self.frames_out:
            if gray:
                frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
            writer.write(frame)
        print("[INFO] Video Export Completed")
        