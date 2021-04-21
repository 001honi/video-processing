# Author: Selahaddin HONI | 001honi@github
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


    def read_frames(self,gray=False,rescale=0):
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
            if rescale:
                W = int(self.shape[1] * rescale / 100) 
                H = int(self.shape[0] * rescale / 100) 
                frame = cv2.resize(frame, (W,H), interpolation=cv2.INTER_AREA)
                                   
            self.frames_inp.append(frame)
        
        print("[INFO] Video Import Completed")


    def write(self, path="out.avi",fps=30,gray2bgr=False):            
        H = self.frames_out[0].shape[0]
        W = self.frames_out[0].shape[1]
        # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fourcc = -1  # for Windows machines
        writer = cv2.VideoWriter(path,fourcc,fps,(W,H),True)
        for frame in self.frames_out:
            if gray2bgr:
                frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            writer.write(frame)
        print("[INFO] Video Export Completed")
