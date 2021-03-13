import numpy as np

def extract_roi(frame,coord,from_center=True):
    (x,y,w,h) = coord
    if from_center:
        roi = frame[y-h//2:y+h//2, x-w//2:x+w//2]
    else:
        roi = frame[y:y+h, x:x+w]
    return roi

def insert_roi(frame,roi,coord,from_center=True):
    (x,y,w,h) = coord
    if from_center:
        frame[y-h//2:y+h//2, x-w//2:x+w//2] = roi
    else:
        frame[y:y+h, x:x+w] = roi
    return frame
        