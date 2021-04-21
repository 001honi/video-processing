import numpy as np
import numpy.matlib
import cv2

class AffineMotionParams():

    def Set(self,u,v):
        self.u = u
        self.v = v
        self.x  = []
        self.y  = []
        self.Yu = []
        self.Yv = []

    def AppendPixel(self,px):
        self.x.append(px[0]); self.y.append(px[1])
        self.Yu.append(self.u[px[1],px[0]]+px[0])
        self.Yv.append(self.v[px[1],px[0]]+px[1])

    def Estimate(self):
        assert len(self.x)  == len(self.y)
        assert len(self.Yu) == len(self.Yv)

        X = []
        for i in range(len(self.x)):
            X.append([1, self.x[i], self.y[i]])
        X = np.array(X)

        try:
            least_sq = np.linalg.inv(np.matmul(X.T,X))
        except:
            return None
        least_sq = np.matmul(least_sq,X.T)

        self.a123 = np.matmul(least_sq,self.Yu)
        self.a456 = np.matmul(least_sq,self.Yv)

        # return (self.a123,self.a456)

    def Compensate(self):
        (H,W) = self.u.shape
        a1,a2,a3 = self.a123;   a4,a5,a6 = self.a456
        uu = self.u - np.matlib.repmat(a1,H,W)
        vv = self.v - np.matlib.repmat(a4,H,W)
        return (uu,vv)
        

    def VisualParams(self, a123=None, a456=None):
        if a123 is None:
            a123 = self.a123 ; a456 = self.a456

        (HH,WW) = (50,420)
        frame = np.ones((HH, WW, 3), np.uint8)*255

        FONT = cv2.FONT_HERSHEY_SIMPLEX
        w=10;  h=20;  hh=14;  ww=100

        a1,a2,a3 = a123;   a4,a5,a6 = a456
        cv2.putText(frame, "|",(w+ww*2-15,h),FONT,0.6,0); 
        cv2.putText(frame, "|",(w+ww*2-15,h+hh),FONT,0.6,0)

        cv2.putText(frame, "a2:",(w,h),FONT,0.5,0)
        cv2.putText(frame, f"{a2: .2f}",(w+25,h),FONT,0.5,self.ColorHelper(a2),2)
        cv2.putText(frame, "a3:",(w+ww,h),FONT,0.5,0)
        cv2.putText(frame, f"{a3: .2f}",(w+ww+25,h),FONT,0.5,self.ColorHelper(a3),2)
        cv2.putText(frame, "a5:",(w,h+hh),FONT,0.5,0)
        cv2.putText(frame, f"{a5: .2f}",(w+25,h+hh),FONT,0.5,self.ColorHelper(a5),2)
        cv2.putText(frame, "a6:",(w+ww,h+hh),FONT,0.5,0)
        cv2.putText(frame, f"{a6: .2f}",(w+ww+25,h+hh),FONT,0.5,self.ColorHelper(a6),2)

        cv2.putText(frame, "a1:",(w+ww*2,h),FONT,0.5,0)
        cv2.putText(frame, f"{a1: .2f}",(w+ww*2+25,h),FONT,0.5,self.ColorHelper(a1),2)
        cv2.putText(frame, "a4:",(w+ww*2,h+hh),FONT,0.5,0)
        cv2.putText(frame, f"{a4: .2f}",(w+ww*2+25,h+hh),FONT,0.5,self.ColorHelper(a4),2)

        (motion,color) = self.CamMotionHelper()
        cv2.putText(frame, motion,(w+ww*3,h+4),FONT,0.6,color,2)
        cv2.putText(frame, "Camera Motion",(w+ww*3,h+hh+4),FONT,0.4,0)

        return frame

    def ColorHelper(self,x):
        COLOR = (0,0,0)
        if x>0.05:
            COLOR = (0,0,255)
        elif x<=-0.05:
            COLOR = (255,0,0)
        return COLOR 

    def CamMotionHelper(self):
        a1,a2,a3 = self.a123;   a4,a5,a6 = self.a456
        motion = "NONE"
        color  = (0,0,0)
        if abs(a1)>0.05:
            motion = "PAN"
            color  = self.ColorHelper(a1)
        if abs(a4)>0.05 and abs(a4)>abs(a1):
            motion = "TILT"
            color  = (0,255,0)
            # color  = self.ColorHelper(a4)
        return (motion,color)

    
    def getCollage(self,flow1,flow2,pred1,pred2):
        """ Put 4 frames together to show a collage """

        h = 70 ; w = 10
        H,W = pred1.shape
        HH,WW = h+2*H+20, 2*(W+w)
        frame = np.ones((HH, WW, 3), np.uint8)*255

        text = ["Effect of Motion Compensation", \
            "Flow fields are generated via Horn&Schunck :: alpha 1"]

        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame, text[0], (w, 23), FONT, 0.6, 0, 1)
        cv2.putText(frame, text[1], (w, 40), FONT, 0.4, 0, 1)
        cv2.line(frame, (w, 46), (WW-w, 46),0)

        cv2.putText(frame, "BEFORE Compensation", (w+W//4, h-4), FONT, 0.5, 0, 2)
        cv2.putText(frame, "AFTER Compensation", (w+W+W//4, h-4), FONT, 0.5, 0, 2)

        cv2.putText(frame, "flow field", (w, h-4), FONT, 0.4, 0, 1)
        cv2.putText(frame, "flow field", (w+W, h-4), FONT, 0.4, 0, 1)
        cv2.putText(frame, "prediction", (w, h+2*H+10), FONT, 0.4, 0, 1)
        cv2.putText(frame, "prediction", (w+W, h+2*H+10), FONT, 0.4, 0, 1)


        frame[h:h+H,       w:w+W,   :] = flow1
        frame[h:h+H,     w+W:w+2*W, :] = flow2
        frame[h+H:h+2*H,   w:w+W,   :] = cv2.cvtColor(pred1, cv2.COLOR_GRAY2BGR)
        frame[h+H:h+2*H, w+W:w+2*W, :] = cv2.cvtColor(pred2, cv2.COLOR_GRAY2BGR)

        return frame
        
