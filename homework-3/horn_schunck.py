# Title : Implementation of Horn and Schunck. (1981). 'Determining Optical Flow'
# Author: Selahaddin HONI | 001honi@github
# April, 2021
# --------------------------------------------------------------------------------------------
# Credits: Tom Runia (2018) flow_wis :: Flow field colorization functions | tomrunia@github
# ============================================================================================
import numpy as np
import cv2
from numpy import multiply as mul
from scipy.signal import convolve2d
import flow_vis


class HornSchunck():
    """ 
    Implementation of original paper 
    Horn and Schunck. (1981). 'Determining Optical Flow'

    Args:

        alpha   : the regularization coefficient of smoothness constraint
        maxIter : number of iterations required to update new flow velocities 
        applyGauss  : apply Gaussian smoothing as a preprocess {False by default}
        GaussKernel : Gaussian Filter kernel size {(15,15) by default}

    Returns:

        (u,v) : optical flow velocities
    """

    # Kernels for Partial Derivative estimates
    kernelX = 0.25 * np.array([[-1,1],[-1,1]])
    kernelY = 0.25 * np.array([[-1,-1],[1,1]])
    kernelT = 0.25 * np.array([[1,1],[1,1]])  #-0.25 for frame1 
    
    # Kernel for Laplacian of the Flow Velocities estimate
    kernelL = np.array([[1/12,1/6,1/12],[1/6,-1,1/6],[1/12,1/6,1/12]])


    def __init__(self,alpha=0.01,maxIter=20):

        self.alpha   = alpha
        self.maxIter = maxIter
        self.nIter   = 0

        self.applyGauss  = False
        self.GaussKernel = (15,15)

        self.anchor  = None
        self.target  = None
        self.anchorP = None
        
        self.flowField = None

        self.Ex = None 
        self.Ey = None 
        self.Et = None 

        self.u = None
        self.v = None



    def estimate(self,frame1,frame2,plotErr=False):

        # Store original frames 
        self.anchor = frame1
        self.target = frame2

        # Apply Gaussian smoothing for both frames
        if self.applyGauss:
            frame1 = cv2.GaussianBlur(frame1,self.GaussKernel,0)
            frame2 = cv2.GaussianBlur(frame2,self.GaussKernel,0)

        # Precompute image and temporal derivatives Ex,Ey,Et
        self.estimateDerivatives(frame1,frame2)

        # Initialize flow field
        self.u = np.zeros_like(frame1)  
        self.v = np.zeros_like(frame1)  

        # Store relative errors
        err = [1]

        # Iterative Solution:
        for i in range(self.maxIter):
            # Compute new set of velocity estimates from the estimated
            # derivatives and the avg. of the prev. velocity estimates

            # Estimating the Laplacian of the Flow Velocities
            u_ = self.estimateAvgFlow(self.u)
            v_ = self.estimateAvgFlow(self.v)

            # New set of velocity estimates
            (u, v) = self.estimateNewFlow(u_,v_) 

            # Calculate relative error
            e = np.mean(np.absolute(u-self.u)) / np.mean(np.absolute(u))
            # if error starts to grow : break the loop
            if e > err[-1]:
                self.nIter = i
                break

            # Update new estimates
            self.u, self.v = (u,v) 

            # Store error value 
            err.append(e)

        if plotErr:
            import matplotlib.pyplot as plt
            plt.bar(list(range(i)),err[1:])
            plt.title("Relative Error for Velocity: 'u'")
            plt.xlabel("Iteration")
            plt.ylabel("Relative Error")
            plt.xticks(list(range(i)))
            plt.show()
        return (self.u, self.v)



    def estimateDerivatives(self,frame1,frame2):
        """ Estimating the Partial Derivatives """

        self.Ex = convolve2d(frame1,HornSchunck.kernelX,mode='same') + \
            convolve2d(frame2,HornSchunck.kernelX,mode='same')
        self.Ey = convolve2d(frame1,HornSchunck.kernelY,mode='same') + \
            convolve2d(frame2,HornSchunck.kernelY,mode='same')
        self.Et = - convolve2d(frame1,HornSchunck.kernelT,mode='same') + \
            convolve2d(frame2,HornSchunck.kernelT,mode='same')



    def estimateAvgFlow(self,u):
        """ Estimating the Laplacian of the Flow Velocities for both u and v"""

        return convolve2d(u,HornSchunck.kernelL,mode='same')



    def estimateNewFlow(self,u_,v_):
        """ Iterative update rule for new-flow velocities """

        nom = mul(self.Ex,u_) + mul(self.Ey,v_) + self.Et
        den = (self.alpha)**2 + mul(self.Ex,self.Ex) + mul(self.Ey,self.Ey)
        div = np.divide(nom,den)

        u = u_ - mul(self.Ex,div)
        v = v_ - mul(self.Ey,div)

        return (u,v)



    def predict(self,u=None,v=None,prevent_black=True):
        """" Predict target frame from anchor using flow field velocities """

        if u is None:
            u=self.u ; v=self.v

        anchorP = np.zeros_like(self.anchor)
        if prevent_black:
            anchorP = self.anchor.copy() 

        for x in range(anchorP.shape[1]):
            for y in range(anchorP.shape[0]):

                # for each pixel extract the u,v components
                # then round to nearest integer value
                dx = round(u[y,x])
                dy = round(v[y,x])

                # select this pixel from anchor and move it
                # (x+dx, y+dy) in predicted anchor

                # if dx and/or dy are both zero : continue
                if prevent_black:
                    if not dx:
                        if not dy:
                            continue

                # if movement out of bound : continue
                if (x+dx)<0 or (y+dy)<0 or \
                    (x+dx)>=anchorP.shape[1] or \
                        (y+dy)>=anchorP.shape[0]:
                    continue

                anchorP[y+dy,x+dx] = self.anchor[y,x]

        self.anchorP = anchorP
        return self.anchorP



    def getFlowField(self,u=None,v=None,flow_norm=None):
        """ Flow-field colorization with flow_vis by Tom Runia """
        if u is None:
            u=self.u ; v=self.v
        self.flowField = flow_vis.flow_to_color(u,v,convert_to_bgr=True,flow_norm=flow_norm)
        return self.flowField


    def getCollage(self,a,t):
        """ Put 4 frames together to show a collage """

        h = 70 ; w = 10
        H,W = self.anchor.shape
        HH,WW = h+2*H+20, 2*(W+w)
        frame = np.ones((HH, WW, 3), np.uint8)*255

        info = f"alpha: {self.alpha} | iter: {self.nIter}"
        info = info + f" | GaussKernel{self.GaussKernel}" if self.applyGauss else info
        text = ["Horn & Schunck Algorithm", info]

        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame, text[0], (w, 23), FONT, 0.5, 0, 1)
        cv2.putText(frame, text[1], (w, 40), FONT, 0.4, 0, 1)
        cv2.line(frame, (w, 46), (WW-w, 46),0)

        cv2.putText(frame, f"anchor-{a:03d}", (w, h-4), FONT, 0.4, 0, 1)
        cv2.putText(frame, f"target-{t:03d}", (w+W, h-4), FONT, 0.4, 0, 1)
        cv2.putText(frame, "flow field", (w, h+2*H+10), FONT, 0.4, 0, 1)
        cv2.putText(frame, "predicted anchor", (w+W, h+2*H+10), FONT, 0.4, 0, 1)

        frame[h:h+H,       w:w+W,   :] = cv2.cvtColor(self.anchor , cv2.COLOR_GRAY2BGR) 
        frame[h:h+H,     w+W:w+2*W, :] = cv2.cvtColor(self.target , cv2.COLOR_GRAY2BGR) 
        frame[h+H:h+2*H,   w:w+W,   :] = self.flowField
        frame[h+H:h+2*H, w+W:w+2*W, :] = cv2.cvtColor(self.anchorP, cv2.COLOR_GRAY2BGR) 

        return frame

    
    def getCollage2(self,visual_params):
        """ Returns combined original frame and its flow field """

        h = 70; w = 10
        H,W = self.anchor.shape
        HH,WW = h+H+50, 2*(W+w)
        frame = np.ones((HH, WW, 3), np.uint8)*255

        info = f"Horn & Schunck Flow :: alpha {self.alpha}"
        text = ["Affine Motion Parameter Estimation", info]

        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame, text[0], (w, 23), FONT, 0.5, 0, 1)
        cv2.putText(frame, text[1], (w, 40), FONT, 0.4, 0, 1)
        cv2.line(frame, (w, 46), (WW-w, 46),0)


        cv2.putText(frame, "original", (w, h-4), FONT, 0.4, 0, 1)
        cv2.putText(frame, "optical flow", (w+W, h-4), FONT, 0.4, 0, 1)

        frame[h:h+H,       w:w+W,   :] = cv2.cvtColor(self.anchor , cv2.COLOR_GRAY2BGR) 
        frame[h:h+H,     w+W:w+2*W, :] = self.flowField
        frame[h+H:h+H+50,  w:w+420, :] = visual_params

        return frame

