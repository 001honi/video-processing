from utils import extract_roi, insert_roi
import numpy as np
import cv2
import itertools

class Block():
    x_min, x_max = 0, 0
    y_min, y_max = 0, 0

    def __init__(self,x,y,w,h):
        self.coord  = (x,y,w,h)
        self.center = (x,y)
        self.size   = (w,h)
        self.mv     = (0,0)
        self.matching = None

    def neighbor_block(self,dx,dy):
        (x0,y0) = self.center
        x0 = x0-dx ; y0 = y0-dy

        if x0<Block.x_min or x0>Block.x_max \
            or y0<Block.y_min or y0>Block.y_max:
            x0, y0 = None, None
        return (x0,y0)


class BlockMatching():

    def __init__(self,dfd,blockSize,searchMethod,searchRange):
        """
        dfd : Displaced frame difference => {MAD, MSE}
        blockSize : {(sizeH,sizeW)}
        searchMethod : {0:Exhaustive, 1:Three-Step}
        searchRange : (int) +/- pixelwise range 
        """
        self.p = 1 if dfd.upper() == "MAD" else 2  
        self.blockSize = blockSize
        self.searchMethod = searchMethod 
        self.searchRange = searchRange

        self.anchor = None
        self.target = None
        self.shape  = None

        self.blocks = None

        self.anchorP = None
        self.motionField = None
    
    def step(self,target,anchor):
        self.target = target
        self.anchor = anchor
        self.shape  = anchor.shape
        self.anchorP = np.zeros((self.shape),dtype="uint8")

        self.frame2blocks()

        if self.searchMethod == 0:
            self.EBMA()
        elif self.searchMethod == 1:
            pass
        else:
            print("Search Method not found!")

        self.plot_motionField()
        self.blocks2frame()     

    def frame2blocks(self):
        """Divides the frame matrix into block objects."""
        
        (H,W) = self.shape 
        (sizeH,sizeW) = self.blockSize

        Block.y_min = sizeH//2 ; Block.y_max = H - sizeH//2
        Block.x_min = sizeW//2 ; Block.x_max = W - sizeW//2

        self.blocks = []
        for h in range(H//sizeH):
            for w in range(W//sizeW):
                # initialize Block() objects with center coordinates
                x = h*sizeH + sizeH//2; y = w*sizeW + sizeW//2
                self.blocks.append(Block(y,x,sizeW,sizeH))
    
    def blocks2frame(self):
        """Construct the predicted frame from the matching blocks"""

        frame = np.zeros(self.shape,dtype="uint8")

        for block in self.blocks:
            # get block coordinates for anchor frame; then,
            # apply the matching block to output frame if any
            if block.matching is not None:
                frame = insert_roi(frame,block.matching,block.coord,from_center=True)

        self.anchorP = frame

    def plot_motionField(self):
        frame = np.zeros(self.shape,dtype="uint8")

        for block in self.blocks:
            (x1,y1) = block.center
            (x2,y2) = block.mv[0]+x1, block.mv[1]+y1
            cv2.arrowedLine(frame, (x1,y1), (x2,y2), 255, 1)
        
        self.motionField = frame
    
    def EBMA(self):
        """Exhaustive Search Algorithm"""

        dx = dy = [i for i in range(-self.searchRange,self.searchRange+1)]
        searchArea = [r for r in itertools.product(dx,dy)]

        for block in self.blocks:
            # get block coordinates and
            # extract the block from anchor frame
            block_a = extract_roi(self.anchor,block.coord,from_center=True)

            dfd_norm_min = np.Inf

            # search the matched block in target frame in search area
            for (dx,dy) in searchArea:

                (x0,y0) = block.neighbor_block(dx,dy)
                if x0 is None:
                    continue

                # extract the block from target frame
                w,h = block.size
                coord = (x0,y0,w,h)
                block_t = extract_roi(self.target,coord,from_center=True)

                # calculate displaced frame distance
                dfd = block_t - block_a
                dfd_norm = np.linalg.norm(dfd,ord=self.p) 

                if dfd_norm < dfd_norm_min:
                    block.matching = block_t
                    block.mv = (dx,dy)
                    dfd_norm_min = dfd_norm

