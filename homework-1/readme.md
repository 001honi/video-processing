<br />
<p align="center">
  <h1 align="center">Motion Estimation</h2>
  
  <h2 align="center">Block Matching Algorithm</h2>

  <p align="center"><a href="https://001honi.github.io/repos/video-processing/block-matching/report.html"><strong>Follow the report »</strong></a>
  </p>
</p>

## Usage

_BlockMatching()_ class is written in Python. It takes 5 arguments:
<ul>
            <li><strong>dfd :</strong> {0:MAD, 1:MSE} Displaced frame difference </li>
            <li><strong>blockSize :</strong> (sizeH,sizeW) </li>
            <li><strong>searchMethod :</strong> {0:Exhaustive, 1:Three-Step} </li>
            <li><strong>searchRange :</strong> (int) +/- pixelwise range </li>
            <li><strong>motionIntensity:</strong> True (default) <br>
               <i>Normalization for motion vector intensities. Assigns 255 to the largest amplitude motion vector. 
                 Also, there is a threshold that intensity value cannot be less than 100.</i></li>
</ul>  
In main script, there are two more parameters to control the predictions:
<ul>
            <li><strong>predict_from_prev :</strong> False (default) </li>
            <li><strong>N:</strong> 5 (default) <br>
               <i>If the predictions made from previous predicted frames, anchor is updated after N frames.</i></li>
</ul>  

Initialize an object of **BlockMatching()** class with an arbitrary name, **bm()** is suggested.
  ```py
 bm = BlockMatching(dfd=dfd,
            blockSize=blockSize,
            searchMethod=searchMethod,
            searchRange=searchRange,
            motionIntensity=False)
  ```
  Then use **step()** method to run the program. After execution, you can reach the generated motion field via **bm.motionField** and the predicted anchor **bm.anchorP** properties.
  ```py
  bm.step(anchor,target)
  motionField = bm.motionField
  anchorP = bm.anchorP
  ```
Use **visualize()** method of **Video()** class to put the 4 frames together to show a collage. Here, **a** and **t** arguments are the frame numbers of the anchor and target frames, respectively.
  ```py
 collage = video.visualize(anchor,target,motionField,anchorP,text,a,t)
  ```

## Demo 

![sample](videos/sample.gif) <br>
