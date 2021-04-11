<br />
<p align="center">
  <h1 align="center">Motion Estimation</h1>
  
  <h2 align="center">Optical Flow — Horn & Schunck Algorithm</h2>

  <p align="center"><a href="https://001honi.github.io/repos/video-processing/optical-flow/report.html"><strong>Follow the report »</strong></a>
  </p>
</p>

## Usage

_HornSchunck()_ class is written in Python. It returns optical flow velocities (u,v) for given parameters:
<ul>
            <li><strong>alpha :</strong> (float) the regularization coefficient of smoothness constraint</li>
            <li><strong>maxIter :</strong> (int) maximum number of iterations required to obtain flow velocities </li>
            <li><strong>applyGauss :</strong> (bool) apply Gaussian smoothing as a preprocess {False by default} </li>
            <li><strong>GaussKernel :</strong> (tuple) Gaussian Filter kernel size {(15,15) by default}
</ul>  

Initialize an object of **HornSchunck()** class with an arbitrary name, **optFlow()** is suggested.
  ```py
  optFlow = HornSchunck(alpha=alpha,maxIter=maxIter)
  optFlow.applyGauss  = True 
  optFlow.GaussKernel = (5,5)
  ```
Then use **estimate()** method to calculate optical flow velocities (u,v). After execution, **getFlowField()** method returns the colorized flow field.
  ```py
 (u,v) = optFlow.estimate(frame1,frame2) 
 flowField = optFlow.getFlowField() 
  ```
Prediction of target frame from anchor via flow velocities is possible with **predict()** method. Moreover, **getCollage()** method puts 4 frames together to show a collage.
  ```py
 anchorP = optFlow.predict() 
 collage = optFlow.getCollage(f1_idx,f2_idx)
  ```

## Reference Paper

[1] Horn and Schunck. (1981). [Determining Optical Flow](https://www.researchgate.net/publication/222450615_Determining_Optical_Flow)


## Credits

Utilized from [OpticalFlow_Visualization](https://github.com/tomrunia/OpticalFlow_Visualization) by Tom Runia (tomrunia@github) for optical flow field colorization.
