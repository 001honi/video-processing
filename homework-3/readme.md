<br />
<p align="center">
  <h1 align="center">Motion Parameters</h1>
  
  <h2 align="center">Estimation of 2D Affine Motion Parameters</h2>

  <p align="center"><a href="https://001honi.github.io/repos/video-processing/affine-motion/report.html"><strong>Follow the report »</strong></a>
  </p>
</p>

## About

Optical flow vectors (u,v) were calculated over Horn & Schunck algorithm. Then, least square approximation was applied to estimate 6-parameters of the affine motion model. Significant point here is the selection of appropriate pixels to solve the linear system.

I have used non-intelligent or in other words video-specific method to pick the relavant pixels. The woody region in the scene is static but appears as moving in the flow field due to camera motion. I thought, sampling pixels from this woody region is the simplest way to reach to the camera motion parameters.

Last, motion compensation section is removed due to its poor performance. However, it can be found a sample in videos folder.

Sorry for unorganized scripts, this time.

## Demo

![Demo](videos/demo.gif)

## Credits

Utilized from [OpticalFlow_Visualization](https://github.com/tomrunia/OpticalFlow_Visualization) by Tom Runia (tomrunia@github) for optical flow field colorization. <br>
Revision: [Not officially merged pull request](https://github.com/tomrunia/OpticalFlow_Visualization/pull/7) by Kedar (kedartatwawadi@github)

