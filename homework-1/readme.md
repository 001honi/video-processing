# Homework-1
## Block Matching Algorithm

### Implementation
<ul>
            <li><strong>dfd :</strong> {0:MAD, 1:MSE} Displaced frame difference </li>
            <li><strong>blockSize :</strong> (sizeH,sizeW) </li>
            <li><strong>searchMethod :</strong> {0:Exhaustive, 1:Three-Step} </li>
            <li><strong>searchRange :</strong> (int) +/- pixelwise range </li>
            <li><strong>motionIntensity:</strong> True (default) <br>
               <i>Normalization for motion vector intensities. Assigns 255 to the largest amplitude motion vector. 
                 Also, there is a threshold that intensity value cannot be less than 100.</i></li>
</ul>  

### Sample Result

![sample](videos/sample.gif) <br>
