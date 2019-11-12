# Multiple-Object-Detection-YOLO

Object Detection on video streams and webcam footage using YOLOv3, a single stage detector trained on the COCO dataset, and OpenCV.

### Requirements:
`numpy`, `argparse`, `imutils`, `time`, `cv2` , `os`

### Sample Usage:
#### 1. To detect objects in a video from the disk; 
use the `yolodetection.py` file

Example (in the terminal): 

`python yolodetection.py --input videoname.mp4 --output videoname.mp4 --yolo yolo-coco`

<a href="https://github.com/skhiearth/skhiearth.github.io/blob/master/images/timessquare.gif?raw=true" target="_blank">
<img src="https://github.com/skhiearth/skhiearth.github.io/blob/master/images/timessquare.gif?raw=true" 
alt="Object Detection" width="400" height="300" border="10" /></a>

#### 2. To detect objects in a webcam video; 
use the `yolowebcam.py` file

Example (in the terminal): 

`python yolowebcam.py --yolo yolo-coco`

**NOTE: Don't change the `yolo-coco` folder downloaded from the repo. It contains the weights and class labels required for the object detection.**
