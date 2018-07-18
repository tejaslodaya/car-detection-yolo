# YOLO

YOLO ("you only look once") is a popular algoritm because it achieves high accuracy while also being able to run in real-time, almost clocking 45 frames per second. A smaller version of the network, Fast YOLO, processes an astounding 155 frames per second while still achieving double the mAP of other real-time detectors. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

### Generic YOLO model
----------------------

<img src="https://raw.githubusercontent.com/tejaslodaya/car-detection-yolo/master/nb_images/model_architecture.png" style="width:500px;height:250;">


### YOLO model in car detection
-------------------------------

<img src="https://raw.githubusercontent.com/tejaslodaya/car-detection-yolo/master/nb_images/fig1.png" style="width:500px;height:250;">

- The **input** is a batch of images of shape (m, 608, 608, 3)
- The **output** is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers `(p_c, b_x, b_y, b_h, b_w, c)` as explained above. If you expand `c` into an 80-dimensional vector, each bounding box is then represented by 85 numbers. 
- The YOLO architecture if 5 anchor boxes are used is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85)
  <img src="https://raw.githubusercontent.com/tejaslodaya/car-detection-yolo/master/nb_images/fig2.png" style="width:420px;height:240px;">
- Each cell gives you 5 boxes. In total, the model predicts: 19x19x5 = 1805 boxes just by looking once at the image (one forward pass through the network)! That is way too many boxes. Filter the algorithm's output down to a much smaller number of detected objects. 

### Filtering
-------------
To reduce the number of detected objects, apply two techniques:
1. Score-thresholding: 
  Throw away boxes that have detected a class with a score less than the threshold
2. Non-maximum suppression (NMS):
  <img src="https://raw.githubusercontent.com/tejaslodaya/car-detection-yolo/master/nb_images/non-max-suppression.png" style="width:500px;height:400;">
  In this example, the model has predicted 3 cars, but it's actually 3 predictions of the same car. Running non-max suppression (NMS) will select only the most accurate (highest probabiliy) one of the 3 boxes.
  
  The steps to perform NMS are:
  1. Select the box that has the highest score.
  2. Compute its overlap with all other boxes, and remove boxes that overlap it more than iou_threshold.
  3. Iterate over all the boxes which have an overlap with the selected box until there are no more boxes with a lower score than the current selected box.
  
For example, suppose there are two boxes, box1 and box2. 
p(box1) = 0.9
p(box2) = 0.6
iou(box1, box2) = 0

Steps followed:
1. Select box1 since it has the highest probability among the available scores.
2. Since there's no overlap, box1 is selected.
3. Iterating, box2 has the highest probability now. Select box2.
4. End of the loop.

For a more detailed discussion, follow [issue](https://github.com/tejaslodaya/car-detection-yolo/issues/1)

### Results
-----------
Input image:
  <img src="https://raw.githubusercontent.com/tejaslodaya/car-detection-yolo/master/nb_images/prediction_input.jpg" style="width:768px;height:432px;">

Output image:
  <img src="https://raw.githubusercontent.com/tejaslodaya/car-detection-yolo/master/nb_images/prediction_output.jpg" style="width:768px;height:432px;">

### NOTE
--------
1. Training a YOLO model takes a very long time and requires a fairly large dataset of labelled bounding boxes for a large range of target classes. This project uses existing pretrained weights from the official YOLO website, and further processed using a function written by Allan Zelener.
2. Complete model architecture can be found [here](https://github.com/tejaslodaya/car-detection-yolo/blob/master/model.png)
3. Instructions to generate `yolo.h5` file can be found [here](https://github.com/allanzelener/YAD2K). Place that in `model_data` folder
4. Input images can be found in `images` directory. Corresponding output images can be found in `out` directory.

### References
--------------
1. Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - You Only Look Once: Unified, Real-Time Object Detection (2015)
2. Joseph Redmon, Ali Farhadi - YOLO9000: Better, Faster, Stronger (2016)
3. Allan Zelener - YAD2K: Yet Another Darknet 2 Keras
4. The official YOLO website (https://pjreddie.com/darknet/yolo/)




