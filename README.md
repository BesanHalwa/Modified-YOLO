# Modified-YOLO

We take inspiration from convolution operation and present neighbourhood block subtraction technique. Convolution is the weighted summation of neighbourhood, used to extract certain specific features in image (depending the convolutional kernel), our neighbourhood block subtraction enhances the corner and borders in the image while suppressing the background noise in the image.<br/><br/>
We train YOLOv3 as Modified YOLOv3 with neighbourhood block subtraction on PASCAL VOC 2007 train and val data and 2012 train data. We test our model on PASCAL VOC 2007 test data. Our model archives 73% mAP with 0.25 threshold IOU and 64.02% mAP with threshold 0.5 IOU. When compared with the original implementation of YOLOv3 trained on MS COCO our modified YOLOv3 archives 3.4% higher mAP with negligible difference in fps. Our modified YOLOv3 out performs almost all the entries of the Pascal VOC 2007 challenge. In 12 of the 20 classes our model exceeds the best entry the 2007 challenge.<br/><br/>
We believe that neighbourhood block subtraction has the potential to be used as a feature detector in classification related tasks.


![Modified YOLO Architecture](https://github.com/BesanHalwa/Modified-YOLO/blob/master/Modified%20YOLO%20Architecture.png "Modified YOLO Architecture")<br/>
<br/>
We add neighbourhood block subtraction of block size 4X4 and pre process the train and test data.

## Training Log
We start training our YOLO model with darknet53conv74 weights.<br/>
Hyper parameters settings are as follows: <br/>
batch = 64, <br/>
subdivisions = 16, <br/>
height = 416, width = 416, channels = 3, <br/>
momentum = 0.9, <br/>
decay = 0.0005, <br/>
saturation = 1.5, <br/>
exposure = 1.5, hue = 0.1, <br/>
learning rate = 0.001, <br/>
maxbatches = 50200. <br/>
We trained for 50200 iterations on Pascal VOC 2007 tarin and val, 2012 train data, this took us approximately 65 hours on Nvidia GTX 1080 Ti. <br/>
We validated our model on Pascal VOC 2007 test set. <br/>
We build the darknet with GPU = 1, CUDNN = 1, OPENCV = 0, OPENMP = 0, DEBUG = 0. <br/>

Training log can be obtained from Google Drive [Training log](https://drive.google.com/file/d/1QlnXawBu9KFbojvmwFTqiXVBvn5GOKuC/view?usp=sharing )

## Results

### Result 0
On Pascal VOC 2007 test we achieve 63.93% mAP at 0.5 IOU thresh and 74% mAP with 0.25 IOU. <br/>

Class wise mAP precision are as follows<br/>

class_id = 0, name = aeroplane, ap = 72.05%   	 (TP = 188, FP = 34) <br/>
class_id = 1, name = bicycle, ap = 75.63%   	 (TP = 240, FP = 37) <br/>
class_id = 2, name = bird, ap = 54.69%   	 (TP = 225, FP = 69) <br/>
class_id = 3, name = boat, ap = 49.49%   	 (TP = 125, FP = 76) <br/>
class_id = 4, name = bottle, ap = 35.34%   	 (TP = 151, FP = 124) <br/>
class_id = 5, name = bus, ap = 75.93%   	 (TP = 147, FP = 53) <br/>
class_id = 6, name = car, ap = 74.70%   	 (TP = 833, FP = 187) <br/>
class_id = 7, name = cat, ap = 76.18%   	 (TP = 259, FP = 86) <br/>
class_id = 8, name = chair, ap = 47.52%   	 (TP = 341, FP = 307) <br/>
class_id = 9, name = cow, ap = 68.70%   	 (TP = 167, FP = 86) <br/>
class_id = 10, name = diningtable, ap = 60.45%   	 (TP = 117, FP = 75)<br/> 
class_id = 11, name = dog, ap = 72.58%   	 (TP = 346, FP = 180) <br/>
class_id = 12, name = horse, ap = 81.63%   	 (TP = 268, FP = 97) <br/>
class_id = 13, name = motorbike, ap = 76.99%   	 (TP = 225, FP = 66) <br/>
class_id = 14, name = person, ap = 71.15%   	 (TP = 3012, FP = 739) <br/>
class_id = 15, name = pottedplant, ap = 31.79%   	 (TP = 142, FP = 91) <br/>
class_id = 16, name = sheep, ap = 56.86%   	 (TP = 146, FP = 119) <br/>
class_id = 17, name = sofa, ap = 66.59%   	 (TP = 152, FP = 84) <br/>
class_id = 18, name = train, ap = 71.41%   	 (TP = 195, FP = 58) <br/>
class_id = 19, name = tvmonitor, ap = 58.94%   	 (TP = 175, FP = 55) <br/>
<br/>
 for thresh = 0.25, precision = 0.74, recall = 0.62, F1-score = 0.67 <br/>
 for thresh = 0.25, TP = 7454, FP = 2623, FN = 4578, average IoU = 57.08 %<br/> 
<br/>
 IoU threshold = 50 %, used Area-Under-Curve for each unique Recall <br/>
 mean average precision (mAP@0.50) = 0.639302, or 63.93 % <br/>
 <br/>
loss for iterations 1 to 50200 <br/>
![loss for iterations 1 to 50200](https://github.com/BesanHalwa/Modified-YOLO/blob/master/Loss_0to50K.png "loss for iterations 1 to 50200") <br/>
<br/>
loss for iterations 150 to 50200 <br/>
![loss for iterations 150 to 50200](https://github.com/BesanHalwa/Modified-YOLO/blob/master/Loss_150to50K.png "loss for iterations 150 to 50200")<br/>
<br/>
loss for iterations 10000 to 50200 <br/>
![loss for iterations 10000 to 50200](https://github.com/BesanHalwa/Modified-YOLO/blob/master/Loss_10Kto50K.png "loss for iterations 10000 to 50200")<br/>
<br/>
loss for iterations 20000 to 50200 <br/>
![loss for iterations 20000 to 50200](https://github.com/BesanHalwa/Modified-YOLO/blob/master/Loss_20Kto50K.png "loss for iterations 20000 to 50200")<br/>

### Result 1
On Pascal VOC 2007 test we achieve 64.02% mAP at 0.5 IOU thresh and 73% mAP with 0.25 IOU. <br/>

Class wise mAP precision are as follows<br/>

class_id = 0, name = aeroplane, ap = 75.32%   	 (TP = 199, FP = 35) <br/>
class_id = 1, name = bicycle, ap = 75.19%   	 (TP = 231, FP = 42) <br/>
class_id = 2, name = bird, ap = 55.15%   	 (TP = 225, FP = 76) <br/>
class_id = 3, name = boat, ap = 50.42%   	 (TP = 138, FP = 92) <br/>
class_id = 4, name = bottle, ap = 36.54%   	 (TP = 154, FP = 88) <br/>
class_id = 5, name = bus, ap = 73.14%   	 (TP = 154, FP = 70) <br/>
class_id = 6, name = car, ap = 74.03%   	 (TP = 828, FP = 156) <br/>
class_id = 7, name = cat, ap = 77.94%   	 (TP = 262, FP = 79) <br/>
class_id = 8, name = chair, ap = 45.78%   	 (TP = 333, FP = 360) <br/>
class_id = 9, name = cow, ap = 66.06%   	 (TP = 159, FP = 109) <br/>
class_id = 10, name = diningtable, ap = 59.19%   	 (TP = 116, FP = 71) <br/>
class_id = 11, name = dog, ap = 72.63%   	 (TP = 345, FP = 171) <br/>
class_id = 12, name = horse, ap = 81.56%   	 (TP = 269, FP = 102) <br/>
class_id = 13, name = motorbike, ap = 76.66%   	 (TP = 227, FP = 77) <br/>
class_id = 14, name = person, ap = 70.90%   	 (TP = 3031, FP = 752) <br/>
class_id = 15, name = pottedplant, ap = 32.35%   	 (TP = 153, FP = 104) <br/>
class_id = 16, name = sheep, ap = 56.96%   	 (TP = 141, FP = 110) <br/>
class_id = 17, name = sofa, ap = 68.81%   	 (TP = 153, FP = 88) <br/>
class_id = 18, name = train, ap = 73.14%   	 (TP = 195, FP = 49) <br/>
class_id = 19, name = tvmonitor, ap = 58.58%   	 (TP = 164, FP = 65) <br/>
<br/>
 for thresh = 0.25, precision = 0.73, recall = 0.62, F1-score = 0.67 <br/>
 for thresh = 0.25, TP = 7477, FP = 2696, FN = 4555, average IoU = 56.73 % <br/>
<br/>
 IoU threshold = 50 %, used Area-Under-Curve for each unique Recall <br/>
 mean average precision (mAP@0.50) = 0.640169, or 64.02 % <br/>


## Weight files 

Weight file for Result 0 [Weight](https://drive.google.com/file/d/1Ere9g5KwtK4AlOGWME_vM1xPA4TmSuR0/view?usp=sharing)<br/>

Weight file for Result 1 [Weight](https://drive.google.com/file/d/1fCetTTdQlK4OZ-aKzgghU7JwSrXGtoXh/view?usp=sharing)<br/>

## Project Report
Project Report on [Modified-YOLO](https://drive.google.com/file/d/17J3_DACZvo-eB14C1e0OpI7muYoaCq8k/view?usp=sharing) 

## Result Comparison
Comparison of our modified model with entries of VOC 2007 challenge. The highlighted entries are our results. Bold and underlined entries represent the best result in class. Out of 20 classes in Pascal Voc, our model perform produced highest mAP for 12 classes.

![Result Comparison](https://github.com/BesanHalwa/Modified-YOLO/blob/master/Comparision.png "Result Comparison")

## Possible Improvements

### 1. Neighbourhood Block Subtraction: <br/>
The neighbourhood block subtraction gives us hope for potential research direction. We would try out blocks of varying sizes and compare the results, this will give us an idea about the ideal size of the block. We also would like to try out consecutive neighbourhood block subtraction layer by layer. As of now we sequentially perform the block subtraction, however this process could be more efficiently realised (in terms of time) by parallel execution with cuda. One other implementation we would like to make is include the block subtraction mechanism in the model architecture itself. We would also test this approach with other detection architectures like SSD and fast R-CNN. More experimentation will help us to develop more reasoning for the process.<br/><br/>

### 2. Spatial Transformations<br/>
### 3. Spatial Transformer Networks: ![Paper](http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf "Spatial Transformer Networks")<br/>
### 4. Inverse Compositional Spatial Transformer Networks: ![Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Inverse_Compositional_Spatial_CVPR_2017_paper.pdf)

