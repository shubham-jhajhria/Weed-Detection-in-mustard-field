# Weed-Detection-in-mustard-field

# deployed project will be uploaded soon


# Introduction
<p align="justify"> The aim is to find an efficient and robust deep learning model. The model is divided into two parts: </p>
    
    a. Classification of crop and weeds.
    b. Realtime detection of weeds in crop field.


# ABSTRACT
<p align="justify"> Weeds are one of the most important factors affecting agricultural production. The waste and pollution of farmland ecological environment caused by full-coverage chemical herbicide spraying are becoming increasingly evident. With the continuous improvement in the agricultural production level, accurately distinguishing crops from weeds and achieving precise spraying only for weeds are important. However, precise spraying depends on accurately identifying and locating weeds and crops. The current project emphasizes on deep learning-based methods to solve weed detection problems. Extracting and selecting discriminating features with ML methods is difficult because crops and weeds can be similar. This problem can be addressed efficiently by using DL approaches based on their strong feature learning capabilities.

    
Keywords: Deep learning (DL), weeds, herbicide, Machine learning (ML). </p>


# Data Preparation and Image Pre-processing
<p align="justify"> After acquiring data from different sources, it is necessary to prepare data for training, testing, and to validate models. Raw data is not always suitable for the DL model. The dataset preparation approaches include applying different image processing techniques, data labelling, using image augmentation techniques to increase the number of input data, or impose variations in the data and generating synthetic data for training. Commonly used image processing techniques are - background removal, resizing the collected image, green component segmentation, removing motion blur, de-noising, image enhancement, extraction of colour vegetation indices, and changing the colour model.
    
Some level of image processing before providing the data as an input to the DL model. It helps the DL architecture to extract features more accurately. Here we discuss image pre-processing operations used in the related studies.
Image Resizing: the three different resolutions 224*224, 255*255 and 277*277 are used for classification. The images having 255*255 as patch size achieved good accuracy and required less time to train the model. 640*640 is used in YOLO for detection purpose.
    
Image augmentation: Image augmentation is a technique of altering the existing data to create some more data for the model training process. 
    
Data labelling: Data labeling is a process that involves adding target attributes to data and labeling them to train the machine and deep learning models. Data labeling is crucial in object detection to perform various computer vision tasks such as object segmentation, object counting, and object tracking.
For current model LabelImg is used for labelling the images. </p>

# Objective of project
- To learn different supervised deep learning architectures.
- To compare efficiency for a particular scenario by using different architectures.

# Machine Learning algorithm used in the recommendation system is:
## Self-Build model: 
<p align="justify"> The summary of self-implemented model is given below. Batch size of 2, epochs = 50, Adam is used as optimizer, sparse categorical cross entropy as loss is used. ReLU is used as optimizer for all layers except last one. In last one SoftMax is used as optimizer. A testing accuracy of 94.71 % is achieved. </p>
![image](https://github.com/shubham-jhajhria/Weed-Detection-in-mustard-field/assets/108121919/5af3af87-1079-4c9a-8b87-8f93914e9c00)

## YOLO v8:
<p align="justify"> For Realtime detection pretrained model of YOLO v8 is used. And retrained on the current dataset.
Model summary: 225 layers, 3011238 parameters, 3011222 gradients, 8.2 GFLOPs
    
Above is the summary of model. An image size of 640*640, 20 epochs and Adam is used as optimizer.</p>

# The classification report for self build model:
![image](https://github.com/shubham-jhajhria/Weed-Detection-in-mustard-field/assets/108121919/b114a310-4da8-4fa9-9974-06063db8b5fb)

# Results(losses) for YOLO v8 model.:
![image](https://github.com/shubham-jhajhria/Weed-Detection-in-mustard-field/assets/108121919/619e1d5c-3df6-4db2-b04b-816ddeb920d2)
# Confusion matrix for YOLO V8:
![image](https://github.com/shubham-jhajhria/Weed-Detection-in-mustard-field/assets/108121919/68e3f2f6-3c74-4979-8683-d47199c4c0af)
<p align="justify">  Here, the model is detecting crop as crop every-time but 25% times it it detecting background as crop also. Weed is detected as weed and its accuracy is 82% but 18% times it is detecting weed as background. 75% times it is detecting background as weed. Optimizations are needed for background detection.  </p>

# Conclusion
<p align="justify"> Early weed detection is crucial in agricultural productivity, as weeds act as a pest to crops. Then features were extracted from the images to distinguish properties of weeds and the crop. Three different classifiers were tested using those properties: SELF-Build, AlexNet and VGG. The experimental results demonstrate that SELF-Build performed better than the other classifiers in terms of accuracy and other performance metrics. Self build offered 94.71% accuracy in weed detection from RGB images, whereas AlexNet and VGG offered only 58.4% and 58.53% accuracy, respectively. 
Other potential future research directions include the need for large generalized datasets. In future, will explore more images, and will apply more deep learning algorithms to increase the accuracy of weed detection and try to embed it in drone for surveying the field. It could be useful for other agricultural applications, including detection of plant diseases, classification of agricultural land cover, recognition of fruits, etc. </p>

