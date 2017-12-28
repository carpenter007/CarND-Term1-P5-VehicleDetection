# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project provides a software pipeline to detect vehicles in a video (the pipeline is tested on test_video.mp4 and implemented on project_video.mp4). Please check the 'write up' for details how this works. 

In detail, the goals / steps of this project are the following:

* Performing a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and training a classifier Linear SVM classifier
* Additionally, I also applied a colour transform and appended binned colour features, as well as histograms of colour, to the HOG feature vector. 
* All the features are normalized and the selection for training and testing data is randomized.
* A sliding-window technique is implemented. The trained classifier is used to search for vehicles in images.
* The pipeline is used on a video stream (test_video.mp4 / project_video.mp4). Using a heat map of recurring detections frame by frame helps to reject outliers and follow detected vehicles.
* A bounding box is estimated for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train the classifier. These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

Some example images for testing the pipeline on single frames are located in the `test_images` folder.  
`ouput_images` contains some images which show the output of several processing steps in the pipeline.
