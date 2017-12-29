**Vehicle Detection Project**

In detail, the goals / steps of this project are the following:

* Performing a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and training a classifier Linear SVM classifier
* Additionally, I also applied a colour transform and appended binned colour features, as well as histograms of colour, to the HOG feature vector. 
* All the features are normalized and the selection for training and testing data is randomized.
* A sliding-window technique is implemented. The trained classifier is used to search for vehicles in images.
* The pipeline is used on a video stream (test_video.mp4 / project_video.mp4). Using a heat map of recurring detections frame by frame helps to reject outliers and follow detected vehicles.
* A bounding box is estimated for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2_1]: ./output_images/HOG_example_non_vehicle.jpg
[image2_2]: ./output_images/HOG_example_vehicle.jpg
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./output_video/project_video.mp4


---

### Feature extraction

An essential step to detect vehicles in an image is the feature extraction. Besides extracting features from colors (histogram of the occurence of colors and computing binned color features ) the Histogram of Oriented Gradients (HOG) is a robust feature which can be used to classify vehicles or non-vehicles.

The code to extract HOG features is contained in the third code cell of the IPython notebook `VehicleDetection.ipynb`.  

There I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2_1]

![alt text][image2_2]

I tried various combinations of parameters for the HOG feature extraction. 
The YUV and YCrCb colorspace performed best. This is probably because at these colorspaces the illumination is extracted.
The orientation seems to be good if it is greater then 8. 11 diffenent orientations fitted best to my application.
I used 16x16 pixels per cell and 2x2 cells per block. I've determined with a small subset of images to get the best fitting values.

For the color features I create binned color features with a spartial size of (16x16). For the histogram of colors I use 64 bins for each image channel. All the color features are concatenated to one feature vector.

The parameters are choosen manually. It was always a trade-off between performance, accuray and overfitting. Once I found parameters which allow a good prediction on several test images, I stopped tuning for a higher accuray and started to focus on processing performance.

Ultimately I got a feature vector with a length of 2148.


### Create a classifier

I started by reading in all the `vehicle` and `non-vehicle` images. These abeled images were taken from the GTI vehicle image database and the KITTI vision benchmark suite. All images are 64x64 pixels.The database is very important for this project. The given classified database consists of about 9000 non-vehicle and 9000 vehicle images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]



All the selected HOG and color features are used to train a classifier.

At the beginning I started with a non-linear SVM and optimized the hyper-parameters using the Scikit Learn GridSearch method.
The test results were quite good with an accuracy above 99%. But later on I had to realize that the results aren't that good on the test images nor at the test video. I faced a kind of overfitting.

From this point I focused more on the dataset. I switched from the non-linear to a linear SVM which can be trained very fast. The features are staight forward enough to seperate them linear to classify a vehicle or a non-vehicle.

To prevent overfitting and to reduce false detections I increased the non-vehicle database by 50% by copying some training images. Further I had to  avoids having nearly identical images in both training and test sets. Therefore, before shuffling the data, I splitted the GTI vehicle image database (which contrains a lot of time-series images) in a test and train set. 

At the end, the classifier has an accuracy of 98% but works even better on the project images / videos then the non-linear SVM.

The classifier and the used hyperparameters are stored in the file `svc_pickle.p` and can be read out using pickle:

```python
dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
```


### Sliding Window Search

At this point we have a classifier which is able to predict a vehicle or a non-vehicle from a 64x64 pixel image. To get a correct prediction based on a whole camera image, a method must be found to check all the frames on which a car could be occur.
- Depending on where a car is placed on the image, it appears smaller or bigger. I had to choose different patch sizes and resize them to a 64x64 pixel image.
- Making sure that the performance for scanning on camera image is not too bad, we have to consider that small cars will appear in the horizont region while bigger cars will appear near our camera (at the bottom of the image). This information helps to define the seaching areas for different patches as small as possible. This step does also reduce false detections because we avoid seaching for cars in areas on which no car should occur.
- The sliding window shall overlap to avoid missing vehicles which occur between two patches

I've got the best results, seaching with three different patch sizes: 
- 64x64 pixel patch size (1.0 * 64x64) pixels
- scaled patch of (0.7 * 64x64) pixels
- scaled patch of (2.0 * 64x64) pixels

The patches are overlapping 4 times in each direction when sliding through the window. This number is quite high, but it gave the opportunity to eliminate false detections and the processing performance didn't drop much.


![alt text][image3]

### Pipeline and pipeline optimization

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color (16x16) and histograms of color (64) in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

Using the classifier and the sliding window search to predict vehicles the results will be filtered before printing a car label on the given camera image.
Because the sliding window search uses different patches which overlap, one vehicle will be detected more than one time in a camera image. This gives the opportunity to filter out false or non-confidable predictions. 

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. 
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  
I then assumed each blob corresponded to a vehicle.  
I constructed bounding boxes to cover the area of each blob detected.  

On the heatmap I implemented a threshold of 4 on each camera image when predicting a vehicle.

Here's an example result showing the heatmap from a test image of the project video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the image:

Found boxes and their heatmaps:
![alt text][image5]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:
![alt text][image6]

Here the resulting bounding boxes are drawn onto the image:
![alt text][image7]



---

### Video Implementation

Here's a [link to my video result](./project_video.mp4)

To make the prediction more robust and to get a smoother label for predicted vehicles, a camera image frame overlapping filter is used. Using 8 Frames, and a further threshold of 28 (sum of overlapping vehicle predictions over 8 frames) makes sure that false predictions are prevented. Additionally it helps to avoid flicking labels, even if there are one or two single frames on which a car isn't recognized at all.

The pipeline only stores the boxes over the given number of frames, but it doesn't store the whole heatmap for each frame. This allows a flexible use of different filter methods and reduces allocated dataspace.


---

### Discussion

While the pipeline is very robust against false predictions, it is able to detect cars which are not to far and moves not to fast. It does not recognize cars from the opposite direction which drive very fast.

As an optical improvement, the resulting labels could be mean-filtered. Using the information of previously found cars and car sizes could help to make the labels smoother.

The pipeline processes ~1 frame / second which was my personal goal. Anyhow this is far away from real-time behaviour. Reducing the number of sliding windows would be a fast performance improvement.

Using a non-linear SVM (e.g. RBF kernel) can improve the accuracy, but in my case it even increase the processing time three times.

Using deeplearning networks could give new opportunities for accuracy and performance.

