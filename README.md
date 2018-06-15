# Vehicle detection and tracking
The goal of this project is to detect and track the vehicles on the road, using computer vision and machine learning techniques. 

To achieve this goal, I split the project into two parts. In the first part, I train a model to identify the vehicle images from non-vehicle images (`model.py`). In the second part, I analyze the single images or the video frames to detect any vehicles on the image using the trained model (`main.py`).

## Building the model
To build the model, I first extract features of the input images to feed into the model. The input images are located in the `data` folder (is not included here, due to the limitation for the number of files), which includes series of 8792 vehicle and 8968 non-vehicle images. After data augmentation (horizontal flipping), number of car images are 17584 and number of non-car images are 17936. 

These steps are done through `extract_features` function, which is defined in `functTrain.py`. After I build a linear SVM model, I save the model for the later use. This pipeline is coded in `model.py`.

### Feature extraction and data preprocessing

To extract the image features, I used spatial, histogram, and HOG features, with the following properties:

color_space = 'LUV'  (Can be RGB, HSV, LUV, HLS, YUV, YCrCb)

orient = 11   (HOG orientations)

pix_per_cell = 8  (HOG pixels per cell)

cell_per_block = 2  (HOG cells per block)

hog_channel = 0  (Can be 0, 1, 2, or "ALL")

spatial_size = (16, 16)  (Spatial binning dimensions)

hist_bins = 128   (Number of histogram bins)

hist_range = (0,256) 

The feature vector length created using this parameter set is 3308.

Second, I preprocess the features data by scaling them using `sklearn.preprocessing.StandardScaler()` class to have a zero mean and unit standard deviation. 

Then, I split the data into training and test sets with 80-20 ratio, and shuffled the training set.


### Building and saving the model

I use linear SVM classifier to train my model, and saved the model under `model/svm_model.pkl`. This makes the model available for use anytime in future for vehicle detection. Moreover, I save the model properties and data scaler to make them available when needed. They are saved in `model` folder under `properties.pkl` and `scaler.pkl` names, respectively.

The model accuracy, I achieved is 0.9927


### Choosing the feature detection & HOG properties

I considered different combination of the properties parameters. I compared the accuracy of the models and the time (mili second) the model needs to extract the feature of a sample input image using each set. The table bellow shows the highest accuracy achieved using the property sets.

| color_space  | spatial_size|hist_bins|orient  |hog_channel|time       |accuracy|
|:------------:|:-----------:|:-------:|:-------:|:-------:|:----------:|:-------:|
|LUV| (16, 16) 					|128		| 11  |0, 1, 2 | 0.54 |0.9927|
|LUV| (32, 32) 					|128		| 11  |0, 1, 2 | ~0.6 |0.9927|
|LUV | (16, 16) 					|128		| 9  |0, 1, 2 | 0.54 |0.9921|
|LUV |(32, 32) 					|128		| 9  |0, 1, 2 | ~0.6 |0.9921|

Apparently, I chose the set with the highest accuracy and lower feature-extraction time.


## Detecting Vehicles on test images and video

This step all is coded through `find_cars` in `main.py`, which contains the model trained above, the properties of the model and the scaler that I scaled the training and test data. 

### Sliding windows

To detect the vehicles, first I scan and define all possible windows of size (60, 60), (80, 80), (100, 100), (120, 120), (140, 140), (160, 160), (180, 180), (200, 200) with 80% overlap in the lower half of the image. These measures are chosen regarding the range of the vehicle sizes on the test images. Since (60, 60), (80, 80) windows only could contain vehicles very far in horizon, I restrict the search area for these windows even more. This is done through `slide_window` function in `main.py`

### Searching windows

In this step, I resize each window into (64, 64) window, extract their features (as described above), and scale the feature vector using the scaler obtained when building the model. Then, I predict whether this feature belong to the image of a car. If yes, this window is shortlisted for the further analyses. This step is performed by `search_windows` function in `functDetect.py`.

The shortlisted windows in one of the test images is shown bellow:

![windows](https://github.com/hanieh-hassanzadeh/Vehicle-detection-and-tracking/blob/master/outputImages/test5_all_windows.jpg)


### Removing false positive windows

To remove the false positive windows, I performed three steps. First, I define the hot spots as pixels which are included in more than some threshold number windows shortlisted. Then, I distinguish the location of the possible vehicles (windows) by `scipy.ndimage.measurements.labels` function. Second, I neglect the final window which are smaller than 1500 pixels. The third step is only applied for the vide frames. In this step, I consider the detected windows from two consequent frames to find the hot spots. This method highlights the vehicles location, and reduces the number of false positives dramatically. These steps are coded in `draw_labeled_bboxes`, `heatmapped`, and `add_heat` functions in `functDetect.py`, as well as the `find_car` class in `main.py`.

Finally, I draw boxes around the remaining windows. 

The following image demonstrates how many of false positive windows windows are removed using these steps.

![final](https://github.com/hanieh-hassanzadeh/Vehicle-detection-and-tracking/blob/master/outputImages/test5.jpg)

## Final video

[Here](https://github.com/hanieh-hassanzadeh/Advanced-Lane-Finder/blob/master/outputvideo/project_video_annotated.mp4) is a link to the final annotated video.

## Discussion

 - The most challenging issue I face here is the time needed to process the whole video. I believe, I should work on my methods to speed up the detection process. This may cause some realtime application on a real autonomous car on the road.

My potential solution: 
  1. Improve the model.
  2. predict the location of the cars using the data from previous frame, and search only a small subdomain for the vehicles.
  3. Use deep learning classifies, where helpful


- Another potential challenge could be that there are some road-vehicles in different countries which are not introduced to the model. Therefore, the model may fail facing such vehicles.

My potential solution: 
  Introduce more input images from diverse vehicles from all over the world.
