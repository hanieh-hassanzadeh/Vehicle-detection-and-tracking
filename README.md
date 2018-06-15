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

The model accuracy, I achived is 0.9927


### Choosing the feature detection & HOG properties

I considered different combination of the properties parameters. I compared the accuracy of the models and the time (mili second) the model needs to extract the feature of a sample input image using each set. The table bellow shows the highest accuracy acieved using the property sets.

| color_space | spatial_size|hist_bins|orient|hog_channel|time|accuracy|
|:------------:|:-----------:|:-------:|:-------:|:-------:|:------------------------------:|:---------------------:|
|LUV| (16, 16) 					|128		| 11  |0, 1, 2 | 0.54 |0.9927|
|LUV| (32, 32) 					|128		| 11  |0, 1, 2 | ~0.6 |0.9927|
|LUV | (16, 16) 					|128		| 9  |0, 1, 2 | 0.54 |0.9921|
|LUV |(32, 32) 					|128		| 9  |0, 1, 2 | ~0.6 |0.9921|

Apparantly, I chose the set with the highest accuracy and lower feature-extraction time.


## Detecting Vehicles on test images and video




