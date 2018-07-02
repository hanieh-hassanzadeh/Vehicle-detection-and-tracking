import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from sklearn.utils import shuffle
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import pickle
from  functTrain import *
import csv

imagesCar   = glob.glob('../data/vehicles/*/*')
imagesNonCar    = glob.glob('../data/non-vehicles/*/*')
hardAugNonCar   = glob.glob('../data/nonV_from_video/*')

imagesNonCar = shuffle(imagesNonCar)[:len(imagesCar) * 2- len(hardAugNonCar)]
imagesNonCar = imagesNonCar + hardAugNonCar

print('# of car images: ', len(imagesCar))
print('# of non car images: ', len(imagesNonCar))
##############SVM
# Define parameters for feature extraction
properties = {}
properties['color_space'] = color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
properties['orient'] = orient = 11  # HOG orientations
properties['pix_per_cell'] = pix_per_cell = 16 # HOG pixels per cell
properties['cell_per_block'] = cell_per_block = 2 # HOG cells per block
properties['hog_channel'] = hog_channel = 0 # Can be 0, 1, 2, or "ALL"
properties['spatial_size'] = spatial_size = (16, 16) # Spatial binning dimensions
properties['hist_bins'] = hist_bins = 128    # Number of histogram bins
properties['spatial_feat'] = spatial_feat = True # Spatial features on or off
properties['hist_feat'] = hist_feat = True # Histogram features on or off
properties['hog_feat'] = hog_feat = True # HOG features on or off
properties['hist_range'] = hist_range = (0,256)

car_features = extract_features(imagesCar, color_space=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)

notcar_features = extract_features(imagesNonCar, color_space=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)

print ('Car samples: ', len(car_features))
print ('Notcar samples: ', len(notcar_features))

X = np.vstack((car_features, notcar_features)).astype(np.float64)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features)))) # Define the labels vector

X, y = shuffle(X, y)
# Split up data into randomized training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=22)
 
X_train_scaler = StandardScaler().fit(X_train) # Fit a per-column scaler
scaled_X_train = X_train_scaler.transform(X_train) # Apply the scaler to X train
scaled_X_test  = X_train_scaler.transform(X_test) # Apply the scaler to X test

print('Feature vector length:', len(X_train[0]))
featureVecLength = len(X_train[0])

#Shuffle the training set to avoid any bias due to dara order
scaled_X_train, y_train = shuffle(scaled_X_train, y_train)
##############################################
#Building the model
model = LinearSVC() # Use a linear SVC
model.fit(scaled_X_train, y_train) # Train the classifier

accuracy = round(model.score(scaled_X_test, y_test), 4)
print('Test Accuracy of SVC = ', accuracy) # Check the score of the SVC

# save the model to disk
pickle.dump(model, open('model/svm_model.pkl', 'wb'))

# save the properties
pickle.dump(properties, open('model/properties.pkl', 'wb'))

#save the StandardScaler() function
pickle.dump(X_train_scaler, open('model/scaler.pkl', 'wb'))
