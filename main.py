import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import numpy as np
import pickle
import cv2
from functDetect import *
import os
import collections

class find_cars():
    def __init__(self, clf, scaler, properties):
        self.clf            = clf
        self.scaler         = scaler
        self.color_space    = properties['color_space']  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient         = properties['orient']  # HOG orientations
        self.pix_per_cell   = properties['pix_per_cell']  # HOG pixels per cell
        self.cell_per_block = properties['cell_per_block']  # HOG cells per block
        self.spatial_size   = properties['spatial_size']   # Spatial binning dimensions
        self.hist_bins      = properties['hist_bins']     # Number of histogram bins
        self.hog_channel    = properties['hog_channel'] # Can be 0, 1, 2, or "ALL"
        self.spatial_feat   = properties['spatial_feat']  # Spatial features on or off
        self.hist_feat      = properties['hist_feat']  # Histogram features on or off
        self.hog_feat       = properties['hog_feat'] # HOG features on or off
        self.hist_range     = properties['hist_range'] 
        self.lastNFramesWins= collections.deque()

    def __call__(self, img):
        windows = []

        for n in [60, 80, 100, 120, 140, 160, 180, 200]:
            window_size = (n, n)
            ystart = int(img.shape[0]/2 + 40) 
            ystop  = int(img.shape[0]-50)
            xstart = int(img.shape[1]/2+50)
            xstop  = int(img.shape[1])
            overlap = 0.8 
            # if the size of the window is smaller than (100,100), cut the search area
            if n<100:
                ystop = int(ystop - 200)
                xstop  = int(img.shape[1] - 250)
            if n>120:
                overlap = 0.9 
            # scan the area for any possible window
            windows.append(slide_window(img, x_start_stop=[xstart, xstop], y_start_stop=[ystart, ystop],
                    xy_window=window_size, xy_overlap=(overlap, overlap)))
            # search whether there is any vehicle in the window
        currentFrameWindows = search_windows(img, windows, self.clf, self.scaler,
                self.color_space, self.spatial_size,
                self.hist_bins, self.hist_range, self.orient, self.pix_per_cell, self.cell_per_block,
                self.hog_channel, self.spatial_feat, self.hist_feat, self.hog_feat)
        threshold = 27
        if len(self.lastNFramesWins)<22:
            threshold = 3#20
        else:
            del self.lastNFramesWins[0]
        self.lastNFramesWins.append(currentFrameWindows)
        #img1 = np.copy(img)
        #img1 = draw_boxes(img1, self.lastNFramesWins, color=(0, 0, 255), thick=6)
        #plt.imshow(img1)
        #plt.savefig('test5_annot.jpg')
        # heatmapping to remove the false positives
        heatmap = np.zeros_like(img)
        heatmap, labels = heatmapped(heatmap, self.lastNFramesWins, threshold=threshold)
        #draw boxes arund the hot areas
        boxed_image =draw_labeled_bboxes(img, labels)

        return boxed_image


#******************************************************************
# load a pe-trained svc model from a serialized (pickle) file
clf = pickle.load( open("model/svm_model.pkl", "rb" ) )
# get the scaler of the training data
scaler = pickle.load( open("model/scaler.pkl", "rb" ) )
# get properties of the model
properties = pickle.load( open("model/properties.pkl", "rb") )

car_detected = find_cars(clf, scaler, properties)
process_images = False
process_video  = True
#########test images##############
if process_images:
    #Load the test images
    testImgFiles = glob.glob('./test_images/test5.jpg')
    #If the output directory doesn't exist, create one
    try:
        os.stat("outputImages")
    except:
        os.mkdir("outputImages")

    for file in testImgFiles:
        img = mpimg.imread(file)
        annotated = car_detected(img)
        plt.imshow(annotated)
        plt.savefig('./outputImages/%s'%file[-9:])
        print ('./outputImages/%s'%file[-9:])
#########test video##############
if process_video:
    #If the output directory doesn't exist, create one
    try:
        os.stat("outputVideo")
    except:
        os.mkdir("outputvideo")

    outputVideo = './outputVideo/project_video_annotated.mp4'
    clip1 = VideoFileClip("./project_video.mp4")
    clipImgs = clip1.fl_image(car_detected) #This function expects color images!!
    clipImgs.write_videofile(outputVideo, audio=False)
    #clip = clipImgs.subclip(0, 20)
    #clip.write_videofile(outputVideo, audio=False)    
