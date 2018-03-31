import cv2                 
import numpy as np         
import os                 
from random import shuffle 
from tqdm import tqdm  
from matplotlib import pyplot as plt 
import imutils   
import skimage.measure



IMG_SIZE = 128


class Preprocessing():
    def __init__(self,path):
        self.train_dir = path


    def extract_color_histogram(self,image, bins=(8, 8, 8)):
        # extract a 3D color histogram from the HSV color space using
        # the supplied number of `bins` per channel
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
            [0, 180, 0, 256, 0, 256])
     
        # normalizing the histogram if we are using OpenCV 2.4.X
        if imutils.is_cv2():
            hist = cv2.normalize(hist)
     
        # otherwise, perform "in place" normalization in OpenCV 3 
       
        else:
            cv2.normalize(hist, hist)
     
        # return the flattened histogram as the feature vector
        return hist.flatten()

    def label_img(self,img):
        word_label = img.split('(')[-2]
        
        if word_label == 'airplanes ': return 0
        
        elif word_label == 'Faces_easy ': return 1

        elif word_label == 'Motorbikes ': return 2

        elif word_label == 'Flower ': return 3
        
        elif word_label == 'Guitar ': return 4


    def create_train_data(self):
        training_data = []
        for img in tqdm(os.listdir(self.train_dir)):
            label = self.label_img(img)
            path = os.path.join(self.train_dir,img)
            img = cv2.imread(path)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))         
            img= self.extract_color_histogram(img)
            training_data.append([np.array(img),label])
        shuffle(training_data)
        np.save('train_data.npy', training_data)
        return training_data



