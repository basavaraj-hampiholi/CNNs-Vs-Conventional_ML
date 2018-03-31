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


    def Harris(self,img):
        corners = cv2.goodFeaturesToTrack(img, 100, 0.01, 10)
        corners = np.int0(corners)
        for corner in corners:
           x,y = corner.ravel()
           cv2.circle(img,(x,y),3,255,-1)        
        fimage = skimage.measure.block_reduce(img,(3,3),np.max)
        return fimage


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
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))         
            img= self.Harris(img)
            training_data.append([np.array(img),label])
        shuffle(training_data)
        np.save('train_data.npy', training_data)
        return training_data


#train_data = create_train_data()
