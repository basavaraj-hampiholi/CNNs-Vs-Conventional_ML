import cv2                 
import numpy as np         
import os                 
from random import shuffle 
from tqdm import tqdm  
from matplotlib import pyplot as plt 
import imutils   
import skimage.measure



TEST_DIR = '/home/basavaraj/Downloads/train/test'
IMG_SIZE = 128


class Preprocessing():
    def __init__(self,path):
        self.train_dir = path


    def build_filters(self):
        filters = []
        pi = np.pi/3
        for ksize in np.linspace(11,31,num=2):
            ksize= int(ksize)
            for lam in np.arange(9,11):
                for theta in np.arange(0, np.pi, np.pi / 4):
                    kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, lam, 0.5, 2.0, ktype=cv2.CV_32F)
                    kern /= 1.5*kern.sum()
                    filters.append(kern)
        return filters

    def process(self,img, filters):
        accum = []
        img = (img - img.mean()) / img.std()
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            fimage = skimage.measure.block_reduce(fimg,(9,9),np.max)
            accum.append(np.array(fimage))
        return accum

    def Gabor_Filter(self,img):
        filters = self.build_filters()
        res_img = self.process(img, filters)    
        return res_img 

    def label_img(self,img):
        word_label = img.split('(')[-2]
        # conversion to one-hot array [cat,dog]
        #                            [much cat, no dog]
        if word_label == 'airplanes ': return 0
        #                             [no cat, very doggo]
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
            img= self.Gabor_Filter(img) 
            training_data.append([np.array(img),label])
        shuffle(training_data)
        np.save('train_data.npy', training_data)
        return training_data


#train_data = create_train_data()
