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


    def Canny(self,img):
        median = cv2.medianBlur(img,1)   
        fimg = cv2.Canny(median,100,200)
        edges = skimage.measure.block_reduce(fimg,(3,3),np.max)
    #plt.subplot(121),plt.imshow(img,cmap = 'gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #plt.show()
        return edges

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
            img= self.Canny(img)
            training_data.append([np.array(img),label])
        shuffle(training_data)
        np.save('train_data.npy', training_data)
        return training_data


#train_data = create_train_data()
