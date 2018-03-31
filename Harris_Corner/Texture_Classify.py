import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.svm import SVC

from matplotlib import pyplot as plt
import time as tm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import StratifiedKFold
from evolutionary_search import maximize

from LinearDiscriminantAnalysis import LinearDiscriminantAnalysis
from Harris_Corner import Preprocessing

from scipy import linalg
import cv2                 
import imutils   
import skimage.measure


def main():

  start= tm.time()
  #********************************************************************
  # There are two options to load the data. If you are running
  # preprocessing for the first time or with some parameter 
  # changes type 'True'.Provide the path of the train data folder   
  # like /home/hampiholi/SSProject/train_1/.This starts proprocessing
  # from scratch and takes some time. Or you can type 'False' 
  # to load the data directly, which saves time from preprocessing again
  #*********************************************************************

  create_data = bool(input("Do you want to preprocess the data? Please type 'True' or 'False':"))

  if create_data:
    TRAIN_DIR = raw_input("Enter the path of the directory: ")
    preprocessed_data = Preprocessing(TRAIN_DIR)
    data = preprocessed_data.create_train_data()
  else: 
    data = np.load('train_data.npy',encoding = 'latin1')
  
  #*************************************************************
  # Loading the data  
  #*************************************************************
  x = np.array([i[0] for i in data])
  X = x.reshape((x.shape[0],-1))
  X = X + 0.00001*np.random.rand(X.shape[0],X.shape[1]) 
  print('Shape of preprocessed data:', X.shape)

  y = np.array([i[1] for i in data])
  print('Size of lables',len(y))
  
  #*************************************************************
  # Choose the number of components for LDA. The features are
  # sorted in the order of eigen_values             
  #*************************************************************
  n_components = int(input("Enter desired number of components for LDA: "))

  print('Number of features before reduction:',X.shape[1])
  lda = LinearDiscriminantAnalysis(X,y,n_components)
  X = lda.LDA()              
  print('Number of features after reduction:',X.shape[1])

  
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

  #*************************************************************
  # The classifiers are used with default settings from sklearn.
  # You can specify your own parameters in SVC() or in kNN()              
  #*************************************************************
  
  x = int(input("Choose the classifier-Press 1 for SVM and 2 for kNN: "))
  if x==1:
    clf = SVC()
    clf = clf.fit(X_train, y_train)

  elif x==2:
    clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    clf = clf.fit(X_train, y_train)

  accuracy = clf.score(X_test, y_test)
  print('Accuracy:',accuracy)
  end= tm.time()
  train_duration= end - start
  print('Training Duration',train_duration/60)



if __name__ == '__main__':
  main()