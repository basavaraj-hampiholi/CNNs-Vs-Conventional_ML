import numpy as np

class LinearDiscriminantAnalysis():
  def __init__(self,data,label,n_components):
      self.data = data
      self.label = label
      self.n_components = n_components

  def LDA(self):
    np.set_printoptions(precision=4)
    print('Input data shape:',self.data.shape)
    total_cl= len(self.label)

    mean_vectors = []
    for cl in range(0,5):
      mean_vectors.append(np.mean(self.data[self.label==cl], axis=0))
      #print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))
    #print(len(mean_vectors[0]))

    l1= self.data.shape[0]
    l2= self.data.shape[1]
    S_W = np.zeros((l2,l2))
    for cl,mv in zip(range(0,3),mean_vectors):
       class_sc_mat = np.zeros((l2,l2))                  # scatter matrix for every class
       for row in self.data[self.label == cl]:         
         row, mv = row.reshape(l2,1), mv.reshape(l2,1) # make column vectors
         class_sc_mat += (row-mv).dot((row-mv).T)
       S_W += class_sc_mat 
                                  # sum class scatter matrices
    #print('within-class Scatter Matrix:\n', S_W)
    #print('Shape of S_W: {0}'.format(S_W.shape))


    overall_mean = np.mean(self.data, axis=0)
    S_B = np.zeros((l2,l2))
    for i,mean_vec in enumerate(mean_vectors):  
       n = self.data[self.label==i+1,:].shape[0]
       mean_vec = mean_vec.reshape(l2,1) # make column vector
       overall_mean = overall_mean.reshape(l2,1) # make column vector
       S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    #print('between-class Scatter Matrix:\n', S_B)
    #print('Shape of S_B: {0}'.format(S_B.shape))


    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
   
    
    
    #Sorting the Eigen Values
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    


    W = eig_pairs[0][1].reshape(l2,1)   
    for i in range(1, self.n_components):
       A = eig_pairs[i][1].reshape(l2,1)
       W = np.c_[W,A]

    #print('W shape:',W.shape)
          
    #print('Matrix W:\n', W.real)
    
    X_lda = self.data.dot(W.real)
    return X_lda
    


