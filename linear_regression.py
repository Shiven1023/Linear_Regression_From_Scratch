import numpy as np
import pandas as pd

class LogisticRegression():
    def __init__(self,model_name,lr ,n_iters=1000):
        self.model_name = model_name
        self.lr = lr
        self.n_iters = n_iters
        
    
    def fit(self,X,Y):
        samples,features = X.shape #taking no. of samples and no. of features from the input numpy array
        self.weights = np.zeroes(features) #initialising weights and bias as 0
        self.bias = 0

        for i in range(self.n_iters):
            y_hat = np.dot(X,self.weights) + self.bias
            dw = np.dot(X,(y-y_hat))*/samples

   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        '''for i in range(samples): # loop through each data sample and evaluate square error for it
            f_x = np.dot(self.weights,X[i]) + self.bias
            sq_error += (y[i]-f_x)**2
        avg_sq_error = (sq_error)/(2*samples)'''  
                                
    
        
    




            












