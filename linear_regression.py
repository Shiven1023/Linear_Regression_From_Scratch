import numpy as np
import pandas as pd
from sklearn import datasets # only used for the purpose of getting a dataset
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
class LinearRegression():
    def __init__(self,model_name,lr=0.01,n_iters=10000):
        self.model_name = model_name
        self.lr = lr
        self.n_iters = n_iters
        
    
    def fit(self,X,y):
        samples,features = X.shape #taking no. of samples and no. of features from the input numpy array
        self.weights = np.zeros(features) #initialising weights and bias as 0
        self.bias = 0

        for i in range(self.n_iters):
            f_x = np.dot(X,self.weights) + self.bias
            dw = np.dot(X.T,(y-f_x))/samples
            db = np.sum((y-f_x))/samples
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
    def predict(self,X):
        f_x = np.dot(X,self.weights) + self.bias
        return f_x


X,y = datasets.make_regression(n_samples = 100,n_features = 1,noise = 10,random_state=3)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 1200)
my_model = LinearRegression("LinearReg")
my_model.fit(X_train,y_train)
prediction = my_model.predict(X_test)
plt.scatter(X,y)
plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                                
    
        
    




            












