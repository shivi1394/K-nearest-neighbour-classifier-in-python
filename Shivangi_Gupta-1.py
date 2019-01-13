#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:55:23 2018

@author: shivangi
"""

from sklearn.datasets import load_iris
from collections import Counter

data = load_iris()

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time as t

X_train, X_test, y_train, y_test = train_test_split(data['data'],data['target'], random_state = 0)

class KNN(object):
    
    def __init__(self):
        pass
    
    def euclideanDistance(self,test, train):
        self.m=[]
        for i in range(len(test)):
            self.m.append(np.argsort(np.sqrt(np.sum((test[i]-train)**2,axis=1))))
        return self.m
    
    def train(self, X, y):
      
        self.X_train = X_train
        self.y_train = y_train


    def Label(self, num):
        label=[]
        for i in range(len(num)):
    
            if (len(Counter(num[i]).most_common(2))>1):
                if (Counter(num[i]).most_common()[0][1]==Counter(num[i]).most_common()[1][1]):
                    label.append(-1)
                else:
                    label.append(Counter(num[i]).most_common()[0][0])
            else:
                label.append(Counter(num[i]).most_common()[0][0])
        return label
    
    def predict(self, X_test, k): 
        """
        It takes X_test as input, and return an array of integers, which are the 
        class labels of the data corresponding to each row in X_test. 
        Hence, y_project is an array of lables voted by their corresponding 
        k nearest neighbors
        """
        self.a=[]
        for i in range(k):
            for j in range(len(X_test)):
                self.a.append(self.euclideanDistance(X_test,X_train)[j][i])
        return np.transpose(np.split(np.array(y_train[self.a]),k))
        
    
    
    def report(self, testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if testSet[x] == predictions[x]:
                correct += 1
        return (correct/float(len(testSet))) * 100.0
    
    def reportArray(self,n):
        Arr=[]
        for i in range(n):
            Arr.append(self.report(y_test,self.Label(self.predict(X_test,k=i+1))))
        return Arr

K_neighbour=KNN()  
    
def k_validate(X_test, y_test):
    plt.plot(K_neighbour.reportArray(n=100))
    plt.show()
    
k_validate(X_test,y_test)    
start=t.time()
K_neighbour.predict(X_test,k=110)
end=t.time()
print(end-start,'seconds')
    
    


