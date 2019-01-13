# K-nearest-neighbour-classifier-in-python
The code is a part of my Homework assignment of Data mining class.

Homework question :
In this homework, you will implement a K-nearest neighbor classifier, which probably is the simplest kind of 
classification model. The KNN model is a simple non-parameterized classification model. So in the training part, 
the model simply remembers all the training data and their corresponding class labels. In the predict part, 
the model simply computes the distance of the input data to the training data and choose k nearest neighbors. 
The simple majority vote from the k closest neighbors is the class label for the input data.

Issue: If there is no simple majority vote, (i.e., two or more class labels receive the same maximum votes, 
then we do not classify the input)

Hyperparameter(s): k is a hyperparameter. Since k is a hyper parameter, we need to use the X_test to find the best k. 
Before you do that, you need to plot the accuracy against k.

I prepare the following skeleton program for you and you are asked to:

complete the skeleton
time your predict() function.
Plot the accuracy against k
Pick the best k for the highest accuracy. Let’s denote it as x%.
The answer the following question: can we report the x% accuracy as the accuracy to users of the program. Why or why not?
