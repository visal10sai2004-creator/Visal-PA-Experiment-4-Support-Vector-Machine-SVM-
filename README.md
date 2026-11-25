## Experiment 4: Support Vector Machine (SVM)

## Aim

To implement and analyze the performance of a Support Vector Machine (SVM) model for classification using Python.

## Overview

A Support Vector Machine is a supervised machine-learning algorithm used for classification. It finds a decision boundary (hyperplane) that separates classes while maximizing the margin between the closest data points, called support vectors.

This experiment uses the Iris dataset and compares two SVM kernels:

 Linear kernel -- suitable when classes can be separated approximately by a straight decision boundary.
 
 RBF (Radial Basis Function) kernel -- can model non-linear decision boundaries.

For visualization, the experiment uses the first two features of the Iris dataset.

## Algorithm

1.Start.

2.Load the Iris dataset and select the first two features for visualization.

3.Split the data into training and testing sets.

4.Standardize the features using StandardScaler.

5.Train an SVM classifier with a linear kernel.

6.Predict the test labels and calculate the classification accuracy.

7.Train another SVM classifier with the RBF kernel.

8.Predict the test labels and calculate the classification accuracy.

9.Plot the decision boundaries for both models.

10.Stop.


## Program

```

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import svm

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

iris=load_iris()


X=iris.data[:,:2]

y=iris.target

feature_names=iris.feature_names[:2]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

scaler=StandardScaler()

X_train_s=scaler.fit_transform(X_train)

X_test_s=scaler.transform(X_test)

svm_linear=svm.SVC(kernel='linear',C=1.0,probability=True)

svm_linear.fit(X_train_s,y_train)

y_pred_lin=svm_linear.predict(X_test_s)

acc_lin=accuracy_score(y_test,y_pred_lin)

print("Linear SVM Accuracy:",acc_lin)

svm_rbf=svm.SVC(kernel='rbf',C=1.0,gamma='scale')

svm_rbf.fit(X_train_s,y_train)

y_pred_rbf=svm_rbf.predict(X_test_s)

acc_rbf=accuracy_score(y_test,y_pred_rbf)

print("RBF SVM Accuracy:",acc_rbf)

from google.colab import drive
drive.mount('/content/drive')

x_min, x_max = X_train_s[:, 0].min()-1, X_train_s[:, 0].max() + 1

y_min, y_max=X_train_s[:, 1].min()-1, X_train_s[:, 1].max() +1

xx, yy =np.meshgrid(np.linspace(x_min, x_max, 200),

np.linspace(y_min, y_max, 200))

Z=svm_rbf.predict(np.c_[xx.ravel(),

yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8,6))

plt.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(X_train_s[:,0], X_train_s[:,1], c=y_train,

edgecolor='k', s=40)

plt.title('SVM (RBF) decision regions (training set)')

plt.xlabel(feature_names[0])

plt.ylabel(feature_names[1])

plt.show()

```

## Output

```

Linear SVM Accuracy: 0.6888888888888889
RBF SVM Accuracy: 0.6888888888888889

Mounted at /content/drive

```

<img width="689" height="547" alt="image" src="https://github.com/user-attachments/assets/505c35e3-26f6-4599-ad74-61b9d06aa178" />


## Result

Thus, the Support Vector Machine (SVM) classification model was successfully implemented using Python on the Iris dataset. Both Linear and RBF kernels achieved an accuracy of 68.89%. The decision boundary graph illustrated the classification regions of the RBF kernel. Hence, SVM classification was successfully performed using two different kernel functions.
