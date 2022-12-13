# -*- coding: utf-8 -*-

#import libraries
import pandas as pd
from sklearn import svm
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#load dataset
Admission = pd.read_csv(r'C:/Users/86198/IS517/data/Admission.csv')
print(Admission)

#Scatter Plot
plt.scatter(Admission['GPA'], Admission['GMAT'])
plt.xlabel('GPA')
plt.ylabel('GMAT')
plt.show()

#Normalize X
min_max_scaler = MinMaxScaler()
Admission[['GPA', 'GMAT']] = min_max_scaler.fit_transform(Admission[['GPA', 'GMAT']])
print(Admission)

#Scatter Plot Normalized X
plt.scatter(Admission['GPA'], Admission['GMAT'])
plt.xlabel('GPA')
plt.ylabel('GMAT')
plt.show()

#Set Features and Target Variable
X = Admission[['GPA', 'GMAT']]
Y = Admission['Decision']

#create a classifier
clf = svm.SVC(kernel="linear", C=1.0)
clf.fit(X,Y)
Y_pred=clf.predict(X)

confusion_matrix = pd.crosstab(Y, Y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ',metrics.accuracy_score(Y, Y_pred))
plt.show()

#print (X)
print (Y_pred)

#prediction = clf.predict.([[3.78,591]]) 
#print ('Predicted Result: ', prediction)

#print(classification_report(Y, Y_pred))

#Weight values for the linear equation from the trained SVM model
w = clf.coef_[0]

#Y-offset for the linear equation
b = -w[0] / w[1]

print(w, b)

plt.figure(figsize=(10,6))
plt.title("Linear Kernel", fontsize=16)
plt.scatter(X['GPA'], X['GMAT'], c=Y, s=30, cmap='cool')

#Plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

#Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

#Plot decision boundary and margins
ax.contour(
    XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
)
#Plot support vectors
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.show()


