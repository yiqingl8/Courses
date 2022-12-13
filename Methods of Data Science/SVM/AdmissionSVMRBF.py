# -*- coding: utf-8 -*-

#import libraries
import pandas as pd
from sklearn import svm
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

#load dataset
Admission = pd.read_csv(r'C:/Users/86198/IS517/data/Admission.csv')
#print(Admission)

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

#Non-linear algorithm for model
nonlinear_clf = svm.SVC(kernel='rbf', gamma =0.7, C=1.0)
nonlinear_clf.fit(X,Y)
Y_pred=nonlinear_clf.predict(X)

confusion_matrix = pd.crosstab(Y, Y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ',metrics.accuracy_score(Y, Y_pred))
plt.show()

#print (X)
print (Y_pred)

#prediction = clf.predict.([[3.78,591]]) 
#print ('Predicted Result: ', prediction)

print(classification_report(Y, Y_pred))