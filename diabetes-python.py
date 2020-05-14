#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

tf.__version__

##Read the Diabetes dataset
df = pd.read_csv('diabetes.csv')
print(df.head(5))

##Descriprive Analysis to know different dimensions of the dataset using shape and decribe
df.shape
df.describe().to_csv('diabetes-describe.csv')
df.median().to_csv("Diabetes-Median.csv")
df.info()

##Data Distribution using histogram
df.hist(figsize=(12, 12), color='blue', edgecolor='black', xlabelsize=10, ylabelsize=11)
df.plot(kind='box', figsize=(20,10))
plt.show()

#check the duplicates in the dataset 
diabetesdup= df.duplicated().sum()
print("Total number of Duplicates is " + str(diabetesdup))

#replacing 0 vlues with NaN (blank) and replace the NaN with the mean of that column 
cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df[cols] = df[cols].replace(['0', 0], np.nan)

print(df.isnull().any())
column_means = df.mean()
df = df.fillna(column_means)
df.to_csv('Engineered dataset.csv')
df.describe().to_csv('diabetes-describe-afterNan.csv')
df[['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = StandardScaler().fit_transform(df[['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']])
df.to_csv('diabetes-describe-afterstandard.csv')

#correlation analysis by Pearson method 

corr_matrix=df.corr()
print(corr_matrix)

#defining the input and output of the model 
y = df.Outcome # define the target variable (dependent variable) as y
columns= ['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X = pd.DataFrame(df, columns=columns) # load the dataset as a pandas data frame



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


##first model: LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
#to show the difference between the test set and the prediction 

from sklearn.metrics import mean_squared_error 
from sklearn import metrics
from sklearn.metrics import r2_score

print('Linear Regression model results:')
#Mean Square error for test set which indicates squared error loss and it is a rsk function, the closer to 0 is the better  
MSE = mean_squared_error(y_test, reg.predict(X_test)) 
print("test set Mean Square Error : ", MSE) 

#Mean absolute error for test set 
MAE= metrics.mean_absolute_error(y_test, reg.predict(X_test))
print("test set mean absolute error: %.4f"% MAE)

#R Squared score for test set 
RSQR= r2_score(y_test, reg.predict(X_test))
print("test set R Squared: %.4f"% RSQR)

#shows the accuracy of the model 
print("Accuracy Score:", reg.score(X_test, y_test))

#cross validation with kfold method 
kfold = model_selection.KFold(n_splits=5, random_state=100)
model_kfold = LinearRegression()
results_kfold = model_selection.cross_val_score(model_kfold, X, y, cv=kfold)
print("Accuracy of Kfold cross validayion: %.2f%%" % (results_kfold.mean()*100.0))


##Second model: Artificial Neural Network

y=df['Outcome']
X=df.drop('Outcome',axis=1)
# Part 1 - Splitting the dataset into the Training set and Test set
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Part 2 - Building the ANN

# Initializing the ANN
from keras.models import Sequential
from keras.layers import Dense

ann=Sequential()
ann.add(Dense(32, input_dim=8, activation='relu'))
ann.add(Dense(32, activation='relu'))
ann.add(Dense(1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 4 - Making the predictions and evaluating the model

print('predicting the result')
# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

print('making the confusion matrix')
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)




