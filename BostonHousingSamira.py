#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
##Read the Boston Housing dataset
df=pd.read_csv('Boston-Housing.csv')
df.head(5)

##Descriprive Analysis to know different dimensions of the dataset

df.shape
df.describe()
df.median().to_csv("Boston-Median.csv")
df.info()

#The first column has no name and it is used as an index, therefore this column has to be removed 
df=df.drop(['Unnamed: 0'],axis=1)
df.duplicated().sum()

##Data Distribution
df.hist(figsize=(12, 12), color='blue', edgecolor='black', xlabelsize=10, ylabelsize=11)


##Correlation Analysis, export it to csv, also sort the correlations and put them in the csv file 
Bostoncorr=df.corr()
Bostoncorr.to_csv('Boston-corr.csv')
c=df.corr().abs()
s=c.unstack()
so=s.sort_values(kind="quicksort")
so.to_csv('Boston-corr-sorted.csv')



##building the predictive model 
#assign medv to Y and drop it from the main dataset 
y=df['medv']
X=df.drop('medv',axis=1)
df.info()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

#split the data between training and test with 30% ratio 
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=0)
y_train.shape

#first model: LinearRegression
Bostonreg=LinearRegression()
Bostonreg.fit(x_train,y_train)
#to show the difference between the test set and the prediction 

from sklearn.metrics import mean_squared_error 
from sklearn import metrics
from sklearn.metrics import r2_score
print('Linear Regression model results:')
#Mean Square error for test set 
MSE = mean_squared_error(y_test, Bostonreg.predict(x_test)) 
print("test set Mean Square Error : ", MSE) 

#Mean absolute error for test set 
MAE= metrics.mean_absolute_error(y_test, Bostonreg.predict(x_test))
print("test set mean absolute error: %.4f"% MAE)

#R Squared score for test set 
RSQR= r2_score(y_test, Bostonreg.predict(x_test))
print("test set R Squared: %.4f"% RSQR)

#Second model: Gradient Boost model as an alternative 
from sklearn import ensemble 
modelgb=ensemble.GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        max_features=0.1,
        loss='huber',
        random_state=0)
modelgb.fit(x_train,y_train)

print('Gradintboost model results:')

#Mean Square error for test set 
MSE = mean_squared_error(y_test, modelgb.predict(x_test)) 
print("test set Mean Square Error : ", MSE) 

#Mean absolute error for test set 
MAE= metrics.mean_absolute_error(y_test, modelgb.predict(x_test))
print("test set mean absolute error: %.4f"% MAE)

#R Squared score for test set 
RSQR= r2_score(y_test, modelgb.predict(x_test))
print("test set R Squared: %.4f"% RSQR)


