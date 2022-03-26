#importing the dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

#importing the Boston House Price Dataset

house_price_dataset = sklearn.datasets.load_boston()
print(house_price_dataset)

#loading the dataset to a pandas DataFrame

house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)

#print first 5 rows of the DataFrame

print(house_price_dataframe.head())

#add the target (price) columnt to the DataFrame

house_price_dataframe['price'] = house_price_dataset.target
print(house_price_dataframe.head())

#cheking the number of rows and columns in the DataFrame

print(house_price_dataframe.shape)

#check for missing values

print(house_price_dataframe.isnull().sum())

#statistical measures of the dataset

print(house_price_dataframe.describe())

#understanding the correlation between various features in dataset
#1. positive correlation
#2. negative correlation

correlation = house_price_dataframe.corr()

#constructing a heatmap to understand the correlation

plt.figure(figsize = (10, 10))
sns.heatmap(correlation, cbar = True, square = True, fmt = '.1f', annot = True, annot_kws = {'size':8}, cmap = 'Blues')
plt.show()

#splitting the data and target

X = house_price_dataframe.drop(['price'], axis = 1)
Y = house_price_dataframe['price']
print(X)
print(Y)

#splitting the data into training data and test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
print(X.shape, X_train.shape, X_test.shape)

#model training
#XGBoost regressor
#loading the model

model = XGBRegressor()

#training the model with X_train

model.fit(X_train, Y_train)

#evaluation
#prediction on training data
#accuracy for prediction on training data

training_data_prediction = model.predict(X_train)
print(training_data_prediction)

#r squared error

score_1 = metrics.r2_score(Y_train, training_data_prediction)

#mean absolute error

score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)
print('r squared error: ', score_1)
print('mean absolute error: ', score_2)

#visualizing the actual prices and predicted prices

plt.scatter(Y_train, training_data_prediction)
plt.xlabel('actual prices')
pl.ylabel('predicted prices')
plt.title('actual price vs predicted price')
plt.show()

#prediction on test data
#accuracy for prediction on test data

test_data_prediction = model.predict(X_test)

#r squared error

score_1 = metrics.r2_score(Y_test, test_data_prediction)

#mean absolute error

score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)
print('r squared error: ', score_1)
print('mean absolute error: ', score_2)