import joblib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


house_pricing = pd.read_csv(r"C:\Users\acer\OneDrive\Desktop\ML-PROJECTS\House-price-prediction\boston.csv")
print(house_pricing.head())
print(house_pricing.shape)

# checking for the missing values ...
print(house_pricing.isnull().sum())

print(house_pricing.describe())


x = house_pricing.drop(['PRICE'], axis=1)
y = house_pricing['PRICE']

print(x)
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#moedel training..

model = XGBRegressor()
model.fit(x_train,y_train)

# model evaluation..
training_data_prediction = model.predict(x_train)

# R square eror
score1= metrics.r2_score(y_train, training_data_prediction)

# mean absolute error
score2 = metrics.mean_absolute_error(y_train, training_data_prediction)

print("training data prediction scores : ",score1,"\n",score2)

testing_data_prediction = model.predict(x_test)

# R square eror
score3= metrics.r2_score(y_test, testing_data_prediction)

# mean absolute error
score4 = metrics.mean_absolute_error(y_test, testing_data_prediction)

print("testing data prediction scores : ",score3,"\n",score4)

# visualizing the acccuracy..
plt.scatter(y_train, training_data_prediction)
plt.xlabel("actual prices")
plt.ylabel("predicted price")
plt.title("actual vs predicted prices")
plt.show()


input=(0.08829,12.5,7.87,0,0.524,6.012,66.6,5.5605,5,311,15.2,395.6,12.43)
input_as_numpy_array = np.asarray(input)
input_reshape = input_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_reshape)
print("house price will be :- ",prediction)

joblib.dump(model, 'housepriceprediction.sav')