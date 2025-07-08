# House price prediction

This is a Machine Learning project that predicts house prices based on 13 input features using the XGBoost Regressor model.  
A simple GUI built with Tkinter is provided for easy user interaction.

---

##  Features

- Trained on the Boston Housing Dataset (`boston.csv`)
- Uses XGBoost Regressor for high-accuracy predictions
- Includes a Tkinter-based GUI to predict prices easily
- Accepts 13 features as input and predicts house price in $ dollars..

---

##  Project Structure :-

House price prediction/
├── boston.csv # Dataset with 13 features + target (PRICE)
├── houseprediction.py # ML training and evaluation script
├── housepriceprediction.sav # Saved XGBoost model using joblib
├── gui.py # GUI file using Tkinter
└── README.md # This file

##  Atribute Information :- 

Input features in order:
1) CRIM: per capita crime rate by town
2) ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
3) INDUS: proportion of non-retail business acres per town
4) CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
5) NOX: nitric oxides concentration (parts per 10 million) [parts/10M]
6) RM: average number of rooms per dwelling
7) AGE: proportion of owner-occupied units built prior to 1940
8) DIS: weighted distances to five Boston employment centres
9) RAD: index of accessibility to radial highways
10) TAX: full-value property-tax rate per $10,000 [$/10k]
11) PTRATIO: pupil-teacher ratio by town
12) B: The result of the equation B=1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
13) LSTAT: % lower status of the population

##  Model Details :-

- Model: XGBoost Regressor (`XGBRegressor`)
- Train-test split: 80/20
- Evaluation Metrics :
  - R² Score  
  - Mean Absolute Error
- Visualization : Actual vs Predicted Price (scatter plot)