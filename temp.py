#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Churn_Modelling.csv')

#dependent and independent
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# applying encoding for geography and gender
geography = pd.get_dummies(X['Geography'],drop_first=True)
gender = pd.get_dummies(X['Gender'],drop_first=True)

# concatination
X=pd.concat([X,geography,gender],axis=1)
X=X.drop(['Geography','Gender'],axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
