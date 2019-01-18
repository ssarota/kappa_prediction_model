import pandas as pd
import numpy as np
from skrebate import ReliefF
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours

dataset = pd.read_csv('.\data\input_datasets\machine learning (motions) - 7.8 - basis - surface.csv ')
#dataset = dataset[(dataset['kappa surface (EW)'] > 0.1) | (dataset['kappa surface (EW)'] < 0.06)]
x = dataset.drop(['EQName', 'kappa surface (EW)'], axis=1)
y = dataset.loc[:, ['kappa surface (EW)']]
x = dataset.loc[:, ['Z2.5', 'VsMin', 'VsMax', 'Vs10', 'VsZhole', 'Magnitude']]

#under = EasyEnsemble()
#x, y = under.fit_sample(x,y)

#Split dataset into 3 groups (60% training, 20% validation, and 20% testing datasets)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.25, random_state = 1)


#Normalization
normalization = StandardScaler()
x_train = normalization.fit_transform(x_train)
#x_val = normalization.transform(x_val)
x_test = normalization.transform(x_test)

#No. of columns and rows of the imported dataset
num_features = x_train.shape[1]
num_motions = x_train.shape[0]

import keras
from keras.models import Sequential
from keras.layers import Dense

predictor = Sequential()

predictor.add(Dense(output_dim = 200, init = 'uniform', activation = 'relu', input_dim = num_features, use_bias = True, bias_initializer='zeros')) #Adding the input and first hidden layer
predictor.add(Dense(output_dim = 200, init = 'uniform', activation = 'relu', input_dim = num_features, use_bias = True, bias_initializer='zeros')) #Adding the input and first hidden layer
predictor.add(Dense(output_dim = 200, init = 'uniform', activation = 'relu', input_dim = num_features, use_bias = True, bias_initializer='zeros')) #Adding the input and first hidden layer
predictor.add(Dense(output_dim = 200, init = 'uniform', activation = 'relu', input_dim = num_features, use_bias = True, bias_initializer='zeros')) #Adding the input and first hidden layer
predictor.add(Dense(output_dim = 1, init = 'uniform', activation = 'relu', use_bias = True, bias_initializer='zeros')) #Additional Output Layer

predictor.compile(optimizer = 'adam', loss = 'mse')

predictor.fit(x_train, y_train, batch_size = 100, nb_epoch = 400, validation_split = 0.2)

y_predicted = predictor.predict(x_test)
y_predicted_train = predictor.predict(x_train)

#Plot predicted y vs actual y
import matplotlib.pyplot as plt
plt.axis([0,0.175,  0,0.175])
plt.autoscale(False)
plt.plot([0 , 0.2],[0 , 0.2],'-')
plt.plot(y_predicted, y_test, 'bo')

# R^2 analysis for accuracy measurement
from sklearn.metrics import r2_score
accuracy_percent_test = r2_score(y_test, y_predicted)*100
accuracy_percent_train = r2_score(y_train , y_predicted_train)*100
