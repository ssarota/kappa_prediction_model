import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

import matplotlib.pyplot as plt

#Initial setups to distinguish the results from this code.
file_name = 'Strain01_Vs30,Z2.5,Rrup,Depth_7Cases_'
folder_path = '.\case_'

#The purpose of this script is to train NN using certain features.
data_file_name = '\dataset_kappa surface_1109_(Additional Vss - Vs at 100).csv'
data_folder_path = '.\data\input_datasets'
original_dataset = pd.read_csv(data_folder_path + data_file_name)
dataset = original_dataset.loc[original_dataset['Strain Category']==1]
dataset = original_dataset.loc[:,
                               ['Vs30', 'Rrup1', 'Z2.5', 'Depth2', \
                                'Max Vs30', 'Max Vs50', 'Max Vs100', 'VsMin', \
                                'kappa surface']
                               ]

#separate input dataset (x) that includes all motions
x_all = dataset.drop(['kappa surface'], axis=1)

#Clean and separate dataset
x = dataset.drop(['kappa surface'], axis=1)
y = dataset.loc[:, ['kappa surface']]
                      
#Split dataset into 2 groups (training and testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = 1)

#Normalization
normalization = StandardScaler()
x_train = normalization.fit_transform(x_train)
x_test = normalization.transform(x_test)

#Choose conly relevant columns
x_train = x_train[:,[0, 1, 2, 3, 4]]
x_test = x_test[:,[0, 1, 2, 3, 4]]

#Column names for Gradual Feature Removal
column_names = list(dataset.columns)

#No. of columns and rows of the imported dataset
num_features = x_train.shape[1]
num_motions = x_train.shape[0]

######################### Parameter Tuning ##################################
step = 1
find_max_accuracy = []
find_max_accuracy_train = []
num_neurons_in_layer_1 = []
num_neurons_in_layer_2 = []
multiple_results_from_same_structure = []
multiple_train_results_from_same_structure = []
iter_num_list = []
for iter_num in range(100,101,10): 
    for k in range(65,80,1):
        for j in range(24,40,1):
            multiple_results_from_same_structure = []
            multiple_train_results_from_same_structure = []
            for l in range(10):
                predictor = Sequential()    
                predictor.add(Dense(output_dim = step*k, init = 'uniform', activation = 'relu', input_dim = num_features, use_bias = True,  bias_initializer= 'zeros')) #Adding the input and first hidden layer
                predictor.add(Dense(output_dim = step*j, init = 'uniform', activation = 'relu', input_dim = num_features, use_bias = True,  bias_initializer= 'zeros')) #Adding the input and first hidden layer
                predictor.add(Dense(output_dim = 1, init = 'uniform', activation = 'relu', use_bias = True, bias_initializer='zeros')) #Additional Output Layer
                
                predictor.compile(optimizer = 'adam', loss = 'mse')
                
                predictor.fit(x_train, y_train, batch_size = 100,  nb_epoch = iter_num, validation_split = 0.2)

                y_predicted_train = predictor.predict(x_train) #Train accuracy    
                y_predicted_test = predictor.predict(x_test) #Test accuracy
                
                # R^2 analysis for accuracy measurement
                from sklearn.metrics import r2_score
                accuracy_percent_train = r2_score(y_train , y_predicted_train)*100
                accuracy_percent_test = r2_score(y_test, y_predicted_test)*100
                
                multiple_results_from_same_structure = np.append(multiple_results_from_same_structure, accuracy_percent_test)
                multiple_train_results_from_same_structure = np.append(multiple_train_results_from_same_structure, accuracy_percent_train)
                
            find_max_accuracy = np.append(find_max_accuracy, np.max(multiple_results_from_same_structure))
            find_max_accuracy_train = np.append(find_max_accuracy_train, multiple_train_results_from_same_structure[np.argmax(multiple_results_from_same_structure)])
            num_neurons_in_layer_1 = np.append(num_neurons_in_layer_1, step*k)
            num_neurons_in_layer_2 = np.append(num_neurons_in_layer_2, step*j)
            iter_num_list = np.append(iter_num_list, iter_num)

#Sound after running the above codes
import winsound
duration = 5000 #milliseconds
freq = 800
winsound.Beep(freq, duration)

