import numpy as np

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from imblearn.over_sampling import SMOTE

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

import matplotlib.pyplot as plt

import os
import datetime

current_folder_path = os.getcwd()
start_time = str(datetime.datetime.now())

added_feature_name = 'PGV'
resample_on = True

#The purpose of this script is to train NN using certain features.
data_file_name = '\dataset_kappa surface_1208_(Additional Vss - Vs at 100) (with LN).csv'
data_folder_path = '.\input_data'
original_dataset = pd.read_csv(data_folder_path + data_file_name)
dataset = original_dataset.loc[:,
                               ['Strain Category', \
                                'Vs30', 'VsMin', 'Max Vs30', \
                                'Rrup', 'Z2.5', 'Depth', added_feature_name, \
                                'kappa surface']
                               ]

#Set column names list before running simulation.
#Use non-ln() feature name.        
column_names = list(['Vs30', 'VsMin', 'Max Vs30', \
                     'Rrup', 'Z2.5', 'Depth', added_feature_name, \
                     'kappa surface'])                               

#separate input dataset (x) that includes all motions
x_all = dataset.drop(['kappa surface'], axis=1)

#Clean and separate dataset
x = dataset.drop(['kappa surface'], axis=1)
y = dataset.loc[:, ['kappa surface']]

#Split dataset into 2 groups (training and testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = 1)

#Drop strain category column in x_test and x_all datasets.
x_test = x_test.drop(['Strain Category'], axis=1)
x_all = x_all.drop(['Strain Category'], axis=1)

#Resampling training dataset
train_dataset = pd.concat([x_train,y_train], axis=1, sort=False)
train_strain_cat = train_dataset.loc[:, ['Strain Category']]
train_dataset = train_dataset.drop(['Strain Category'], axis=1)

#Resampling
sampling_method = SMOTE()
train_dataset, train_strain_cat = sampling_method.fit_sample(train_dataset, train_strain_cat)
train_dataset = pd.DataFrame(train_dataset)

train_dataset.columns = column_names

#Drop Strain Category column in dataset variable. 
#The column is only useful for resampling
dataset = dataset.drop('Strain Category', axis=1)

if resample_on == True:
    #Not resampled train dataset
    x_train_no_resample = []
    y_train_no_resample = []
    x_train_no_resample = x_train
    x_train_no_resample = x_train_no_resample.drop(['Strain Category'], axis=1)
    y_train_no_resample = y_train
    
    #Resampled train dataset
    x_train = train_dataset.drop(['kappa surface'], axis=1)
    y_train = train_dataset.loc[:, ['kappa surface']]
else:
    x_train = x_train.drop(['Strain Category'],axis=1)


#Normalization
normalization = StandardScaler()
x_train = normalization.fit_transform(x_train)
if resample_on == True:
    x_train_no_resample = normalization.transform(x_train_no_resample)
x_test = normalization.transform(x_test)
x_all_n = normalization.transform(x_all)

#Column names for Gradual Feature Removal
column_names = list(dataset.columns)

#No. of columns and rows of the imported dataset
num_features = x_train.shape[1]
num_motions = x_train.shape[0]

######################### Parameter Tuning ##################################
batch_num = 100
iter_num = 100
layer_num = 1

find_max_accuracy_all_n = []
find_max_accuracy_test = []
find_max_accuracy_train = []
#Inverse
find_max_accuracy_all_n_exp = []
find_max_accuracy_test_exp = []
find_max_accuracy_train_exp = []

num_neurons_in_each_layer = []
iter_num_list = []
batch_num_list = []
layer_num_list = []

for iter_num in range(100,101,20):
    for layer_num in range(4,5,2): 
        for k in range(25,75,5): #k = Number of neurons
            multiple_test_results_from_same_structure = []
            multiple_train_results_from_same_structure = []
            multiple_all_n_results_from_same_structure = []
            #Inverse
            multiple_test_results_from_same_structure_exp = []
            multiple_train_results_from_same_structure_exp = []
            multiple_all_n_results_from_same_structure_exp = []
            for l in range(5):
                predictor = Sequential()            
                #Hidden Layers
                for m in range(layer_num):
                    predictor.add(Dense(output_dim = k, init = 'uniform', 
                                        activation = 'relu', input_dim = num_features, 
                                        use_bias = True,  bias_initializer= 'zeros')) #Adding the input and first hidden layer
                #Output Layer
                predictor.add(Dense(output_dim = 1, init = 'uniform', 
                                    activation = 'relu', 
                                    use_bias = True, bias_initializer='zeros')) #Additional Output Layer
                
                predictor.compile(optimizer = 'adam', loss = 'mse')
                
                predictor.fit(x_train, y_train, batch_size = batch_num,  
                              nb_epoch = iter_num, validation_split = 0.2)
                
                if resample_on == True:
                    y_predicted_train = predictor.predict(x_train_no_resample) #Train accuracy
                else:
                    y_predicted_train = predictor.predict(x_train) #Train accuracy
                y_predicted_test = predictor.predict(x_test) #Test accuracy
                y_predicted_all_n = predictor.predict(x_all_n) #All motinos
                
                # R^2 analysis for accuracy measurement
                if resample_on == True:
                    accuracy_percent_train = r2_score(y_train_no_resample, y_predicted_train)*100
                else:
                    accuracy_percent_train = r2_score(y_train, y_predicted_train)*100
                accuracy_percent_test = r2_score(y_test, y_predicted_test)*100
                accuracy_percent_all_n = r2_score(y, y_predicted_all_n)*100
                
                multiple_test_results_from_same_structure = np.append(multiple_test_results_from_same_structure, accuracy_percent_test)
                multiple_train_results_from_same_structure = np.append(multiple_train_results_from_same_structure, accuracy_percent_train)
                multiple_all_n_results_from_same_structure = np.append(multiple_all_n_results_from_same_structure, accuracy_percent_all_n)
                
            find_max_accuracy_test = np.append(find_max_accuracy_test, np.max(multiple_test_results_from_same_structure))
            find_max_accuracy_train = np.append(find_max_accuracy_train, multiple_train_results_from_same_structure[np.argmax(multiple_test_results_from_same_structure)])
            find_max_accuracy_all_n = np.append(find_max_accuracy_all_n, multiple_all_n_results_from_same_structure[np.argmax(multiple_test_results_from_same_structure)])
            
            num_neurons_in_each_layer = np.append(num_neurons_in_each_layer, k)
            iter_num_list = np.append(iter_num_list, iter_num)
            batch_num_list = np.append(batch_num_list, batch_num)
            layer_num_list = np.append(layer_num_list, layer_num)
            
            #Accuracy difference between test and train to check overfitting issue.
            accuracy_differnce = find_max_accuracy_train - find_max_accuracy_test

#Genearte/Update structure list datatable
previous_structure_list = pd.read_excel('nn_structure_list_' + added_feature_name+'.xlsx')
new_structure_list = pd.DataFrame({'Test': find_max_accuracy_test, \
                                   'Train': find_max_accuracy_train, \
                                   'Difference': accuracy_differnce, \
                                   'All Data': find_max_accuracy_all_n, \
                                   'Num of Neurons in each layer': num_neurons_in_each_layer, \
                                   'Interation Num': iter_num_list, \
                                   'Bach Size': batch_num_list, \
                                   'Num of Layers': layer_num_list, \
                                   'Resampled': resample_on, \
                                   'Trial Start Time': start_time})
updated_structure_list = previous_structure_list.append(new_structure_list)
writer = ExcelWriter('nn_structure_list_'+added_feature_name+'.xlsx')
updated_structure_list.to_excel(writer,'Sheet1',index=False)
writer.save()

#Sound after running the above codes
import winsound
duration = 5000 #milliseconds
freq = 800
winsound.Beep(freq, duration)
