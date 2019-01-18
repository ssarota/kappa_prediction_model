import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from imblearn.over_sampling import SMOTE

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

import matplotlib.pyplot as plt

import os

current_folder_path = os.getcwd()

added_feature_name = 'PGV'
file_name = added_feature_name
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

#Assign 1 case
case = [[0,1,2,3,4,5,6]]

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
dataset = dataset.drop(['Strain Category'], axis=1)

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
    x_train = x_train.drop(['kappa surface'], axis=1)

#Need to generate imaginary dataset to plot kappa trend of each feature.
x_img_avg = dataset.drop(['kappa surface'], axis=1)
avg = np.mean(x_img_avg)
for i in range(np.shape(x_img_avg)[1]):
    x_img_avg.iloc[:,i] = avg[i]
#x_img_values = Fake values ranging from Max and Min of each column
x_img_values = dataset.drop(['kappa surface'], axis=1) 
for i in range(np.shape(x_img_values)[1]):
    min_value = np.amin(x_img_values.iloc[:,i])
    max_value = np.amax(x_img_values.iloc[:,i])
    x_img_values_each_col = np.linspace(min_value, max_value, 
                                        num=np.shape(x_img_values)[0])
    x_img_values.iloc[:,i] = x_img_values_each_col

#Normalization
normalization = StandardScaler()

x_train_before_normalization = x_train
if resample_on == True:
    x_train_no_resample_before_normalization = x_train_no_resample
x_test_before_normalization = x_test
x_all_before_normalization = x_all

x_train = normalization.fit_transform(x_train)
if resample_on == True:
    x_train_no_resample = normalization.transform(x_train_no_resample)
x_test = normalization.transform(x_test)
x_all = normalization.transform(x_all)

#Normalize imaginary values
x_img_avg_normalized = normalization.transform(x_img_avg)
x_img_values_normalized = normalization.transform(x_img_values)

#Save normalization factors for later
normalization_mean = normalization.mean_
np.save(str(file_name+'_normalize_mean'), normalization_mean)
normalization_scale = normalization.scale_
np.save(str(file_name+'_normalize_scale'), normalization_scale)

#Assign empty temporary variables before FOR loops
temp_case = []
temp_x_train = []
temp_x_test = []
temp_x_all = []

accuracy_of_each_case_test = []
accuracy_of_each_case_train = []
accuracy_of_each_case_all = []

lnSig_of_each_addition_test = []
lnSig_of_each_addition_train = []
lnSig_of_each_addition_all = []

layer_num = 4

#Training Nueral Network and Plot Results and Trends
for i in range(0,1):
    #Variables reset for each case
    temp_case = case[i]
    if resample_on == True:
        temp_x_train_no_resample = x_train_no_resample[:,temp_case]
    temp_x_train = x_train[:,temp_case]
    temp_x_test = x_test[:,temp_case]
    temp_x_all = x_all[:,temp_case]
    
    find_test_accuracy_each_case = []
    find_train_accuracy_each_case = []
    find_max_accuracy_each_case = []

    find_train_lnSig_each_feature = []
    find_test_lnSig_each_feature = []
    find_max_lnSig_each_feature = []
    
    num_features = temp_x_train.shape[1]
    num_motions = temp_x_train.shape[0]
    
    each_case_accuracy_list_test = []
    
    for j in range(10):
        print('Case # = ', i)
        
        predictor = Sequential()
        
        for k in range(0,layer_num):
            #Adding the input and first hidden layer
            predictor.add(Dense(output_dim = 50, init = 'uniform', 
                                activation = 'relu', input_dim = num_features, 
                                use_bias = True,  bias_initializer= 'zeros')) 
            
        #Additional Output Layer
        predictor.add(Dense(output_dim = 1, init = 'uniform', 
                            activation = 'relu', 
                            use_bias = True, bias_initializer='zeros'))
        
        predictor.compile(optimizer = 'adam', loss = 'mse')
        
        predictor.fit(temp_x_train, y_train, batch_size = 100, 
                      nb_epoch = 100, validation_split = 0.2)
        
        #Train accuracy
        if resample_on == True:
            y_predicted_train = predictor.predict(temp_x_train_no_resample)
        else:
            y_predicted_train = predictor.predict(temp_x_train)
        
        #Test accuracy
        y_predicted_test = predictor.predict(temp_x_test) 
        
        #Whole dataset
        y_predicted = predictor.predict(temp_x_all) 
        
        # R^2 analysis for accuracy measurement
        if resample_on == True:
            accuracy_percent_train = r2_score(y_train_no_resample, y_predicted_train)*100     
        else:
            accuracy_percent_train = r2_score(y_train, y_predicted_train)*100     
        accuracy_percent_test = r2_score(y_test, y_predicted_test)*100
        accuracy_percent = r2_score(y, y_predicted)*100
        
        find_test_accuracy_each_case = np.append(find_test_accuracy_each_case, 
                                                 accuracy_percent_test) 
        find_train_accuracy_each_case = np.append(find_train_accuracy_each_case, 
                                                  accuracy_percent_train)
        find_max_accuracy_each_case = np.append(find_max_accuracy_each_case, 
                                                accuracy_percent)
        
        # Calculating ln Sig
        #ln Sigma (Whole dataset)
        n = y.shape[0]
        ln_sig = np.sqrt(np.sum(np.square(np.log(y) - np.log(y_predicted)))/n)
        ln_sig = ln_sig[0]
        
        if resample_on == True: 
            n_train = y_train_no_resample.shape[0]
            ln_sig_train = np.sqrt(np.sum(np.square(np.log(y_train_no_resample) 
                                                    -np.log(y_predicted_train)))
                                                    /n_train)
            ln_sig_train = ln_sig_train[0]
        else:
            n_train = y_train.shape[0]
            ln_sig_train = np.sqrt(np.sum(np.square(np.log(y_train) 
                                                    -np.log(y_predicted_train)))
                                                    /n_train)
            ln_sig_train = ln_sig_train[0]
        
        n_test = y_test.shape[0]
        ln_sig_test = np.sqrt(np.sum(np.square(np.log(y_test)
                                               -np.log(y_predicted_test)))
                                               /n_test)
        ln_sig_test = ln_sig_test[0]
        
        find_train_lnSig_each_feature = np.append(find_train_lnSig_each_feature, 
                                                  ln_sig_train)
        find_test_lnSig_each_feature = np.append(find_test_lnSig_each_feature, 
                                                 ln_sig_test)
        find_max_lnSig_each_feature = np.append(find_max_lnSig_each_feature, 
                                                ln_sig)
    
    #Record accuracy of each case
    index_of_max_test_accuracy = np.argmax(find_test_accuracy_each_case)
    each_case_accuracy_list_test = np.append(each_case_accuracy_list_test, 
                                              np.amax(find_test_accuracy_each_case))
    accuracy_temp_feature = np.amax(find_test_accuracy_each_case)
    accuracy_percent_test = 0
    count_rerun = 0
    
    temp_title = []
    for col_num in temp_case:
        temp_title = np.append(temp_title, column_names[col_num])
    temp_title = str(temp_title)
    
    while accuracy_percent_test <= (accuracy_temp_feature):
        
        predictor = Sequential()
        for k in range(0,layer_num):
            #Adding the input and first hidden layer
            predictor.add(Dense(output_dim = 110, init = 'uniform', 
                                activation = 'relu', input_dim = num_features, 
                                use_bias = True,  bias_initializer= 'zeros')) 
            
        #Additional Output Layer
        predictor.add(Dense(output_dim = 1, init = 'uniform', 
                            activation = 'relu', 
                            use_bias = True, bias_initializer='zeros'))
        
        predictor.compile(optimizer = 'adam', loss = 'mse')
        
        predictor.fit(temp_x_train, y_train, batch_size = 100, 
                      nb_epoch = 100, validation_split = 0.2)
        
        if resample_on == True:
            y_predicted_train = predictor.predict(temp_x_train_no_resample)
        else:
            y_predicted_train = predictor.predict(temp_x_train)
        y_predicted_test = predictor.predict(temp_x_test) #Test accuracy
        y_predicted = predictor.predict(temp_x_all) #Whole dataset
        
        #Save NN model
        predictor.save(file_name+'_nn_structure.h5')
        predictor.save_weights(file_name+'nn_weights.h5')
        weight = predictor.get_weights()
        
        # R^2 analysis for accuracy measurement
        if resample_on == True:
            accuracy_percent_train = r2_score(y_train_no_resample, y_predicted_train)*100     
        else:
            accuracy_percent_train = r2_score(y_train, y_predicted_train)*100     
        accuracy_percent_test = r2_score(y_test, y_predicted_test)*100
        accuracy_percent = r2_score(y, y_predicted)*100
        
        # Calculating ln Sig
        #ln Sigma (Whole dataset)
        n = y.shape[0]
        ln_sig = np.sqrt(np.sum(np.square(np.log(y) - np.log(y_predicted)))/n)
        ln_sig = ln_sig[0]
        
        #ln Sigma (separated into Train and Test sets)
        if resample_on == True:
            n_train = y_train_no_resample.shape[0]
            ln_sig_train = np.sqrt(np.sum(np.square(np.log(y_train_no_resample) 
                                                    -np.log(y_predicted_train)))
                                                    /n_train)
            ln_sig_train = ln_sig_train[0]
        else:
            n_train = y_train.shape[0]
            ln_sig_train = np.sqrt(np.sum(np.square(np.log(y_train) 
                                                    -np.log(y_predicted_train)))
                                                    /n_train)
            ln_sig_train = ln_sig_train[0]
        
        n_test = y_test.shape[0]
        ln_sig_test = np.sqrt(np.sum(np.square(np.log(y_test)
                                               -np.log(y_predicted_test)))
                                               /n_test)
        ln_sig_test = ln_sig_test[0]

        #Fig. measured kappa vs predicted kappa values
        #Results from Training Dataset
        plt.title(temp_title+' (Training Dataset)')
        plt.xlabel(r'$\kappa_{predicted}$ (s)')
        plt.ylabel(r'$\kappa_{measured}$ (s)')
        plt.axis([0,0.2,  0,0.2])
        plt.minorticks_on()
        plt.autoscale(False)
        plt.plot([0,0.2],[0,0.2], linestyle='-', color='k', linewidth=0.8)
        if resample_on == True:
            plt.plot(y_predicted_train, y_train_no_resample, 'ko')
        else:
            plt.plot(y_predicted_train, y_train, 'ko')
        plt.legend(bbox_to_anchor=(0.8, 0.2), loc=2, borderaxespad=0)
        plt.text(0.01, 0.16, 
                 r'$R^2$ = '+str(round(accuracy_percent_train/100,4)), 
                 fontsize=12, )
        plt.text(0.01, 0.145, 
                 r'$\sigma_{ln}$ = '+str(round(ln_sig_train,4)), 
                 fontsize=12, )
        plt.savefig(file_name+'_train_case.png')
        plt.clf()
        
        #Results from Testing Dataset
        plt.title(temp_title+' (Testing Dataset)')
        plt.xlabel(r'$\kappa_{predicted}$ (s)')
        plt.ylabel(r'$\kappa_{measured}$ (s)')
        plt.axis([0,0.2,  0,0.2])
        plt.minorticks_on()
        plt.autoscale(False)
        plt.plot([0,0.2],[0,0.2], linestyle='-', color='k', linewidth=0.8)
        plt.plot(y_predicted_test, y_test, 'ko')
        plt.legend(bbox_to_anchor=(0.8, 0.2), loc=2, borderaxespad=0)
        plt.text(0.01, 0.16, 
                 r'$R^2$ = '+str(round(accuracy_percent_test/100,4)), 
                 fontsize=12, )
        plt.text(0.01, 0.145, 
                 r'$\sigma_{ln}$ = '+str(round(ln_sig_test,4)), 
                 fontsize=12, )
        plt.savefig(file_name+'_test_case.png')
        plt.clf()
        
        #Results from Original Dataset
        plt.title(temp_title)
        plt.xlabel(r'$\kappa_{predicted}$ (s)')
        plt.ylabel(r'$\kappa_{measured}$ (s)')
        plt.axis([0,0.2,  0,0.2])
        plt.minorticks_on()
        plt.autoscale(False)
        plt.plot([0,0.2],[0,0.2], linestyle='-', color='k', linewidth=0.8)
        plt.plot(y_predicted, y, 'ko')
        plt.legend(bbox_to_anchor=(0.8, 0.2), loc=2, borderaxespad=0)
        plt.text(0.01, 0.16, 
                 r'$R^2$ = '+str(round(accuracy_percent/100,4)), 
                 fontsize=12, )
        plt.text(0.01, 0.145, 
                 r'$\sigma_{ln}$ = '+str(round(ln_sig,4)), 
                 fontsize=12, )
        plt.savefig(file_name+'_train+test_original_case.png')
        plt.clf()
        
        print('Accuracy max = ', accuracy_temp_feature)
        print('Accuracy rerun = ', accuracy_percent_test)
        count_rerun += 1
        if count_rerun == 50:
            accuracy_temp_feature -= 2
        elif count_rerun == 100:
            accuracy_temp_feature -= 5
    
    predictor = load_model(file_name+'_nn_structure.h5')
    
    #Plot trend of each feature
    for m in range(len(temp_case)):
        #Set data to x_img_temp
        x_img_temp = np.zeros((np.shape(x_img_values)[0], len(temp_case)))
        x_img_temp = x_img_avg_normalized[:,temp_case]
        x_img_temp[:,m] = x_img_values_normalized[:,m]
        
        #Imput data into NN model
        y_predicted = predictor.predict(temp_x_all)
        y_img_predicted = predictor.predict(x_img_temp)
        
        #R^2 Analysis
        r_square = r2_score(y, y_predicted)
        
        #Log Sigma
        ln_sig = np.sqrt(np.sum(np.square(np.log(y)
                                -np.log(y_predicted)))
                                /np.shape(x)[0])
        ln_sig = ln_sig[0]
        
        #Plot
        feature = dataset.iloc[:,temp_case[m]]
        max_x = np.amax(feature)
        min_x = np.amin(feature)
        feature_img = x_img_values.iloc[:,temp_case[m]]
        
        plt.xlabel(column_names[temp_case[m]])
        plt.ylabel(r'$\kappa$ (s)')
        plt.axis([0,max_x*2,  0, 0.20])
        plt.grid(color='0.5', which = 'major', linestyle='-', linewidth=0.8)
        plt.minorticks_on()
        plt.autoscale(True)
        plt.plot(feature, y, 'ko')
        plt.plot(feature_img, y_img_predicted, '-')
        plt.savefig(file_name
                    +'_trend_feature-'+str(i)
                    +str(m)
                    +'.png')
        plt.clf()
        
        plt.xlabel(column_names[temp_case[m]])
        plt.ylabel(r'$\kappa$ (s)')
        plt.axis([0,max_x*2,  0, 0.20])
        plt.grid(color='0.5', which = 'major', linestyle='-', linewidth=0.5)
        plt.grid(color='0.5', which = 'minor', linestyle='--', linewidth=0.5)
        plt.minorticks_on()
        plt.autoscale(True)
        plt.loglog(feature, y, 'ko')
        plt.loglog(feature_img, y_img_predicted, '-')
        plt.savefig(file_name
                    +'_trend_feature-'
                    +str(m)
                    +'_loglog.png', bbox_inches = "tight")
        plt.clf()
    
    #Record accuracy of each addition
    accuracy_of_each_case_test = np.append(accuracy_of_each_case_test,
                                           accuracy_percent_test)
    accuracy_of_each_case_train = np.append(accuracy_of_each_case_train,
                                            accuracy_percent_train)
    accuracy_of_each_case_all = np.append(accuracy_of_each_case_all, 
                                          accuracy_percent)
    
    lnSig_of_each_addition_test = np.append(lnSig_of_each_addition_test, 
                                            ln_sig_test)
    lnSig_of_each_addition_train = np.append(lnSig_of_each_addition_train, 
                                             ln_sig_train)
    lnSig_of_each_addition_all = np.append(lnSig_of_each_addition_all, 
                                           ln_sig)

import winsound
duration = 5000
freq = 800
winsound.Beep(freq, duration)
