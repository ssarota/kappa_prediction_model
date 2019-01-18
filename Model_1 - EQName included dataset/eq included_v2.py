import pandas as pd
import numpy as np
from skrebate import ReliefF
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

#Import Raw Dataset
original_dataset = pd.read_csv('.\data\input_datasets\dataset_kappa surface_1031_(All fetures + EQ ID + missing values).csv')
dataset = original_dataset

select_features = 1

if select_features == 1:
   selected_features_and_output = np.array(['Strain Category', 'EQName', 'VsMax','VsMin','Vs50','PGA (g)','Rrup1', 'Magnitude', 'Depth2', 'kappa surface'])
   dataset = original_dataset.loc[:, selected_features_and_output]    

#Import NN structure
import keras
from keras.models import load_model
predictor = load_model('nn_structure_f-7_r-3.h5') #Load Structure

#Import Normalization files
from sklearn.preprocessing import StandardScaler
normalization = StandardScaler()
normalization.mean_ = np.load('normalize_mean_f-7_r-3.npy') 
normalization.scale_ = np.load('normalize_scale_f-7_r-3.npy')

##############################################################################

if select_features == 1:
    #Column name list from the dataset of the selected features. (Ex. 9th column = VsZhole)
    column_names = np.array(list(dataset.drop(['Strain Category', 'EQName'],axis = 1)))

#Separating dataset to input and output
x = dataset.drop(['kappa surface', 'EQName', 'Strain Category'], axis=1)
y = dataset.loc[:, ['kappa surface']]

#Normalization
x_normalized = normalization.transform(x)

#No. of columns and rows of the imported dataset
num_features = x.shape[1]
num_motions = x.shape[0]

y_predicted = predictor.predict(x)     

#Both Test and Train together with EQ Name, Strain Category Columns.
#############################################################################
y_predicted_kappa_all = predictor.predict(x_normalized) #Predict Kappa based on both train and test datasets

Column_1 = dataset.loc[:, ['EQName', 'Strain Category']]
Column_2 = x
Column_3 = y
Column_4 = y_predicted_kappa_all

#############################################################################

# R^2 analysis for accuracy measurement
from sklearn.metrics import r2_score
accuracy = r2_score(y, y_predicted_kappa_all)*100

# ln Sigma
#############################################################################
#Separating dataset to input and output

y_original_for_ln_sig = y

#Log Sigma
ln_sig = np.sqrt(np.sum(np.square(np.log(y) - np.log(y_predicted_kappa_all)))/num_motions)
ln_sig = ln_sig[0]

#############################################################################

import matplotlib.pyplot as plt
#Fig. After resampling measured vs predicted kappa values (with selected features)
#Results from Training Dataset
#Results from Original Dataset
plt.xlabel(r'$\kappa_{predicted}$ (s)')
plt.ylabel(r'$\kappa_{measured}$ (s)')
plt.axis([0,0.18,  0,0.18])
plt.minorticks_on()
plt.autoscale(False)
plt.plot([0 , 0.2],[0 , 0.2], linestyle = '-', color = 'k', linewidth = 0.8)
plt.plot(y_predicted_kappa_all, y, 'ko')
plt.legend(bbox_to_anchor=(0.8, 0.2), loc=2, borderaxespad=0)
plt.text(0.01, 0.16, r'$R^2$ = '+str(round(accuracy/100,4)), fontsize=12, )
plt.text(0.01, 0.145, r'$\sigma_{ln}$ = '+str(round(ln_sig,4)), fontsize=12, )
plt.savefig(str('train+test_original_EQName_v2.png'))
plt.clf()

