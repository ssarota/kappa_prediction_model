import pandas as pd
import numpy as np
from skrebate import ReliefF
from sklearn.model_selection import train_test_split

#Import Data set
data_set = pd.read_csv('.\data\input_datasets\dataset_kappa surface_0930_v2.csv')
column_drop_list = ['Strain Category', 'kappa surface (EW)']
data_input, data_output = data_set.drop(column_drop_list, axis=1).values, data_set['kappa surface (EW)'].values
feature_name_list = data_set.drop(column_drop_list, axis=1).columns

#Randomly divide the dataset into train set and test set.
data_input_train, data_input_test, data_output_train, data_output_test = \
    train_test_split(data_input, data_output, test_size=0.5, train_size=0.5)

#Use ReliefF module
    ##Train
feature_selection_train = ReliefF()
feature_selection_train.fit(data_input_train, data_output_train)
    ##Test
feature_selection_test = ReliefF()
feature_selection_test.fit(data_input_test, data_output_test)
    #Overall
feature_selection_all = ReliefF()
feature_selection_all.fit(data_input, data_output)

#Tabulate feature scores
    #Train
feature_score_table_train = pd.DataFrame(feature_selection_train.feature_importances_)
feature_score_table_train.index = feature_name_list
print (feature_score_table_train)
    #Test
feature_selection_test = pd.DataFrame(feature_selection_test.feature_importances_)
feature_selection_test.index = feature_name_list
print (feature_selection_test)
    #Overall
feature_selection_all = pd.DataFrame(feature_selection_all.feature_importances_)
feature_selection_all.index = feature_name_list
print (feature_selection_all)

#Export as a table
feature_selection_all.to_csv('relief_score.csv', header = False, index = True)
