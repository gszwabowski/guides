#!/usr/bin/env python
# coding: utf-8
'''
Commentary by GLS 12/1/22
This script is used to predict ligand function for ligand-receptor complexes resulting from docking.
The 'LFP_rf_model.pkl' and 'LFP_label_encoder.pkl' files must be the same directory as this script for it to run.

Input: a .txt file resulting from use of the dockdb_to_lf_input.svl MOE script.
Output: a table printed to the user/.csv file denoting which ligand-receptor complexes have been classified as binders/non-binders.
'''
#module imports/exception handling
try:
    import sklearn
    from sklearn.preprocessing import StandardScaler, LabelEncoder
except:
    msg = "LFP_classifier.py requires the sklearn module."
    print(msg)
    raise Exception(msg)

try:
    import joblib
except:
    msg = "LFP_classifier.py requires the joblib module."
    print(msg)
    raise Exception(msg)
    
try:
    import pandas as pd
except:
    msg = "LFP_classifier.py requires the pandas module."
    print(msg)
    raise Exception(msg)

try:
    import numpy as np
except:
    msg = "LFP_classifier.py requires the numpy module."
    print(msg)
    raise Exception(msg)

import random, os
import csv
import sys
from typing import Tuple

#open pickled machine learning models, handle exception if they are not present in the same directory as this file
try:
    rf_model = joblib.load('LFP_rf_model.pkl')
except:
    msg = "Make sure that 'LFP_rf_model.pkl' is located in the same directory as this Python script."
    print(msg)
    raise Exception(msg)
    
try:
    le = joblib.load('LFP_label_encoder.pkl')
except:
    msg = "Make sure that 'LFP_label_encoder.pkl' is located in the same directory as this Python script."
    print(msg)
    raise Exception(msg)


# Function for label encoding of inttype columns
def encode_labels_ext(df):
        
    # custom encoding for inttypes
    custom_mapping = {'nan' : 0, 'None' : 1, 'Hbond' : 2, 'Distance' : 3, 'Arene' : 4, 'Ionic' : 5, 'Covalent' : 6}
    
    cols = [col for col in df.columns if 'inttype' in col]

    # loop though all columns and convert strings to categorical integer variables using custom mapping
    for col in cols:
        # convert column values to string type 
        df[col] = df[col].astype(str)
         
        # map inttype columns using custom mapping
        df[col] = df[col].map(custom_mapping)
        
    return df
 

 # End label encoding function

 

# Function for scaling/imputing
def scale_impute_ext(dataframe):

    # get empty/non-empty columns for external dataframe
    empty_ext_columns =  []
    nonempty_ext_columns = []
    for col in dataframe.columns.values:
        # all the values for this feature are null
        if sum(dataframe[col].isnull()) == dataframe.shape[0]:
            empty_ext_columns.append(col)
        else:
            nonempty_ext_columns.append(col)
            
    # get colnames
    colnames = nonempty_ext_columns + empty_ext_columns
    
    # impute data
    from sklearn.impute import SimpleImputer
    my_imputer = SimpleImputer()
    df_imputed = pd.DataFrame(my_imputer.fit_transform(dataframe))
    
    #print('df_imputed columns:', len(df_imputed.columns))

    # scale data
    scaler = StandardScaler()
    to_scale = [col for col in df_imputed.columns.values]
    scaler.fit(df_imputed[to_scale])
    df_imputed[to_scale] = scaler.transform(df_imputed[to_scale]) 
    
    # fill empty columns with nan
    df_imputed[empty_ext_columns] = np.nan
    
    # rename columns
    df_imputed.columns = colnames
    
    return(df_imputed)

# end function for scaling/imputing

      
def main():
    #read the input file, handle exception if no file is given
    try:
        ext_df = pd.read_csv(sys.argv[1])
    except:
        msg = "PH4_classifier requires a .csv or txt file as input."
        print(msg)
        raise Exception(msg)
        
    # consideration for tab-separated files
    if len(ext_df.columns) == 1: # for tab separated files
            ext_df = pd.read_csv(sys.argv[1], sep = '\t')
            
    # reset index of ext_df
    ext_df.reset_index(inplace = True, drop = True)
    
    # ask the user if docking was peformed with homology models and assign values for the 'is_HM' predictor
    val = input("Was docking performed with homology models? (y/n): ")
    
    if val == 'y':
        ext_df['isHM'] = 1
    else:
        ext_df['isHM'] = 0
        
    # ask the user how many docked poses were generated per ligand being to classify
    val = input("How many docked poses per ligand?: ")
    poses_per_ligand = int(val)
    
    print('\n')
    
    #print('number of cols in input data:', len(ext_df.columns))
    
    # fill na values for energy columns
    energy_cols = [col for col in ext_df.columns if 'energy' in col]
    ext_df[energy_cols] = ext_df[energy_cols].fillna(999)
    
    # fix missing columns from input data
    model_features = ['1.31_intenergysum', '1.31_inttype1', '1.31_intenergy1', '1.31_inttype2', '1.31_intenergy2', '1.35_intenergysum', '1.35_inttype1', '1.35_intenergy1', '1.35_inttype2', '1.35_intenergy2', '1.39_intenergysum', '1.39_inttype1', '1.39_intenergy1', '1.39_inttype2', '1.39_intenergy2', '1.42_intenergysum', '1.42_inttype1', '1.42_intenergy1', '1.42_inttype2', '1.42_intenergy2', '2.53_intenergysum', '2.53_inttype1', '2.53_intenergy1', '2.53_inttype2', '2.53_intenergy2', '2.54_intenergysum', '2.54_inttype1', '2.54_intenergy1', '2.54_inttype2', '2.54_intenergy2', '2.56_intenergysum', '2.56_inttype1', '2.56_intenergy1', '2.56_inttype2', '2.56_intenergy2', '2.57_intenergysum', '2.57_inttype1', '2.57_intenergy1', '2.57_inttype2', '2.57_intenergy2', '2.58_intenergysum', '2.58_inttype1', '2.58_intenergy1', '2.58_inttype2', '2.58_intenergy2', '2.60_intenergysum', '2.60_inttype1', '2.60_intenergy1', '2.60_inttype2', '2.60_intenergy2', '2.61_intenergysum', '2.61_inttype1', '2.61_intenergy1', '2.61_inttype2', '2.61_intenergy2', '2.63_intenergysum', '2.63_inttype1', '2.63_intenergy1', '2.63_inttype2', '2.63_intenergy2', '2.64_intenergysum', '2.64_inttype1', '2.64_intenergy1', '2.64_inttype2', '2.64_intenergy2', '2.65_intenergysum', '2.65_inttype1', '2.65_intenergy1', '2.65_inttype2', '2.65_intenergy2', '3.25_intenergysum', '3.25_inttype1', '3.25_intenergy1', '3.25_inttype2', '3.25_intenergy2', '3.26_intenergysum', '3.26_inttype1', '3.26_intenergy1', '3.26_inttype2', '3.26_intenergy2', '3.28_intenergysum', '3.28_inttype1', '3.28_intenergy1', '3.28_inttype2', '3.28_intenergy2', '3.29_intenergysum', '3.29_inttype1', '3.29_intenergy1', '3.29_inttype2', '3.29_intenergy2', '3.30_intenergysum', '3.30_inttype1', '3.30_intenergy1', '3.30_inttype2', '3.30_intenergy2', '3.32_intenergysum', '3.32_inttype1', '3.32_intenergy1', '3.32_inttype2', '3.32_intenergy2', '3.33_intenergysum', '3.33_inttype1', '3.33_intenergy1', '3.33_inttype2', '3.33_intenergy2', '3.35_intenergysum', '3.35_inttype1', '3.35_intenergy1', '3.35_inttype2', '3.35_intenergy2', '3.36_intenergysum', '3.36_inttype1', '3.36_intenergy1', '3.36_inttype2', '3.36_intenergy2', '3.37_intenergysum', '3.37_inttype1', '3.37_intenergy1', '3.37_inttype2', '3.37_intenergy2', '3.40_intenergysum', '3.40_inttype1', '3.40_intenergy1', '3.40_inttype2', '3.40_intenergy2', '4.56_intenergysum', '4.56_inttype1', '4.56_intenergy1', '4.56_inttype2', '4.56_intenergy2', '4.57_intenergysum', '4.57_inttype1', '4.57_intenergy1', '4.57_inttype2', '4.57_intenergy2', '4.59_intenergysum', '4.59_inttype1', '4.59_intenergy1', '4.59_inttype2', '4.59_intenergy2', '4.60_intenergysum', '4.60_inttype1', '4.60_intenergy1', '4.60_inttype2', '4.60_intenergy2', '4.61_intenergysum', '4.61_inttype1', '4.61_intenergy1', '4.61_inttype2', '4.61_intenergy2', '5.31_intenergysum', '5.31_inttype1', '5.31_intenergy1', '5.31_inttype2', '5.31_intenergy2', '5.35_intenergysum', '5.35_inttype1', '5.35_intenergy1', '5.35_inttype2', '5.35_intenergy2', '5.36_intenergysum', '5.36_inttype1', '5.36_intenergy1', '5.36_inttype2', '5.36_intenergy2', '5.38_intenergysum', '5.38_inttype1', '5.38_intenergy1', '5.38_inttype2', '5.38_intenergy2', '5.39_intenergysum', '5.39_inttype1', '5.39_intenergy1', '5.39_inttype2', '5.39_intenergy2', '5.40_intenergysum', '5.40_inttype1', '5.40_intenergy1', '5.40_inttype2', '5.40_intenergy2', '5.41_intenergysum', '5.41_inttype1', '5.41_intenergy1', '5.41_inttype2', '5.41_intenergy2', '5.42_intenergysum', '5.42_inttype1', '5.42_intenergy1', '5.42_inttype2', '5.42_intenergy2', '5.43_intenergysum', '5.43_inttype1', '5.43_intenergy1', '5.43_inttype2', '5.43_intenergy2', '5.46_intenergysum', '5.46_inttype1', '5.46_intenergy1', '5.46_inttype2', '5.46_intenergy2', '5.47_intenergysum', '5.47_inttype1', '5.47_intenergy1', '5.47_inttype2', '5.47_intenergy2', '6.44_intenergysum', '6.44_inttype1', '6.44_intenergy1', '6.44_inttype2', '6.44_intenergy2', '6.48_intenergysum', '6.48_inttype1', '6.48_intenergy1', '6.48_inttype2', '6.48_intenergy2', '6.51_intenergysum', '6.51_inttype1', '6.51_intenergy1', '6.51_inttype2', '6.51_intenergy2', '6.52_intenergysum', '6.52_inttype1', '6.52_intenergy1', '6.52_inttype2', '6.52_intenergy2', '6.54_intenergysum', '6.54_inttype1', '6.54_intenergy1', '6.54_inttype2', '6.54_intenergy2', '6.55_intenergysum', '6.55_inttype1', '6.55_intenergy1', '6.55_inttype2', '6.55_intenergy2', '6.58_intenergysum', '6.58_inttype1', '6.58_intenergy1', '6.58_inttype2', '6.58_intenergy2', '6.59_intenergysum', '6.59_inttype1', '6.59_intenergy1', '6.59_inttype2', '6.59_intenergy2', '6.62_intenergysum', '6.62_inttype1', '6.62_intenergy1', '6.62_inttype2', '6.62_intenergy2', '7.31_intenergysum', '7.31_inttype1', '7.31_intenergy1', '7.31_inttype2', '7.31_intenergy2', '7.32_intenergysum', '7.32_inttype1', '7.32_intenergy1', '7.32_inttype2', '7.32_intenergy2', '7.34_intenergysum', '7.34_inttype1', '7.34_intenergy1', '7.34_inttype2', '7.34_intenergy2', '7.35_intenergysum', '7.35_inttype1', '7.35_intenergy1', '7.35_inttype2', '7.35_intenergy2', '7.36_intenergysum', '7.36_inttype1', '7.36_intenergy1', '7.36_inttype2', '7.36_intenergy2', '7.38_intenergysum', '7.38_inttype1', '7.38_intenergy1', '7.38_inttype2', '7.38_intenergy2', '7.39_intenergysum', '7.39_inttype1', '7.39_intenergy1', '7.39_inttype2', '7.39_intenergy2', '7.40_intenergysum', '7.40_inttype1', '7.40_intenergy1', '7.40_inttype2', '7.40_intenergy2', '7.42_intenergysum', '7.42_inttype1', '7.42_intenergy1', '7.42_inttype2', '7.42_intenergy2', '7.43_intenergysum', '7.43_inttype1', '7.43_intenergy1', '7.43_inttype2', '7.43_intenergy2', '7.46_intenergysum', '7.46_inttype1', '7.46_intenergy1', '7.46_inttype2', '7.46_intenergy2', 'isHM']

    # loop through model_features and ensure that the columns are present in ext_df. if not, create them and fill with NA (inttype cols) or 999 (intenergy cols)         
    for feature in model_features:
        if feature not in ext_df.columns.tolist():
            print('Input data does not have the column', feature + '.', 'Creating the column now.\n')
            if 'type' in feature: # for inttype cols.
                ext_df[feature] = np.nan
            else: # for intenergy cols.
                ext_df[feature] = 999

    #print('number of cols in input data after adding model feature columns:', len(ext_df.columns))

    # ensure that no extraneous features are present in the input data        
    input_features = [col for col in ext_df.columns.tolist() if col != 'PDBID']
    input_features = [col for col in input_features if col != 'Function']

    for input_feature in input_features:
        if input_feature not in model_features:
            ext_df.drop(input_feature, axis = 1, inplace = True)
            #print('Dropped column', input_feature + ' from input data.')

    input_features = ext_df.columns.tolist()

    #print('number of cols in input data after dropping columns not in model_features:', len(ext_df.columns))
    
    #ext_df.to_csv('ext_df_test_before_label_encoding.csv')
            
    # label encoding for input data
    ext_df = encode_labels_ext(ext_df)
    
    #ext_df.to_csv('ext_df_test_after_label_encoding.csv')
    
    # fill NA values in input with 999
    ext_df.fillna(999, inplace = True)
    
    # scaling/imputing for input data
    X_ext_imputed = scale_impute_ext(ext_df)

    # make predictions
    y_ext_pred = rf_model.predict(X_ext_imputed)

    # inverse transformation of predictions
    y_ext_pred_actual = le.inverse_transform(y_ext_pred)
    
    # creation of dataframe to return to user
    pred_df = pd.DataFrame({'Complex Number' : range(1, len(y_ext_pred) + 1), 'Predicted Function' : y_ext_pred_actual})
    
    # remap actual/predicted values so that agonist/inv. agonist/antagonists are considered binders, inactives are considered non-binders
    pred_df['Predicted Function'] = pred_df['Predicted Function'].map({'Antagonist': 'Binder', 'Agonist': 'Binder', 'Inverse agonist' : 'Binder', 'Inactive' : 'Non-binder'})
    
    
    # print current set of predictions to user if poses_per_ligand == 1
    if poses_per_ligand == 1:
        print('\n', pred_df.to_string(index=False))
    
        # write predictions to csv
        pred_df.to_csv(os.path.splitext(sys.argv[1])[0] + '_LFP_preds.csv', index = False)
        print('\nResults written to', os.path.splitext(sys.argv[1])[0] + '_LFP_preds.csv.\n')
    
    # if >1 docked pose per ligand, make final prediction based on the most frequently predicted class across n = poses_per_ligand ligands
    elif poses_per_ligand > 1:
        # create list of ligand numbers to assign to y_ext_preds_binary
        ligand_numbers_all = []
        ligand_numbers = []
        actual_binary_fxns = []
        majority_preds = []

        for ligand_num in range(1, int((len(pred_df)/poses_per_ligand)+1)):
            # creation of list to number docked ligand poses
            for i in range(1, poses_per_ligand+1):
                ligand_numbers_all.append(ligand_num)
                
            # creation of list containing unique ligand numbers
            ligand_numbers.append(ligand_num)
  
        # assign each docked complex a ligand number
        pred_df['Ligand Number'] = ligand_numbers_all

        # for each ligand represented by n = poses_per_ligand docked complexes, get its predictions
        for ligand_num in range(1, int((len(pred_df)/poses_per_ligand)+1)):
            lig_df = pred_df.loc[pred_df['Ligand Number'] == ligand_num]
            # print(lig_df)
                
            # use first index of value_counts (which is descendingly sorted) to get the majority prediction
            majority_preds.append(lig_df['Predicted Function'].value_counts().index.tolist()[0])
            
        # complex numbering for new_pred_df
        complex_numbers = []
        first_lig_complex = 1
        last_lig_complex = 5
        
        for i in range(1, len(ligand_numbers) + 1):
            lig_complex_nums = str(first_lig_complex) + '-' + str(last_lig_complex)
            complex_numbers.append(lig_complex_nums)
            first_lig_complex += poses_per_ligand
            last_lig_complex += poses_per_ligand
            
        # create new predictions df
        new_pred_df = pd.DataFrame({'Ligand Number' : ligand_numbers,
                                    'Docked Complexes' : complex_numbers,
                                    'Prediction' : majority_preds})

        print(new_pred_df)
        
        # write predictions to csv
        new_pred_df.to_csv(os.path.splitext(sys.argv[1])[0] + '_LFP_preds.csv', index = False)
        print('\nResults written to', os.path.splitext(sys.argv[1])[0] + '_LFP_preds.csv.\n')
    
    
if __name__ == '__main__':
   main()
	




