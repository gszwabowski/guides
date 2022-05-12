#!/usr/bin/env python
# coding: utf-8
'''
Commentary by GLS 2/22/22
This script is used to classify pharmacophore models that have been generated with our research group's score-based
pharmacophore modeling protocol. The kmeans_5clusters.pkl and clusterI_regression_model.pkl files must be present in
the same directory as this script for it to run.

Input: a .csv file resulting from use of the scorebased_datacollection.svl MOE script.
Output: a table printed to the user/.csv file denoting which pharmacophore models have been classified as quality. 
'''
#module imports/exception handling
try:
    import sklearn
    from sklearn import preprocessing, neighbors
    from sklearn import model_selection
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn import model_selection
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import scale
except:
    msg = "PH4_classifier.py requires the sklearn module."
    print(msg)
    raise Exception(msg)

try:
    import pickle
except:
    msg = "PH4_classifier.py requires the pickle module."
    print(msg)
    raise Exception(msg)
    
try:
    import pandas as pd
except:
    msg = "PH4_classifier.py requires the pandas module."
    print(msg)
    raise Exception(msg)

try:
    import numpy as np
except:
    msg = "PH4_classifier.py requires the numpy module."
    print(msg)
    raise Exception(msg)

import random, os
import csv
import sys
from typing import Tuple

#open pickled machine learning models, handle exception if they are not present in the same directory as this file
try:
    with open('kmeans_5clusters.pkl', 'rb') as f:
        clustering = pickle.load(f)
except:
    msg = "Make sure that the 'kmeans_5clusters.pkl' is located in the same directory as this Python script."
    print(msg)
    raise Exception(msg)
   
try:   
    with open('clusterI_regression_model.pkl', 'rb') as f:
        sgdc0 = pickle.load(f)
except:
    msg = "Make sure that the 'clusterI_regression_model.pkl' is located in the same directory as this Python script."
    print(msg)
    raise Exception(msg)

def scale_features_single(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    applies standard scaler (z-scores) to training data and predicts z-scores for the test set
    """
    scaler = StandardScaler()
    to_scale = [col for col in X.columns.values]
    scaler.fit(X[to_scale])
    X[to_scale] = scaler.transform(X[to_scale])
    
    return X
      
def main():
    #read the input file, handle exception if no file is given
    try:
        ext_df = pd.read_csv(sys.argv[1])
    except:
        msg = "PH4_classifier requires a .csv file input."
        print(msg)
        raise Exception(msg)
    
    #if filename is not .csv, inform the user
    split_tup = os.path.splitext(sys.argv[1])
    file_extension = split_tup[1]
    if file_extension != '.csv':
        raise Exception('Input filetype must be .csv.')
    
    #fill NA values in input with -99999
    ext_df.fillna(-99999)
    
    #extract columns with text and columns with data that needs to be non-scaled when returned to the user
    receptors = ext_df.Receptor
    hits_actual = ext_df.Hits
    score_types = ext_df['score_type']
    subsets = ext_df.subset
    match_features = ext_df.match_features
    
    #extract predictor columns from input csv
    ext_df = ext_df[['s_score','Hits', 'max_feat', 'avg_feat', 'max_centr', 'min_centr', 'avg_centr', 'features', 'all_same', 'hyd_prop', 'don_prop', 'catdon_prop', 'hydaro_prop', 'aniacc_prop']]
    x = ext_df
    
    # predict cluster labels for the data
    ext_labels = clustering.predict(x)
    X_clstrs = x.copy()
    
    X_scaled = scale_features_single(X_clstrs)
    ext_clusters = X_scaled.copy()
    
    #add receptors, hits_actual, score_type, and subset columns back prior to 0/1/2/3 split
    ext_clusters['Receptor'] = receptors
    ext_clusters['hits_actual'] = hits_actual
    ext_clusters['score_type'] = score_types
    ext_clusters['subset'] = subsets
    ext_clusters['match_features'] = match_features
    ext_clusters['clusters'] = ext_labels
    
    #get cluster values to match to input pharmacophore model clustering results
    uniq_clusters = ext_clusters['clusters'].unique()
    uniqs = uniq_clusters.tolist()
    uniqs.sort()
    
    #locate the "0" cluster
    ext_0 = ext_clusters.loc[ext_clusters.clusters == 0]
    ext_0_receptors = ext_0.Receptor
    ext_0_hits_actual = ext_0.hits_actual
    ext_0_score_types = ext_0['score_type']
    ext_0_subsets = ext_0.subset
    ext_0_match_features = ext_0.match_features

    #drop columns from the dataframe that are not used as predictors
    X_ext_0 = ext_0.drop(columns=['Receptor', 'hits_actual', 'score_type', 'subset', 'match_features'])
    
    #predict quality classes for input pharmacophore models segregated into the first cluster
    print('\nPharmacophore models predicted as quality:\n')
    y_pred = (sgdc0.predict(X_ext_0))
    
    #add columns that were dropped prior to classification
    X_ext_0['Receptor'] = ext_0_receptors
    X_ext_0['hits_actual'] = ext_0_hits_actual
    X_ext_0['score_type'] = ext_0_score_types
    X_ext_0['subset'] = ext_0_subsets
    X_ext_0['match_features'] = ext_0_match_features
    X_ext_0['quality_pred'] = y_pred

    #print the ph4s classified as quality and write them to .csv
    selected_ph4s = X_ext_0.loc[X_ext_0['quality_pred'] == 1]
    selected_ph4s = selected_ph4s[['Receptor','hits_actual', 'score_type', 'subset', 'match_features', 'quality_pred']]
    print(selected_ph4s)
    ph4_preds = X_ext_0.loc[X_ext_0['quality_pred'] == 1]
    ph4_preds.to_csv(os.path.splitext(sys.argv[1])[0] + '_clusterI_ph4_preds.csv', index = False)
    print('\nResults written to', os.path.splitext(sys.argv[1])[0] + 'clusterI_ph4_preds.csv.\n')
            
if __name__ == '__main__':
   main()
	




