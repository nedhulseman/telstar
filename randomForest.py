'''
    @data      --   Telstar's Kaggle Competition
    @author    --   Ned Hulseman
    @date      --   2/14/2017
    @Purpose   --   Interworks data assesment
    
    
    @link https://www.kaggle.com/c/telstra-recruiting-network/data

    Script Objective:
        Predict service faults and create some predictive models using
        random forest
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import GridSearchCV


data_path = r'C:\Users\nedhu\Desktop\Interworks Assesment\Data\\'
trainX  =  pd.read_csv(data_path + 'X_train.csv', index_col=0)
testX   =  pd.read_csv(data_path + 'X_test.csv', index_col=0)
trainY  =  pd.read_csv(data_path + 'y_train.csv', index_col=0)
testY   =  pd.read_csv(data_path + 'y_test.csv', index_col=0)



############################### random forest ##################################
###############################################################################



param_grid={
            'n_estimators':[23, 24, 25, 26, 27, 28],
            'min_samples_leaf':[ 1, 2, 3]
            }
gscv = GridSearchCV(RandomForestClassifier(), param_grid, scoring='log_loss')
gscv.fit(trainX, trainY.values.ravel())
log_loss(testY, gscv.predict_proba(testX)) #0.58966148756025327
gscv.best_params_


######################## RF Submission Predictions ############################    
###############################################################################



training = pd.read_csv(data_path + 'training.csv', index_col=0)
labels = pd.read_csv(data_path + 'training_labels.csv', index_col=0)
prediction_data = pd.read_csv(data_path + 'prediction_data.csv', index_col=0)


clf=RandomForestClassifier(min_samples_leaf = 2,
                           n_estimators = 26)
clf.fit(training, labels)
probs=clf.predict_proba(prediction_data) 
probs_df = pd.DataFrame(probs, columns = ['predict_0', 'predict_1', 'predict_2'])
probs_df['id'] = prediction_data.index.values
probs_df = probs_df.set_index('id')
probs_df.to_csv(data_path + 'predictions.csv')
###############################################################################    
###############################################################################














