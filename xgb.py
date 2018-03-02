'''
    @data      --   Telstar's Kaggle Competition
    @author    --   Ned Hulseman
    @date      --   2/14/2017
    @Purpose   --   Interworks data assesment
    
    
    @link https://www.kaggle.com/c/telstra-recruiting-network/data

    Script Objective:
        Predict service faults and create some predictive models using
        xgboost
'''

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import GridSearchCV


data_path = r'C:\Users\nedhu\Desktop\Interworks Assesment\Data\\'
trainX  =  pd.read_csv(data_path + 'X_train.csv', index_col=0)
testX   =  pd.read_csv(data_path + 'X_test.csv', index_col=0)
trainY  =  pd.read_csv(data_path + 'y_train.csv', index_col=0)
testY   =  pd.read_csv(data_path + 'y_test.csv', index_col=0)



    
################################ Boosting #####################################    
###############################################################################

'''
    We will start off by creating a baseline xgb model and then tune paramters
'''
clf=XGBClassifier(learning_rate=0.08, 
                   max_depth=10, 
                   objective='binary:logistic', 
                   nthread=3, 
                   gamma=0.2, 
                   subsample=0.9,
                   n_estimators=100,
                   )
clf.fit(trainX, trainY)
probs=clf.predict_proba(testX) 
preds=clf.predict(testX) 
confusion_matrix(testY, preds)
log_loss(testY, probs) #0.53162637402662305

    
 '''
 {'learning_rate': 0.3,
 'max_depth': 3,
 'n_estimators': 100,
 'objective': 'binary:logistic',
 'subsample': 0.9}
'''   
param_grid={
        'n_estimators':[50, 80, 100, 130],
        'learning_rate':[.1, .3, .5],
        'objective':['binary:logistic'],
        'subsample':[.9],
        'max_depth':[2,  3]
        }
gscv = GridSearchCV(XGBClassifier(), param_grid, scoring='log_loss')
gscv.fit(trainX, trainY.values.ravel())
log_loss(testY, gscv.predict_proba(testX)) #0.52972168295331945
gscv.best_params_
    

clf=XGBClassifier(learning_rate=.5, 
                   max_depth=10, 
                   objective='binary:logistic', 
                   nthread=3, 
                   gamma=0.2, 
                   subsample=0.9,
                   n_estimators=400,
                   )
clf.fit(trainX, trainY)
probs=clf.predict_proba(testX) 
preds=clf.predict(testX) 
confusion_matrix(testY, preds)
log_loss(testY, probs) #0.53162637402662305
###############################################################################    
###############################################################################    


######################## XGB Submission Predictions ###########################    
###############################################################################


training = pd.read_csv(data_path + 'training.csv', index_col=0)
labels = pd.read_csv(data_path + 'training_labels.csv', index_col=0)
prediction_data = pd.read_csv(data_path + 'prediction_data.csv', index_col=0)


clf=XGBClassifier(learning_rate=.5, 
                   max_depth=10, 
                   objective='binary:logistic', 
                   nthread=3, 
                   gamma=0.2, 
                   subsample=0.9,
                   n_estimators=400,
                   )
clf.fit(training, labels)
probs=clf.predict_proba(prediction_data) 
probs_df = pd.DataFrame(probs, columns = ['predict_0', 'predict_1', 'predict_2'])
probs_df['id'] = prediction_data.index.values
probs_df = probs_df.set_index('id')
probs_df.to_csv(data_path + 'predictions.csv')
###############################################################################    
###############################################################################  













    
    
    
