'''
    @data      --   Telstar's Kaggle Competition
    @author    --   Ned Hulseman
    @date      --   2/14/2017
    @Purpose   --   Interworks data assesment
    
    
    @link https://www.kaggle.com/c/telstra-recruiting-network/data
    
    Script Objective:
        Using this script we want to combine the datasets to create a training
        and test script that can be used to model on
        
    steps:
        1) Conduct a brief exploration in the various datasets to determine 
           how to merge data together
        2) format each dataset individually so we can do a one-to-many merge
        3) merge all datasets into one train and one prediction dataset
        4) split the training dataset into training and test data, then output 
           all datasets
           
    script output:
        This script will deliver 5 datasets:
            1) X_train
            2) y_train
            3) X_test
            4) y_test
            5) prediction_data -- used for final Kaggle submission
    
'''


import pandas as pd
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split



data_path = r'C:\Users\nedhu\Desktop\Interworks Assesment\Data\\'

event_type    =  pd.read_csv(data_path + 'event_type.csv')
log_feature   =  pd.read_csv(data_path + 'log_feature.csv')
resource_type =  pd.read_csv(data_path + 'resource_type.csv')
severity_type =  pd.read_csv(data_path + 'severity_type.csv')
test          =  pd.read_csv(data_path + 'test.csv')
train         =  pd.read_csv(data_path + 'train.csv')





######################## Brief Document Exploration ###########################
###############################################################################
dataframes = [event_type, log_feature, resource_type, severity_type, test, train]
for df in dataframes:
    print(df.head())
    #print(df.shape)
    print(' ')
    

'''
    event_type (311170, 2)
    log_feature (58671, 3)
    resource_type (21076, 2)
    severity_type (18552, 2)
    test (11171, 2)
    train (7381, 3)
'''

###############################################################################
###############################################################################







######################### Format data for merge ###############################
###############################################################################

'''
    GOAL
        @trian & @test have 18,552 unique observations collectively.
        We want to make bonus data sets have a unique id for each row
        so that they have 18,552 rows.
    
    STEPS TO MERGE
        @train strip off fault_severity and onehot location
        @test onehot location
        @severity_type is 1 to 1 but we need to turn location into dummy variables
        @log_feature is 1 to 1 after pivoting
        @event_type 1)add col of 1's 2)pivot 
        @resource_type 1)add col of 1's 2)pivot 
    
'''
#train
trainY         =  train[['id', 'fault_severity']]
trainY          =  trainY.set_index('id')
trainX         =  train.drop(['fault_severity'], axis=1)
trainX         =  trainX.set_index('id')
trainX_encoded =  pd.get_dummies(data=trainX)
trainX_encoded = trainX_encoded.astype(np.float64)
#test
testX         =  test.set_index('id')
testX_encoded =  pd.get_dummies(data=testX)


#severity_type 
severitytype_encoded = pd.get_dummies(data=severity_type, columns = ['severity_type'])
severitytype_encoded = severitytype_encoded.set_index('id')

# log_feature
#log_feature['volume'] = np.log(log_feature['volume'])
log_feature           =  log_feature.set_index('id')
logfeature_pivotted   =  log_feature.pivot(columns='log_feature', 
                         values='volume')
logfeature_pivotted   =  logfeature_pivotted.fillna(0)

# event_type
event_type            =  event_type.set_index('id')
event_type['values']  =  1
eventtype_pivotted    =  event_type.pivot(columns='event_type', 
                         values='values')
eventtype_pivotted    =  eventtype_pivotted.fillna(0)

# resource_type
resource_type= resource_type.set_index('id')
resource_type['values']  =  1
resourcetype_pivotted    =  resource_type.pivot(columns='resource_type', values='values')
resourcetype_pivotted    =  resourcetype_pivotted.fillna(0)

train_mergable_dataframes =  [trainX_encoded, severitytype_encoded, logfeature_pivotted, 
                              eventtype_pivotted, resourcetype_pivotted]
test_mergable_dataframes  =  [testX_encoded, severitytype_encoded, logfeature_pivotted, 
                             eventtype_pivotted, resourcetype_pivotted]


###############################################################################
###############################################################################




################# Merge datasets to create training and test ##################
###############################################################################


training_merge  =  reduce(lambda left, right: pd.merge(left, right, how='left',
                   left_index=True, right_index=True), train_mergable_dataframes)

prediction_data =  reduce(lambda left, right: pd.merge(left, right, how='left', 
                   left_index=True, right_index=True), test_mergable_dataframes)

training_merge = training_merge.astype(np.float64)
prediction_data = prediction_data.astype(np.float64)

#training_merge_col_headers = list(training_merge.columns.values)
#prediction_data = prediction_data[training_merge_col_headers]

'''
    We have a problem here.. because the location variables are so sparse
    The training_merge df contains levels of location that are not available in
    the prediction_data and vice versa. In order to fix this we will...
        1) remove any levels that prediction_data has that aren't in the 
           training data
        2) append all training_merge columns not in prediction_data to 
            prediction_data.. All of these columns will be instantiated to
            be zero
'''
#get lists of variables not in training_merge and prediction_data
notInPred=[x for x in training_merge.columns.values if not x in prediction_data.columns.values]
notInTrain=[x for x in prediction_data.columns.values if not x in  training_merge.columns.values]


dropped_levels = prediction_data.drop(notInTrain, axis=1) #drop columns not in training
notInPred_df = pd.DataFrame(np.zeros(shape=(len(prediction_data),len(notInPred))),
                         columns = notInPred) #create all-zero df from notInPred
notInPred_df.index = prediction_data.index.values
prediction_data = pd.merge(dropped_levels, notInPred_df, how='inner', 
                           left_index=True, right_index=True) 
prediction_data = prediction_data[training_merge.columns.values]
'''
    Lets confirm that observations from each dataset match
    We will do this by tracking a specific id and check that it matches the original data
'''
# =============================================================================
# train.head() #lets investigate id=14121
# train[train['id'] == 14121]
# training_merge.loc[14121]['location_location 118'] # value is 1 which is correct
# severity_type[severity_type['id'] == 14121]
# training_merge.loc[14121]['severity_type_severity_type 2'] # value is 1 which is correct
# log_feature.loc[14121]
# training_merge.loc[14121][['feature 312', 'feature 232']] # contains 2 features that = 19
# event_type.loc[14121]
# training_merge.loc[14121][['event_type 34', 'event_type 35']] # contains 2 features that = 1
# resource_type.loc[14121]
# training_merge.loc[14121]['resource_type 2'] # contains 1 features that = 1
# 
# 
# pd.set_option("display.max_rows", 1000)
# training_merge.loc[14121] #Everything looks great
# =============================================================================
###############################################################################
###############################################################################





#################### create training, test, pred and output ####################
###############################################################################

X_train, X_test, y_train, y_test = train_test_split(training_merge, trainY, train_size=.8, random_state=314)

training_merge.to_csv(data_path + 'training.csv')
trainY.to_csv(data_path + 'training_labels.csv')
X_train.to_csv(data_path + 'X_train.csv')
X_test.to_csv(data_path + 'X_test.csv')
y_train.to_csv(data_path + 'y_train.csv')
y_test.to_csv(data_path + 'y_test.csv')
prediction_data.to_csv(data_path + 'prediction_data.csv')

###############################################################################
###############################################################################







