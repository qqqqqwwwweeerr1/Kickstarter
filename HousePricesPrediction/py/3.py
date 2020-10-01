# Set up code checking
# import os
# if not os.path.exists("../input/train.csv"):
#     os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")
#     os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")
# from learntools.core import binder
# binder.bind(globals())
# from learntools.ml_intermediate.ex1 import *
# print("Setup Complete")


import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



X_train = X_full
X_valid = X_test_full
print(X_train)
print(X_valid)
y_train = X_train['SalePrice']
y_valid = X_train['SalePrice']

X_train = X_train.drop('SalePrice', axis=1)

missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
print('--------------------------------30----------------------------')
print(missing_val_count_by_column)


# X_full = X_full.drop('Alley', axis=0)
# '''
# 去除空列
# '''
# # Get names of columns with missing values
# cols_with_missing = [col for col in X_full.columns
#                      if X_full[col].isnull().any()]
# y_valid = X_full.SalePrice
# # Drop columns in training and validation data
# reduced_X_train = X_full.drop(cols_with_missing, axis=1)
# reduced_X_valid = X_test_full.drop(cols_with_missing, axis=1)
#
# print("MAE from Approach 1 (Drop columns with missing values):")
# # print(score_dataset(reduced_X_train, reduced_X_valid, X_full, y_valid))

'''
去除空行V4
'''
print("----------------------------------52---------------------------------------------")
X_train = X_train.drop('Alley', axis=1)
X_valid = X_valid.drop('Alley', axis=1)


# print(X_train)

reduced_X_train = X_train.dropna()
reduced_X_valid = X_valid.dropna()

print(type(reduced_X_train))
reduced_X_train.to_csv('../output/reduced_X_train.csv', index=False)
reduced_X_valid.to_csv('../output/reduced_X_valid.csv', index=False)
X_train.to_csv('../output/X_train.csv', index=False)
X_valid.to_csv('../output/X_valid.csv', index=False)


#将所有numberic的列加入模型

# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)


drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
print("MAE from Approach 1 (Drop categorical variables):")
# print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))










#
# '''
# 检测object类型列
# '''
# # Get list of categorical variables
# s = (X_full.dtypes == 'object')
# object_cols = list(s[s].index)
#
# print("Categorical variables:")
# print(object_cols)
#
#
# drop_X_train = X_train.select_dtypes(exclude=['object'])
# drop_X_valid = X_valid.select_dtypes(exclude=['object'])
#
# print("MAE from Approach 1 (Drop categorical variables):")
# print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

# Obtain target and predictors
y = X_full.SalePrice

# features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr',
#             'TotRmsAbvGrd', 'GrLivArea', 'YrSold']


features = drop_X_train.columns.values

print('features:---------------------------------------------')
print(features)
print('features:---------------------------------------------')

X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

my_dict = {}

from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    my_dict[models[i]] = mae
    print("Model %d MAE: %d" % (i+1, mae))



# Fill in the best model

best_model = min(my_dict, key=my_dict.get)
print("best_model : "+str(best_model))




# Define a model
my_model = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0) # Your code here






# Fit the model to the training data
my_model.fit(X, y)

# Generate test predictions
preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
# output.to_csv('../output/kaggle/working/submission.csv', index=False)
output.to_csv('./submission.csv', index=False)















