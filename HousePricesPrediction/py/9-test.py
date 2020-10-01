import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

print(X_full)

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y,
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train_full.columns if
                X_train_full[cname].dtype in ['int64', 'float64']]


my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
print(my_cols)
print(X_train)


#
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
#
#
# numerical_transformer = SimpleImputer(strategy='constant')
#
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_cols),
#         ('cat', categorical_transformer, categorical_cols)
#     ])
# # model = RandomForestRegressor(n_estimators=100, random_state=0)
# #
# # clf = Pipeline(steps=[('preprocessor', preprocessor),
# #                       ('model', model)
# #                      ])
# # clf.fit(X_train, y_train)
# # preds = clf.predict(X_valid)
# # print('MAE:', mean_absolute_error(y_valid, preds))
#
#
# model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
# model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
# model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
# model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
# model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)
#
# models = [model_1, model_2, model_3, model_4, model_5]
#
# # Function for comparing different models
# def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
#     clf = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('model', model)
#                      ])
#
#     clf.fit(X_t, y_t)
#     preds = clf.predict(X_v)
#     return mean_absolute_error(y_v, preds)
#
# for i in range(0, len(models)):
#     mae = score_model(models[i])
#     print("Model %d MAE: %d" % (i+1, mae))
#
# # -----------------------------test model-------------------------------------------------
# clf = Pipeline(steps=[('preprocessor', preprocessor),
#                   ('model', model_3)
#                  ])
#
# clf.fit(X_train, y_train)
# preds_test = clf.predict(X_test)
# output = pd.DataFrame({'Id': X_test.index,
#                        'SalePrice': preds_test})
# # output.to_csv('../output/kaggle/working/submission.csv', index=False)
# output.to_csv('../output/submission.csv', index=False)
#




