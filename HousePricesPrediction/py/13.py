import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

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

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])



from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
# Define the model
# model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
my_model_2 = XGBRegressor(n_estimators=30000, learning_rate=0.0067, n_jobs=4)  # Your code here

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', my_model_2)
                      ])
clf.fit(X_train, y_train)  # Your code here


from sklearn.metrics import mean_absolute_error

# Get predictions
predictions_t = clf.predict(X_test) # Your code here
predictions_v = clf.predict(X_valid) # Your code here


# Calculate MAE
mae_2 = mean_absolute_error(predictions_v ,y_valid) # Your code here

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_2)

# Mean Absolute Error: 16499.379106913526

print(X_test.index)
print('fffffffffffffffffffffffffffff')
print(predictions_t)
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': predictions_t})
# output.to_csv('../output/kaggle/working/submission.csv', index=False)
output.to_csv('../output/submission.csv', index=False)