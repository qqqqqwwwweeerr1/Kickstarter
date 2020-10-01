import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')


# Number of missing values in each column of training data
missing_val_count_by_column = (X_full.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
print(missing_val_count_by_column)

many_null_feature = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
X_full = X_full.drop(many_null_feature, axis=1)

y = X_full.SalePrice
X = X_full.drop('SalePrice', axis=1)


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
X_train.to_csv('../output/X_train1.csv', index=False)
X_train = X_train.dropna()
X_valid = X_valid.dropna()
y_train = y_train.dropna()
y_valid = y_valid.dropna()


X_full.to_csv('../output/X_full.csv', index=False)
X_test_full.to_csv('../output/X_test_full.csv', index=False)
X_train.to_csv('../output/X_train2.csv', index=False)
X_valid.to_csv('../output/X_valid.csv', index=False)
y_train.to_csv('../output/y_train.csv', index=False)
y_valid.to_csv('../output/y_valid.csv', index=False)

# sklearn.impute.SimpleImputer
# this can imput missing values

drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
print(drop_X_train)
print(drop_X_valid)

# All categorical columns
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))


print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)



from sklearn.preprocessing import OneHotEncoder

# Use as many lines of code as you need!
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))
# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index
# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
print(OH_X_train)