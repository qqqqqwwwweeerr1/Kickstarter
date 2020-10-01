import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

print(X_full.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_full.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
print(missing_val_count_by_column)

many_null_feature = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
X_full = X_full.drop(many_null_feature, axis=1)

y = X_full.SalePrice
X = X_full.drop('SalePrice', axis=1)


print(X)
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
print(X_train)



X_train.to_csv('../output/X_train1.csv', index=False)
X_train = X_train.dropna()
X_valid = X_valid.dropna()
y_train = y_train.dropna()
y_valid = y_valid.dropna()

print(X_train)
print(X_valid)
print(y_train)
print(y_valid)



X_full.to_csv('../output/X_full.csv', index=False)
X_test_full.to_csv('../output/X_test_full.csv', index=False)
X_train.to_csv('../output/X_train2.csv', index=False)
X_valid.to_csv('../output/X_valid.csv', index=False)
y_train.to_csv('../output/y_train.csv', index=False)
y_valid.to_csv('../output/y_valid.csv', index=False)















