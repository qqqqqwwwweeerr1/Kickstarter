
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *

print("Setup Complete")
print(home_data)

print(home_data.columns)

#home_data = home_data.dropna(axis=0)
#y = home_data.SalePrice
y = home_data.SalePrice

# Create the list of features below
feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF",
                      "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

# select data corresponding to features in feature_names
# X = home_data[feature_names]
# home_data = home_data.dropna(axis=1, how='all')
# home_data = home_data.dropna(axis=0)
X = pd.DataFrame(data=home_data, columns=feature_names)








