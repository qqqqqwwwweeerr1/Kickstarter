import numpy as np

import pandas as pd

df = pd.DataFrame([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [np.nan, np.nan, np.nan]],
               columns=['A', 'B', 'C'])


print(df)
print("---------------------------------------------------------")
a1 = df.agg(['sum', 'min'])
print(a1)
print("---------------------------------------------------------")

a2 = df.agg({'A' : ['sum', 'min'], 'B' : ['min', 'max']})
print(a2)
print("---------------------------------------------------------")

a3 = df.agg("mean", axis="columns")
print(a3)
print("---------------------------------------------------------")



