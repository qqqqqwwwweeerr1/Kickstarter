import statsmodels.api as sm
import pandas as pd
import math

data = pd.read_csv("temp1.csv")
xx = ['Export_PCHy', 'M0_PCHy', 'PMI_PCHy', 'ws']
yy = 'yZt'

x = data[xx]
y = data[yy]
y2 = y.values

# y = [12.5, 16.7, 14.54, 13.01]
# x = [[23.51, 55.3, 15.3], [23.374, 56.1, 12.05], [23.4, 54.5, 16.7], [23.48, 56.1, 14.54]]
x2 = sm.add_constant(x.values)
reg = sm.OLS(y2, x2)
results = reg.fit()


y_pred = results.predict(x2)







def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(abs(y_pred[i]) + 1) - math.log(abs(y[i]) + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5

rmsle = rmsle(y, y_pred)
print(" Root Mean Squared Logarithmic Error :", rmsle)