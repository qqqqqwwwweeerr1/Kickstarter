import math


# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(abs(y_pred[i]) + 1) - math.log(abs(y[i]) + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5


true = [0.559085, 0.372115, 0.218578, -0.033431, 0.063129, -0.010889, -0.004748, 0.187309, 0.285906, 0.14526, 0.154778,
        0.027247, -0.046379, -0.259953, -0.461212, -0.749929, -0.82602, -0.974357, -0.975688, -1.152518, -1.017474,
        -0.809009, -0.75802, -0.740017, -0.530228, -0.449722, -0.478659, -0.481355, -0.35855, -0.274692, -0.530591,
        -0.560529, -0.555234]
predict = [0.428452, 0.030436, 0.156903, 0.070028, -0.043729, -0.185889, -0.139339, 0.053388, 0.342791, 0.18994,
           0.243775, 0.13973, -0.196076, -0.124803, -0.490906, -0.605145, -0.814873, -0.799923, -0.851381, -0.886527,
           -1.02344, -0.987162, -1.003735, -0.892216, -0.242426, -0.332501, -0.323181, -0.395413, -0.304378, -0.311242,
           -0.365639, -0.554944, -0.806372]
my = [1000]
mypid = [1880]
from sklearn.metrics import mean_absolute_error

rmsle = rmsle(my, mypid)
# mae_2 = mean_absolute_error(true ,predict) # Your code here
print(" Root Mean Squared Logarithmic Error :", rmsle)
