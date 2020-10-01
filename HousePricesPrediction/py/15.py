import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

####  读入数据，划分训练集和测试集
data=pd.read_csv('../input/housing.csv')
housing = data.drop("median_house_value", axis=1)
housing_labels = data["median_house_value"].copy()

L=int(0.8*len(housing))
train_x=housing.ix[0:L,:]
train_y=housing_labels.ix[0:L]
test_x=housing.ix[L:,:]
test_y=housing_labels.ix[L:]

cat_attribute=['ocean_proximity']
housing_num=housing.drop(cat_attribute,axis=1)
num_attribute=list(housing_num)



# X_full = pd.read_csv('../input/train.csv', index_col='Id')
# X_test_full = pd.read_csv('../input/test.csv', index_col='Id')
#
# X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
# y = X_full.SalePrice
# X_full.drop(['SalePrice'], axis=1, inplace=True)
# X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y,
#                                                                 train_size=0.8, test_size=0.2,
#                                                                 random_state=0)

start=time.clock()
####   构建　pipeline
from sklearn.preprocessing import Imputer,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer

##构建新特征需要的类
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributeAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room=add_bedrooms_per_room
    def fit(self, X, y=None):
        return(self)
    def transform(self,X):
        '''X is an array'''
        rooms_per_household=X[:,rooms_ix]/X[:,household_ix]
        population_per_household=X[:,population_ix]/X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room=X[:,bedrooms_ix]/X[:,rooms_ix]
            return(np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room])
        else:
            return(np.c_[X,rooms_per_household,population_per_household])
##根据列名选取出该列的类
class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names=attribute_names
    def fit(self, X, y=None):
        return(self)
    def transform(self,X):
        '''X is a DataFrame'''
        return(X[self.attribute_names].values)
##对分类属性进行onhehot编码
class MyLabelBinarizer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.encoder = LabelBinarizer()
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

##构造数值型特征处理的pipeline
#流水线中的所有组件，除了最后一个，必须是transformer；最后一个估计器可以是任何类型
num_pipeline=Pipeline([
    ('selector',DataFrameSelector(num_attribute)),
    ('imputer',Imputer(strategy='median')),
    ('attribute_adder',CombinedAttributeAdder()),
    ('std_scaler',StandardScaler())
])

##构造分类型特征处理的pipeline
cat_pipeline=Pipeline([
    ('selector',DataFrameSelector(cat_attribute)),
    ('label_binarizer',MyLabelBinarizer())
])

##将上述pipeline用FeatureUnion组合，将两组特征组合起来
full_pipeline=FeatureUnion(transformer_list=[
    ('num_pipeline',num_pipeline),
    ('cat_pipeline',cat_pipeline)
])

##特征选择类：用随机森林计算特征的重要程度，返回最高的Ｋ个特征
class TopFeaturesSelector(BaseEstimator,TransformerMixin):
    def __init__(self,feature_importance_k=5):
        self.top_k_attributes=None
        self.k=feature_importance_k
    def fit(self,X,y):
        reg=RandomForestRegressor()
        reg.fit(X,y)
        feature_importance=reg.feature_importances_
        top_k_attributes=np.argsort(-feature_importance)[0:self.k]
        self.top_k_attributes=top_k_attributes
        return(self)
    def transform(self, X,**fit_params):
        return(X[:,self.top_k_attributes])
##数据预处理以及选取最重要的Ｋ个特征的pipeline
prepare_and_top_feature_pipeline=Pipeline([
    ('full_pipeline',full_pipeline),
    ('feature_selector',TopFeaturesSelector(feature_importance_k=5))
])

##用GridSearchCV计算最随机森林最优的参数
train_x_=full_pipeline.fit_transform(train_x)
# Tree model,select best parameter with GridSearchCV
param_grid={
    'n_estimators':[10,50],
    'max_depth':[8,10]
}
reg=RandomForestRegressor()
grid_search=GridSearchCV(reg,param_grid=param_grid,cv=5)
grid_search.fit(train_x_,train_y)

##构造最终的数据处理和预测pipeline
prepare_and_predict_pipeline=Pipeline([
    ('prepare',prepare_and_top_feature_pipeline),
    ('random_forest',RandomForestRegressor(**grid_search.best_params_))
])

####   对上述总的pipeline用GridSearchCV选取最好的pipeline参数
param_grid2={'prepare__feature_selector__feature_importance_k':[1,3,5,10],
             'prepare__full_pipeline__num_pipeline__imputer__strategy': ['mean', 'median', 'most_frequent']}
grid_search2=GridSearchCV(prepare_and_predict_pipeline,param_grid=param_grid2,cv=2,
                          scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
grid_search2.fit(train_x,train_y)
pred=grid_search2.predict(test_x)

end=time.clock()
print('RMSE on test set={}'.format(np.sqrt(mean_squared_error(test_y,pred))))
print('cost time={}'.format(end-start))
print('grid_search2.best_params_=\n',grid_search2.best_params_)




output = pd.DataFrame({'Id': test_x.index,
                       'SalePrice': pred})
# output.to_csv('../output/kaggle/working/submission.csv', index=False)
output.to_csv('../output/submission-housing.csv', index=False)