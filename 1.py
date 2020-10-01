import pandas as pd

# https://www.kaggle.com/matleonard/baseline-model

ks = pd.read_csv('./input/kickstarter-projects/ks-projects-201801.csv',
                 parse_dates=['deadline', 'launched'])
ks.head(10)

print("head---------------------------------------------------")
print(ks)

upd = pd.unique(ks.state)

print(upd)
ksg = ks.groupby('state')['ID'].count()
print("groupby---------------------------------------------------")
print(ksg)

# Drop live projects
ks = ks.query('state != "live"')
print("!live---------------------------------------------------")
print(ks)
# Add outcome column, "successful" == 1, others are 0
ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))
print("assign-successful---------------------------------------------------")
print(ks)

ks = ks.assign(hour=ks.launched.dt.hour,
               day=ks.launched.dt.day,
               month=ks.launched.dt.month,
               year=ks.launched.dt.year)

ks.head()
print("assign-hour---------------------------------------------------")
print(ks)

from sklearn.preprocessing import LabelEncoder

cat_features = ['category', 'currency', 'country']
encoder = LabelEncoder()

# Apply the label encoder to each column
encoded = ks[cat_features].apply(encoder.fit_transform)
encoded.head(10)
print("apply-fit_transform---------------------------------------------------")
print(encoded)
# Since ks and encoded have the same index and I can easily join them
data = ks[['goal', 'hour', 'day', 'month', 'year', 'outcome']].join(encoded)
data.head()
print("join-'goal', 'hour', 'day', 'month', 'year', 'outcome']---------------------------------------------------")
print(data)

valid_fraction = 0.1
valid_size = int(len(data) * valid_fraction)
print(valid_size)
train = data[:-2 * valid_size]
print(train)
valid = data[-2 * valid_size:-valid_size]
test = data[-valid_size:]
print()

print("Outcome-.4f}--------------------------------------------------")
for each in [train, valid, test]:
    print(f"Outcome fraction = {each.outcome.mean():.4f}")


import lightgbm as lgb

feature_cols = train.columns.drop('outcome')

dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])

print(dtrain)
print(dvalid)

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)
print(bst)

from sklearn import metrics
ypred = bst.predict(test[feature_cols])
print(ypred)
score = metrics.roc_auc_score(test['outcome'], ypred)

print(f"Test AUC score: {score}")









