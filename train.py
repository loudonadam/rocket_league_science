#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score, mean_squared_error
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import xgboost as xgb
import seaborn as sns
import pickle

#parameters
output_file = 'model.bin'

xgb_params = {
    'eta': 0.3, 
    'max_depth': 3,
    'min_child_weight': .7,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

print('Importing data.')
#data import and cleaning
data = pd.read_csv(r'C:\Users\loudo\replay_data_2.csv')
data = pd.DataFrame(data)

# new measure creation
data['is_ot'] = data['is_ot'].astype(int)
data['game_win'] = data['game_win'].astype(int)
car_id_map = {
    23: 'Octane',
    4284: 'Fennec',
    9089: 'Porsche 911 Turbo RLE',
    9088: 'Porsche 911 Turbo',
    9084: 'Nissan Silvia',
    10440: 'Nissan Fairlady Z',
    10896: 'BMW 1 Series',
    4780: 'Battle Bus',
    1533: 'Vulcan',
    8566: '',
    7948: '',
    10900: '',
    4906: '',
    3451: '',
    5488: ''
}
data['car'] = data['car_id'].map(car_id_map)
data['tm8_G'] = pd.DataFrame(data['tm8_name']=='GMARSH').astype(int)
data['tm8_J'] = pd.DataFrame(data['tm8_name']=='The Athlete 16').astype(int)
num_columns = ['game_win','is_ot','game_time','boost_per_min', 'boost_avg_amt',
       'big_stoln_per_min', 'big_clctd_per_min', 'sml_clctd_per_min',
       'pct_zero_boost', 'pct_full_boost', 'pct_0_25_boost', 'pct_25_50_boost',
       'pct_75_100_boost', 'avg_speed', 'pct_supersonic', 'pct_slow',
       'avg_powerslide_duration', 'powerslide_per_min', 'pct_ground',
       'pct_high_air', 'avg_dist_to_ball', 'avg_dist_to_mates',
       'pct_def_third', 'pct_off_third', 'pct_behind_ball', 'pct_most_back',
       'percent_closest_to_ball', 'demos_given_per_min', 'demos_taken_per_min', 'tm8_G', 'tm8_J']
cat_columns = ['tm_color','car']
cat_data = pd.get_dummies(data[cat_columns],columns=cat_columns,dtype=int)

#creating final dataframe
df = pd.concat([data[num_columns],cat_data],axis=1)

#splitting the data into train, test, and val making sure sizes make sense
df_full_train, df_test = train_test_split(df,test_size=.2, random_state=55)

#preserving outcome variable
y_train = (df_full_train.game_win == 1).astype('int').values
y_test = (df_test.game_win == 1).astype('int').values

#removing outcome variable from dfs
del df_full_train['game_win']
del df_test['game_win']


# preparing dicts
train_dict = df_full_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.transform(train_dict)
dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

print('Training the model.')
#training
dfulltrain = xgb.DMatrix(X_train, label=y_train,
                    feature_names=dv.get_feature_names_out().tolist())
dtest = xgb.DMatrix(X_test, feature_names=dv.get_feature_names_out().tolist())
model = xgb.train(xgb_params, dfulltrain, num_boost_round=60)
y_pred = model.predict(dtest)
auc = roc_auc_score(y_test, y_pred)

print('AUC: ', auc)

#save the model
with open(output_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)
