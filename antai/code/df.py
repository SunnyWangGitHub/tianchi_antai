import time
import random
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import IPython
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from scipy.stats import chi
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
import xgboost as xgb
import catboost as cbt
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import rankdata
from sklearn.preprocessing import LabelEncoder
from itertools import product
from scipy.stats import entropy
from scipy.stats import rankdata
from deepforest import CascadeForestClassifier

train = pd.read_pickle('../data/fea_train_v2.pkl' )
test = pd.read_pickle('../data/fea_test_v2.pkl')



cols = [col for col in train.columns if col not in ['id','isDefault','index']]
print("fea shape:" ,len(cols))





nfold = 5
seeds = [19960218]
# nfold = 12
# seeds = [42,19960218,2021,7,999,233,88,13,9527,126]

print('nfold:',nfold)

oof = np.zeros(train.shape[0])

test['isDefault'] = 0
df_importance_list = []

print(train[cols].head(10))

val_aucs = []

test_sqrt = test.copy()

predictions = np.zeros((len(test),len(seeds)))

for seed in seeds:
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
    for i, (trn_idx, val_idx) in enumerate(skf.split(train, train['isDefault'])):
        print('--------------------- {} fold ---------------------'.format(i))
        t = time.time()
        trn_x, trn_y = train[cols].iloc[trn_idx].reset_index(drop=True), train['isDefault'].values[trn_idx]
        val_x, val_y = train[cols].iloc[val_idx].reset_index(drop=True), train['isDefault'].values[val_idx]

        df_model = CascadeForestClassifier(
            random_state=seed,
            n_jobs=-1,
            # use_predictor=True,
            # predictor='forest',
            # predictor='xgboost', #forest
            # predictor='lightgbm',
            # predictor_kwargs ={
            #     'boosting_type' : 'dart',
            #     'learning_rate' : 0.05,
            #     'n_estimators' : 6000,
            #     'num_leaves' : 31,
            #     'subsample' : 0.8,
            #     'metric' :'auc',
            # }
        )

        df_model.fit(
            trn_x.values, trn_y,
        )

        oof[val_idx] = df_model.predict_proba(val_x.values)[:, 1]
        test['isDefault'] += df_model.predict_proba(test[cols].values)[:, 1]/ skf.n_splits / len(seeds)
        test_sqrt['isDefault'] += df_model.predict_proba(test[cols])[:, 1]**0.5 / skf.n_splits / len(seeds)
        
        del df_model

    cv_auc = roc_auc_score(train['isDefault'], oof)
    val_aucs.append(cv_auc)
    print('\ncv_auc: ', cv_auc)
print(val_aucs, np.mean(val_aucs))
result_aucs = np.mean(val_aucs)
save_path = '../sub/df/'+str(result_aucs)
if not os.path.exists(save_path):
    os.mkdir(save_path)
print('save_path',save_path)


submit = pd.read_csv('../data/sample_submit.csv')
submit['id'] = test['id']
submit['isDefault'] = test['isDefault']

submit.to_csv(os.path.join(save_path,'baseline_df_auc_{}.csv'.format(result_aucs)), index = False)

##################################
submit['isDefault'] = test_sqrt['isDefault']

submit.to_csv(os.path.join(save_path,'baseline_df_auc_{}_sqrt.csv'.format(result_aucs)), index = False)
