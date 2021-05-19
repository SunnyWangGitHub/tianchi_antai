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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization


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


                        



train = pd.read_pickle('../data/fea_train_v2.pkl' )
test = pd.read_pickle('../data/fea_test_v2.pkl')




old_cols = [col for col in train.columns if col not in ['id','isDefault','index']]
new_feat_imp_df = pd.read_csv('../sub/xgb/0.7427262249215246/feat_xgb_baseline.csv')
new_feat_imp_df = new_feat_imp_df[new_feat_imp_df['importance']>=0.0005]

imp_fea = new_feat_imp_df['column'].tolist()
cols = [col for col in old_cols if col in imp_fea]

print("fea shape:" ,len(cols))



# def BO_xgb(x,y):
#     t1=time.clock()

#     def xgb_cv(max_depth,gamma,min_child_weight,max_delta_step,subsample,colsample_bytree):
#         paramt={'booster': 'gbtree',
#                 'max_depth': int(max_depth),
#                 'gamma': gamma,
#                 # 'eta': 0.1,
#                 'eta': 0.05,
#                 'objective': 'binary:logistic',
#                 'eval_metric': 'auc',
#                 'subsample': max(min(subsample, 1), 0),
#                 'colsample_bytree': max(min(colsample_bytree, 1), 0),
#                 'min_child_weight': min_child_weight,
#                 'max_delta_step': int(max_delta_step),
#                 'seed': 42,
#                 'n_estimators':1200,
#                 }
#         model=xgb.XGBClassifier(**paramt)
#         res = cross_val_score(model,x, y, scoring='roc_auc', cv=5).mean()
#         return res
#     cv_params ={'max_depth': (5, 12),
#                 'gamma': (0.001, 10.0),
#                 'min_child_weight': (0, 20),
#                 # 'max_delta_step': (0, 10),
#                 'subsample': (0.4, 1.0),
#                 'colsample_bytree': (0.4, 1.0)}
#     xgb_op = BayesianOptimization(xgb_cv,cv_params)
#     xgb_op.maximize(n_iter=10)
#     print(xgb_op.max)

#     t2=time.clock()
#     print('耗时：',(t2-t1))
#     return xgb_op.max


def BO_xgb(x,y):
    t1=time.clock()

    def xgb_cv(max_depth,gamma,min_child_weight,subsample,colsample_bytree):
        paramt={'booster': 'gbtree',
                'max_depth': int(max_depth),
                'gamma': gamma,
                # 'eta': 0.1,
                'eta': 0.05,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'subsample': max(min(subsample, 1), 0),
                'colsample_bytree': max(min(colsample_bytree, 1), 0),
                'min_child_weight': min_child_weight,
                'seed': 19960218,
                'n_estimators':1200,
                }
        model=xgb.XGBClassifier(**paramt)
        res = cross_val_score(model,x, y, scoring='roc_auc', cv=5).mean()
        return res
    cv_params ={'max_depth': (5, 12),
                'gamma': (0.001, 10.0),
                'min_child_weight': (0, 20),
                # 'max_delta_step': (0, 10),
                'subsample': (0.4, 1.0),
                'colsample_bytree': (0.4, 1.0)}
    xgb_op = BayesianOptimization(xgb_cv,cv_params)
    xgb_op.maximize(n_iter=20)
    print(xgb_op.max)

    t2=time.clock()
    print('耗时：',(t2-t1))
    return xgb_op.max

best_params=BO_xgb(train[cols],train['isDefault'])

print(best_params)

