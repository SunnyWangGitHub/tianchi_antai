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
import optuna
from sklearn.model_selection import train_test_split

train = pd.read_pickle('../data/fea_train_v3.pkl' )
test = pd.read_pickle('../data/fea_test_v3.pkl')
y = train['isDefault']
train.drop(['isDefault'], axis=1, inplace=True)
x = train


def objective(trial,data=x,target=y):
    
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2,random_state=42)
    param = {
        'metric': 'auc', 
        'random_state': 42,
        'n_estimators': 20000,
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
#         'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
#         'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
#         'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
#         'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02,0.03,0.04,0.05]),
        'learning_rate':0.01,
        # 'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
        'max_depth': trial.suggest_int('max_depth', 1, 100),
#         'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
#         'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
#         'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
        
        # "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        # "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        
    }
    model = lgb.LGBMClassifier(**param)  
    model.fit(train_x,train_y,eval_set=[(test_x,test_y)],early_stopping_rounds=100,verbose=False)
    preds = model.predict_proba(test_x)[:, 1]
    
    cv_auc = roc_auc_score(test_y, preds)    
    return cv_auc    
    

old_cols = [col for col in train.columns if col not in ['id','isDefault','index']]

# 0.0002 0.742758135028219
# 0.00025 0.7426794660604815

for t in [8]:
    print('!!!!!!!!!!!!!!!!!!!!!, ',t)
    new_feat_imp_df = pd.read_csv('../sub/lgb/0.7414399621379929/feat_lgb_baseline.csv')
    # new_feat_imp_df = pd.read_csv('0.7414399621379929_feat_lgb_baseline.csv')
    new_feat_imp_df = new_feat_imp_df[new_feat_imp_df['importance']>t]
    imp_fea = new_feat_imp_df['column'].tolist()
    cols = [col for col in old_cols if col in imp_fea]
    print("fea shape:" ,len(cols))
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)