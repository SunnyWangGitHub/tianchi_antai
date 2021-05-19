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

# class MeanEncoder:
#     def __init__(self, categorical_features, n_splits=5, target_type='classification', prior_weight_func=None):
#         """
#         :param categorical_features: list of str, the name of the categorical columns to encode

#         :param n_splits: the number of splits used in mean encoding

#         :param target_type: str, 'regression' or 'classification'

#         :param prior_weight_func:
#         a function that takes in the number of observations, and outputs prior weight
#         when a dict is passed, the default exponential decay function will be used:
#         k: the number of observations needed for the posterior to be weighted equally as the prior
#         f: larger f --> smaller slope
#         """

#         self.categorical_features = categorical_features
#         self.n_splits = n_splits
#         self.learned_stats = {}

#         if target_type == 'classification':
#             self.target_type = target_type
#             self.target_values = []
#         else:
#             self.target_type = 'regression'
#             self.target_values = None

#         if isinstance(prior_weight_func, dict):
#             self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
#         elif callable(prior_weight_func):
#             self.prior_weight_func = prior_weight_func
#         else:
#             self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

#     @staticmethod
#     def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
#         X_train = X_train[[variable]].copy()
#         X_test = X_test[[variable]].copy()

#         if target is not None:
#             nf_name = '{}_pred_{}'.format(variable, target)
#             X_train['pred_temp'] = (y_train == target).astype(int)  # classification
#         else:
#             nf_name = '{}_pred'.format(variable)
#             X_train['pred_temp'] = y_train  # regression
#         prior = X_train['pred_temp'].mean()

#         col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({'mean': 'mean', 'beta': 'size'})
#         col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
#         col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
#         col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)

#         nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
#         nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

#         return nf_train, nf_test, prior, col_avg_y

#     def fit_transform(self, X, y):
#         """
#         :param X: pandas DataFrame, n_samples * n_features
#         :param y: pandas Series or numpy array, n_samples
#         :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
#         """
#         X_new = X.copy()
#         if self.target_type == 'classification':
#             skf = StratifiedKFold(self.n_splits)
#         else:
#             skf = KFold(self.n_splits)

#         if self.target_type == 'classification':
#             self.target_values = sorted(set(y))
#             self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
#                                   product(self.categorical_features, self.target_values)}
#             for variable, target in product(self.categorical_features, self.target_values):
#                 nf_name = '{}_pred_{}'.format(variable, target)
#                 X_new.loc[:, nf_name] = np.nan
#                 for large_ind, small_ind in skf.split(y, y):
#                     nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
#                         X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target,
#                         self.prior_weight_func)
#                     X_new.iloc[small_ind, -1] = nf_small
#                     self.learned_stats[nf_name].append((prior, col_avg_y))
#         else:
#             self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
#             for variable in self.categorical_features:
#                 nf_name = '{}_pred'.format(variable)
#                 X_new.loc[:, nf_name] = np.nan
#                 for large_ind, small_ind in skf.split(y, y):
#                     nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
#                         X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None,
#                         self.prior_weight_func)
#                     X_new.iloc[small_ind, -1] = nf_small
#                     self.learned_stats[nf_name].append((prior, col_avg_y))
#         return X_new

#     def transform(self, X):
#         """
#         :param X: pandas DataFrame, n_samples * n_features
#         :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
#         """
#         X_new = X.copy()

#         if self.target_type == 'classification':
#             for variable, target in product(self.categorical_features, self.target_values):
#                 nf_name = '{}_pred_{}'.format(variable, target)
#                 X_new[nf_name] = 0
#                 for prior, col_avg_y in self.learned_stats[nf_name]:
#                     X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
#                         nf_name]
#                 X_new[nf_name] /= self.n_splits
#         else:
#             for variable in self.categorical_features:
#                 nf_name = '{}_pred'.format(variable)
#                 X_new[nf_name] = 0
#                 for prior, col_avg_y in self.learned_stats[nf_name]:
#                     X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
#                         nf_name]
#                 X_new[nf_name] /= self.n_splits

#         return X_new

# def subGradeTrans(x):
#     subGradeTrans_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
#     result = subGradeTrans_dict[x[0]]
#     result = result * 5 + int(x[1])
#     return result

# train_data = pd.read_csv('../data/train.csv')
# test_data = pd.read_csv('../data/testB.csv')



# df = pd.concat([train_data,test_data],axis=0).reset_index()
# df.drop(['policyCode'],axis  = 1,inplace = True)
# del train_data
# del test_data
# gc.collect()


# ##########################################################################################
# df['employmentLength'] = df['employmentLength'].fillna('缺失')

# df['employmentLength'] = df['employmentLength'].map({'缺失':-1,'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,
#                                                            '6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10})

# df['pubRecBankruptcies'] = df['pubRecBankruptcies'].fillna(0)

# df['dti'] = df['dti'].fillna(df['dti'].median())

# df['revolUtil'] = df['revolUtil'].fillna(round(df['revolUtil'].mean(),2))

# for fea in ['n11','n12']:
#     df[fea] = df[fea].fillna(df[fea].mode()[0])

# temp_list = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','n13','n14']
# for fea in temp_list:
#     df[fea] = df[fea].fillna(df[fea].median())



# df['grade'] = df['grade'].map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6})

# temp = df['subGrade'].value_counts()
# #排序编号   
# temp = pd.DataFrame(temp.sort_index(ascending = True))
# #重置index，取index列，转换为字典
# temp = temp.reset_index()['index'].to_dict()
# #将字典中的key：value翻转
# temp = {v:k for k,v in temp.items()}
# # 应用字典到subGrade衍生totalGrade
# df['totalGrade'] = df['subGrade'].map(temp)

# df['subGrade'] = df['subGrade'].apply(lambda x: subGradeTrans(x))

# df['issueDate'] = pd.to_datetime(df['issueDate'],format='%Y-%m-%d')
# #提取时间多尺度 年度+月度
# df['issueDate_year'] = df['issueDate'].dt.year
# df['issueDate_month'] = df['issueDate'].dt.month
# df['issueDate_day'] = df['issueDate'].dt.day
# df['issueDate_weekday'] = df['issueDate'].dt.weekday

# #提取diff ,转换为时间差
# df['issueDate_diff'] = (df['issueDate']-df['issueDate'].min()).dt.days
# df[['issueDate','issueDate_year','issueDate_month','issueDate_diff']]


# df['earliesCreditLine'] = pd.to_datetime(df['earliesCreditLine'])
# #提取时间多尺度 年度+月度
# df['earliesCreditLine_year'] = df['earliesCreditLine'].dt.year
# df['earliesCreditLine_month'] = df['earliesCreditLine'].dt.month

# df['earliesCreditLine_diff'] = (df['earliesCreditLine']-df['earliesCreditLine'].min()).dt.days/365

# df['issue_CreditLine_diff'] = (df['issueDate']-df['earliesCreditLine']).dt.days/365


# df.drop(['issueDate','earliesCreditLine'],axis = 1,inplace = True)


# #annualIncome  /   loanAmnt
# df['loanincome_ratio']= df['annualIncome']/df['loanAmnt']

# # del

# df['avg_income'] = df['annualIncome'] / df['employmentLength']
# df['total_income'] = df['annualIncome'] * df['employmentLength']

# df['avg_loanAmnt'] = df['loanAmnt'] / df['term']
# df['mean_interestRate'] = df['interestRate'] / df['term']
# df['all_installment'] = df['installment'] * df['term']

# df['rest_money_rate'] = df['avg_loanAmnt'] / (df['annualIncome'] + 0.1)  # 287个收入为0
# df['rest_money'] = df['annualIncome'] - df['avg_loanAmnt']

# df['closeAcc'] = df['totalAcc'] - df['openAcc']
# df['openAcc_totalAcc_rate'] = df['openAcc'] / df['totalAcc']  # add

# df['ficoRange_mean'] = (df['ficoRangeHigh'] + df['ficoRangeLow']) / 2
# del df['ficoRangeHigh'], df['ficoRangeLow']

# df['rest_pubRec'] = df['pubRec'] - df['pubRecBankruptcies']

# df['rest_Revol'] = df['loanAmnt'] - df['revolBal']

# df['dis_time'] = df['issueDate_year'] - (2021 - df['earliesCreditLine_year'])




# ########################################################################
# # for col in ['employmentTitle', 'grade', 'subGrade', 'regionCode']:
# #     df['{}_count'.format(col)] = df.groupby([col])['id'].transform('count')

# cate_features = ['applicationType', 'employmentLength', 'employmentTitle', 'grade', 'homeOwnership', 'initialListStatus',
#                  'postCode', 'purpose', 'regionCode', 'subGrade', 'title', 'verificationStatus']
# num_features = [
#     'annualIncome', 'delinquency_2years', 'dti', 'employmentLength','installment', 'interestRate', 'loanAmnt', 'openAcc', 'pubRec', 'pubRecBankruptcies','revolBal', 'revolUtil', 'subGrade', 'term', 'totalAcc',
#     'ficoRange_mean',
#     'loanincome_ratio','avg_income','avg_loanAmnt','mean_interestRate','all_installment','rest_money_rate','rest_money','closeAcc','openAcc_totalAcc_rate','rest_pubRec','rest_Revol',
#     'issueDate_diff','earliesCreditLine_diff','issue_CreditLine_diff'
# ]

# # gen_features = []

# for f in tqdm(cate_features):
#     # df['{}_cnt'.format(f)] = df.groupby([f])[
#     #     f].transform('count') # # log_xgb_v4_6.txt 0.7408332429082879 online:0.7427
#     df[f + '_count'] = df[f].map(df[f].value_counts())


# for f1 in tqdm(cate_features):
#     for f2 in cate_features:
#         if f1 != f2:
#             f_pair = [f1, f2]
#             df['_'.join(f_pair) + '_count'] = df.groupby(f_pair)['id'].transform('count')
#             ### n unique、熵
#             # df = df.merge(df.groupby(f_pair[0], as_index=False)[f_pair[1]].agg({
#             #     '{}_{}_nunique'.format(f_pair[0], f_pair[1]): 'nunique',
#             #     # '{}_{}_ent'.format(f_pair[0], f_pair[1]): lambda x: entropy(x.value_counts() / x.shape[0])
#             # }), on=f_pair[0], how='left')
#             # df = df.merge(df.groupby(f_pair[1], as_index=False)[f_pair[0]].agg({
#             #     '{}_{}_nunique'.format(f_pair[1], f_pair[0]): 'nunique',
#             #     # '{}_{}_ent'.format(f_pair[1], f_pair[0]): lambda x: entropy(x.value_counts() / x.shape[0])
#             # }), on=f_pair[1], how='left')
#             ### 比例偏好
#             df['{}_in_{}_prop'.format(f_pair[0], f_pair[1])] = df['_'.join(f_pair) + '_count'] / df[f_pair[1] + '_count']
#             df['{}_in_{}_prop'.format(f_pair[1], f_pair[0])] = df['_'.join(f_pair) + '_count'] / df[f_pair[0] + '_count']
# #log_xgb_v4_7.txt 0.7418411655833838 0.7441

# ########################################################################
# for f1 in tqdm(cate_features):
#     g = df.groupby(f1, as_index=False)
#     for f2 in num_features:
#         feat = g[f2].agg({
#             '{}_{}_max'.format(f1, f2): 'max', 
#             '{}_{}_min'.format(f1, f2): 'min',
#             '{}_{}_mean'.format(f1, f2): 'mean',
#             '{}_{}_median'.format(f1, f2): 'median',
#         })
#         df = df.merge(feat, on=f1, how='left')



# ########################################################################
# n_feat = ['n0', 'n1', 'n2','n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', ]
# nameList = ['max', 'min', 'mean', 'median']
# statList = ['max', 'min', 'mean', 'median']
# for i in range(len(nameList)):
#     df['n_feat_{}'.format(nameList[i])] = df[n_feat].agg(statList[i], axis=1)
# print('n特征处理后：', df.shape)



# ########################################################################
# train = df[df['isDefault'].isna() == False].reset_index(drop=True)
# test = df[df['isDefault'].isna() == True].reset_index(drop=True)
# train_label = train['isDefault']

# # 0.7427262249215246

# # + totalGrade？


# ##############################################################################
# # class_list = ['postCode', 'regionCode','employmentTitle','title']
# ##############################################################################
# class_list = ['postCode', 'regionCode','employmentTitle','title','purpose','homeOwnership','grade', 'subGrade']
# # class_list = ['applicationType', 'employmentLength', 'employmentTitle', 'grade', 'homeOwnership', 'initialListStatus', # log_xgb_v4_3.txt 0.7403205770486649
# #                  'postCode', 'purpose', 'regionCode', 'subGrade', 'title', 'verificationStatus']
# MeanEnocodeFeature = class_list  # 声明需要平均数编码的特征
# ME = MeanEncoder(MeanEnocodeFeature, target_type='classification')  # 声明平均数编码的类
# train = ME.fit_transform(train, train_label)  # 对训练数据集的X和y进行拟合
#     # x_train_fav = ME.fit_transform(x_train,y_train_fav)#对训练数据集的X和y进行拟合
# test = ME.transform(test)  # 对测试集进行编码
# print('num0:mean_encode train.shape', train.shape, test.shape)


# ##############################################################################
# ## target encoding目标编码，回归场景相对来说做目标编码的选择更多，不仅可以做均值编码，还可以做标准差编码、中位数编码等
# enc_cols = []
# stats_default_dict = {
#     'max': train['isDefault'].max(),
#     'min': train['isDefault'].min(),
#     'median': train['isDefault'].median(),
#     'mean': train['isDefault'].mean(),
#     'sum': train['isDefault'].sum(),
#     'std': train['isDefault'].std(),
#     'skew': train['isDefault'].skew(),
#     'kurt': train['isDefault'].kurt(),
#     'mad': train['isDefault'].mad()
# }
# ### 暂且选择这三种编码
# # enc_stats = ['max', 'min', 'skew', 'std']
# enc_stats = ['mean','max', 'min', 'skew', 'std'] # log_xgb_v4_4 0.7404042973237561
# # enc_stats = ['mean','max', 'min'] # log_xgb_v4_5.txt cv_auc:  0.7401718285443225
# skf = KFold(n_splits=5, shuffle=True, random_state=6666)
# for f in tqdm(['postCode', 'regionCode','employmentTitle','title']):
# # for f in tqdm(['postCode', 'regionCode','employmentTitle','title','purpose','homeOwnership','grade', 'subGrade']): #log_xgb_v4_2.txt   0.740278214825506 
# # for f in tqdm(['applicationType', 'employmentLength', 'employmentTitle', 'grade', 'homeOwnership', 'initialListStatus', # log_xgb_v4_3.txt 0.7403205770486649
# #                  'postCode', 'purpose', 'regionCode', 'subGrade', 'title', 'verificationStatus']):
# # for f in tqdm(['postCode', 'regionCode','employmentTitle','title','applicationType', 'employmentLength', 'initialListStatus', 'verificationStatus']): # .7403611763344211
#     enc_dict = {}
#     for stat in enc_stats:
#         enc_dict['{}_target_{}'.format(f, stat)] = stat
#         train['{}_target_{}'.format(f, stat)] = 0
#         test['{}_target_{}'.format(f, stat)] = 0
#         enc_cols.append('{}_target_{}'.format(f, stat))
#     for i, (trn_idx, val_idx) in enumerate(skf.split(train, train_label)):
#         trn_x, val_x = train.iloc[trn_idx].reset_index(drop=True), train.iloc[val_idx].reset_index(drop=True)
#         enc_df = trn_x.groupby(f, as_index=False)['isDefault'].agg(enc_dict)
#         val_x = val_x[[f]].merge(enc_df, on=f, how='left')
#         test_x = test[[f]].merge(enc_df, on=f, how='left')
#         for stat in enc_stats:
#             val_x['{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].fillna(
#                 stats_default_dict[stat])
#             test_x['{}_target_{}'.format(f, stat)] = test_x['{}_target_{}'.format(f, stat)].fillna(
#                 stats_default_dict[stat])
#             train.loc[val_idx, '{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].values
#             test['{}_target_{}'.format(f, stat)] += test_x['{}_target_{}'.format(f, stat)].values / skf.n_splits

# print('num2:target_encode train.shape', train.shape, test.shape)

# train.drop(['postCode', 'regionCode', 'employmentTitle','title'], axis=1, inplace=True)
# test.drop(['postCode', 'regionCode', 'employmentTitle','title'], axis=1, inplace=True)
# print('输入数据维度：', train.shape, test.shape)




                        

# drop_feats = [f for f in train.columns if train[f].nunique() == 1 or train[f].nunique() == 0]
# train.drop(columns=drop_feats,inplace=True)
# test.drop(columns=drop_feats,inplace=True)
# cols = [col for col in train.columns if col not in ['id','isDefault','index'] + drop_feats]

# # train.to_csv('../data/fea_train_v2.csv',index=False)
# # test.to_csv('../data/fea_test_v2.csv',index=False)
# # train = pd.read_csv('../data/fea_train.csv')
# # test = pd.read_csv('../data/fea_test.csv')



# train.to_pickle('../data/fea_train_v2.pkl')
# test.to_pickle('../data/fea_test_v2.pkl')

# time.sleep(3600*1.5)

train = pd.read_pickle('../data/fea_train_v2.pkl' )
test = pd.read_pickle('../data/fea_test_v2.pkl')





old_cols = [col for col in train.columns if col not in ['id','isDefault','index']]

# all      0.7414399621379929
# 0  1080  0.7416374300092856 ==> dart 0.7425594305829544
# 1  963   0.7415071379928511
# 2  895   0.7414701773321343
# 3  848   0.7416128771770132
# 4  810   0.741520857322394
# 5   775  0.7416578291775182
# 6   750  0.7417024752765454 ==> dart 0.7424520787314424
# 7   723  0.7416120303835705
# 8  694   0.74164929177028   ===>
# 9  669   0.7415229100918341
# 10  637  0.741389388974043

# for t in [0,1,2,3,4,5,6,7,8,9,10]:
for t in [0,6,8]:
# for t in [8]:
    print('!!!!!!!!!!!!!!!!!!!!!, ',t)
    new_feat_imp_df = pd.read_csv('../sub/lgb/0.7414399621379929/feat_lgb_baseline.csv')
    new_feat_imp_df = new_feat_imp_df[new_feat_imp_df['importance']>t]

    imp_fea = new_feat_imp_df['column'].tolist()
    cols = [col for col in old_cols if col in imp_fea]
    
    print('fea shape: ',len(cols))

    # nfold = 5
    nfold = 10
    seeds = [19960218]
    # nfold = 12
    # seeds = [42,19960218,2021,7,999,233,88,13,9527,126]

    print('nfold:',nfold)

    oof = np.zeros(train.shape[0])

    test['isDefault'] = 0
    df_importance_list = []

    # print(train[cols].head(10))

    val_aucs = []

    test_sqrt = test.copy()

    for seed in seeds:
        skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
        for i, (trn_idx, val_idx) in enumerate(skf.split(train, train['isDefault'])):
            print('--------------------- {} fold ---------------------'.format(i))
            t = time.time()
            trn_x, trn_y = train[cols].iloc[trn_idx].reset_index(drop=True), train['isDefault'].values[trn_idx]
            val_x, val_y = train[cols].iloc[val_idx].reset_index(drop=True), train['isDefault'].values[val_idx]


            model = lgb.LGBMClassifier(
                # learning_rate=0.05,
                learning_rate=0.01,
                n_estimators=10230,
                # n_estimators=6000,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                metric=None,
                # boosting_type='dart',
            )

            lgb_model = model.fit(
                trn_x, trn_y,
                eval_set=[(val_x, val_y)],
                # categorical_feature=cate_cols,
                eval_metric='auc',
                early_stopping_rounds=200,
                verbose=200
            )

            oof[val_idx] = lgb_model.predict_proba(val_x)[:, 1]
            test['isDefault'] += lgb_model.predict_proba(test[cols])[:, 1] / skf.n_splits / len(seeds)
            test_sqrt['isDefault'] += lgb_model.predict_proba(test[cols])[:, 1]**0.5 / skf.n_splits / len(seeds)



            df_importance = pd.DataFrame({
                'column': cols,
                'importance': lgb_model.feature_importances_,
            })
            df_importance_list.append(df_importance)

            del lgb_model

        cv_auc = roc_auc_score(train['isDefault'], oof)
        val_aucs.append(cv_auc)
        print('\ncv_auc: ', cv_auc)
    print(val_aucs, np.mean(val_aucs))
    result_aucs = np.mean(val_aucs)
    save_path = '../sub/lgb/'+str(result_aucs)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print('save_path',save_path)

    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby([
        'column'
    ])['importance'].agg('mean').sort_values(ascending=False).reset_index()

    ##################################
    submit = pd.read_csv('../data/sample_submit.csv')
    submit['id'] = test['id']
    submit['isDefault'] = test['isDefault']

    submit.to_csv(os.path.join(save_path,'baseline_lgb_auc_{}.csv'.format(result_aucs)), index = False)

    ##################################
    submit['isDefault'] = test_sqrt['isDefault']

    submit.to_csv(os.path.join(save_path,'baseline_lgb_auc_{}_sqrt.csv'.format(result_aucs)), index = False)


    ##################################

    df_importance.to_csv(os.path.join(save_path,'feat_lgb_baseline.csv'), index = False)

    np.save(os.path.join(save_path,'baseline_lgb_auc_{}.npy'.format(result_aucs)),oof)
