# Tianchi_antai Top21 Solution

## “安泰杯”——金融科技学习赛 Top21解决方案（solo）

### 赛题简介
    以金融风控中的个人信贷为背景，要求选手根据贷款申请人的数据信息预测其是否有违约的可能，以此判断是否通过此项贷款.

### 解决方案
    1.特征工程
        业务特征
        对一阶、二阶类别特征做频数特征
        类别特征和数值特征的组合特征
        对高基数类别特征做mean encoder
        对部分类别特征做target encoder:
    2.模型
        * xgb/lgb/cat+stacking 
            ==> xgb单模单seed auc:0.7454
            ==> 没有足够的提交次数验证lgb、cat的单模
            ==> stacking     auc：0.7465
        * deep-forest,tabnet效果不佳
    3.基于特征重要性的特征选择
        对xgb/lgb/cat分别根据对应模型给出的特征重要性进行了三个程度的特征选择 用于生成多样化模型
    3.超参数搜索
        尝试了optuna 、BayesianOptimization 效果均不行
        手动调节了学习率有较小收益

### 总结
    1.特征工程收益较高，但是没找到magic feature
    2.stacking确实有效
    3.算力有限没能多跑一些seed
    4.模型调参做的不好
    5.伪标签的正确使用还在尝试，本次比赛可能使用姿势不对. ==> https://www.kaggle.com/c/tabular-playground-series-apr-2021/discussion/231738 kaggle这个tips里说到的 check the training metric only by real ground truth labels.  很合理，但是没看到具体实现，感觉上简单样本带权+custom metric似乎不足够？如果任何朋友有兴趣或者知道具体的实现方法，求教！！


