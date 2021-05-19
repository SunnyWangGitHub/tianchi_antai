# 目前可玩的比赛

##        
    招商银行 FinTech精英训练营 比赛
    https://career.cloud.cmbchina.com/index.html#applyHistory

    天池 “安泰杯”——金融科技学习赛 ==> 可以主力打！

###
    baseline_v1:
            0.7344575880320188
         imp>0.0:
            0.7343101102755015
         imp>1.0:
            0.7344724652319666
         imp>2.0
            0.7343892529174953v
    
    baseline_v1 暴力特征
        all:
            0.7362311281537596
        imp>1
            0.736273308129414
        imp>2:
            0.7362615653067335
        imp>4
            0.7360749537729177
        imp>10
            0.7360821736719727
    

basline.py
    5fold  0.7353485976562788
    10fold 0.7355829093436489 0.7378


baseline_xgb.py
    5 fold:
        0.7372409548691479 0.7389
baseline_cat.py
    5 fold:
        0.7369621771729877 0.7380

lgb+xgb+cat 简单voteing 效果不行0.7366 感觉模型0.8以上变少了


xgb_v1
    12fold  + 10 seed :0.7392
    12fold  + 10 seed  + sqrt 0.5:0.7392 ?? 重新提交试试？


+ rankaverage?


xgb_v2:
    特征： https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.6.28371ed3eckgFL&postId=211610   ==> 这个特征+cat/lgb
    5fold 0.7375430099830141  0.7391 还行
          0.7375754753830226
    
    12fold + 10 seed: 

    
    + 暴力特征： 0.7375101302653955

cat_V2:
     + cat_features=cate_cols  0.7373857612292617 + 好一点点
     不加 cat_features=cate_cols 0.7373044367926995






https://zhuanlan.zhihu.com/p/29794651

stacking


新特征 + meanencoder
xgb_v3:
    5 fold: 
        0.7400737448608843 0.7419 


新特征 + meanencoder + targetencoder
xgb_v3:
    5 fold: 0.7402604356412531

xgb_V4:
    meanencoder ==>  
    ['postCode', 'regionCode','employmentTitle','title'] 
        ==> 
    ['postCode', 'regionCode','employmentTitle','title','purpose','homeOwnership','grade', 'subGrade']
    5 fold: 0.7403838611248814 
            <!-- 0.7402877376342777? -->

todo:
    targetencoder 的测试
        编码方式的测试
        只对高基数类别和对所有类别都做 ==> 确实只适合高基数特征，然后 'mean','max', 'min', 'skew', 'std' 最好
    目前最佳：0.7404042973237561
    频次特征
        一阶类别频次特征  0.7408332429082879
        二阶类别频次特征
    
    类别交叉：
        0.7418411655833838 0.7441  ==> to stacking

    类别+num交叉：
        一阶类别、num 交叉
        二阶类别、num交叉
    
    0.7427262249215246    0.7448          ==> to stacking

    0.7427858119992995    0.7448

    + 参数：
        model = xgb.XGBClassifier(
            max_depth=8,
            min_child_weight=5,
            subsample = 0.8,
            colsample_bytree = 0.4,
            learning_rate=0.01,
            n_estimators=10000,
            random_state=seed,
            # tree_method='gpu_hist',
        )
    0.7435630172878048 0.7448？





    rank_average : xgb_v6 lgb cat
    lgb:   lgb lgb+dart   
    cat:
    nn:
        xgb: 
            all: 0.7427262249215246  0.7448  ==> 10 fold: 0.743321865923405
            fea_imp > 0.00035 : fea_shape: 824  0.7427944350515158
             fea_imp >0.0004   : fea shape: 669  0.7429050984697168
             fea_imp> 0.00045  : fea shape: 394  0.7435384279532005 ?
             fea_imp>0.0005 : fea shape:251      0.7435759289237629  
             
            fea_imp>0.0005 : fea shape:251 gpu  0.743685556579735
                            10trail param:0.7436184091215584
                            faical loss :0.7436855565797353

                  
            new_param: 10 fold: gpu seed 19980218
#             model = xgb.XGBClassifier(max_depth=6,
#                                     learning_rate=0.02,
#                                     n_estimators=10000,
#                                     colsample_bytree = 0.7,#0.8
#                                     subsample = 0.7,
#                                      min_child_weight = 1,  # 2 3
#                                       tree_method='gpu_hist',
#                                       random_state = seed
#                                      )
            all:      0.7433362933502479
            fea_imp> 0.00045  fea shape: 394 : 0.7435634657694843
            fea_imp>0.0005    fea shape: 251 : 0.7438129012799012
            fea_imp>  0.0006  fea shape: 152:  0.7428870892014139


            new_param: 10 fold: gpu seed 19980218
            model = xgb.XGBClassifier(max_depth=6,
                                    learning_rate=0.02,
                                    n_estimators=10000,
                                    subsample=0.8,
                                    reg_alpha=10,
                                    reg_lambda=12,
                                    tree_method='gpu_hist',
                                    random_state = seed
                                     )

                all:    0.7440091519290345 ?
                fea_imp>0.00045  fea shape: 394 0.7448640393211063 单模型能不能涨？
                fea_imp> 0.0005  fea shape: 251 0.7447955092062535
                fea_imp>  0.0006  fea shape: 152: 0.7440696870793033
            learning_rate=0.01,
                all:       0.7441654160737187
                fea_imp>0.00045  fea shape  394 : 0.7448781616927469
                fea_imp>0.0005   fea shape: 251:  0.7450030884142667 ==> to sub  0.7454

            10 fold: gpu 
            model = xgb.XGBClassifier(max_depth=6,
                                    learning_rate=0.01,
                                    n_estimators=10000,
                                    subsample=0.8,
                                    reg_alpha=10,
                                    reg_lambda=12,
                                    tree_method='gpu_hist',
                                    random_state = seed
                                     )
                seed 42:
                         0.7441624116559061
                         0.7449567446192733
                         0.7448769802342632
                seed 2021:
                         0.7441806786418259
                         0.7448695764538549
                         0.7449423616194176

        cat :   
            # all  0.7432692816678316
                # 0.0 1583 0.7432692816678316  ==> sub & stacking!
                # 0.005 1319 0.7431691591369094
                # 0.001 1276 0.7431495806401078
                # 0.003 1137 0.7431980910456386
                # 0.005 1047 0.7432328775038025
                # 0.008 942  0.7432393547256885
                # 0.01 893   0.7431496784557912
                # 0.02 698   0.7433305505335608 ==> sub & stacking!
                # 0.03 565   0.7432965713923161

                # 0.02  0.7433305505335608 + cate_cols 0.7434669703043668 sub& stacking  0.7447
            
            10fold:
                all:  0.7435655432814592  + cate_cols: 0.7436636773794019
                0.008 0.7436511991389277  + cate_cols: 0.7435876632934779
                0.02  0.7437433411506406 + cate_cols: 0.7437157589183245
            10fold lr 0.02 + cate:  0.7438445204979373
                                    0.743875157381575
                                    0.7439352373924754

                    lr 0.01         0.7437396991943108
                                    0.7438668602479283
                                    0.743891790910154

            10fold:
                model = cbt.CatBoostClassifier(eval_metric='AUC',
                                            max_depth=8,
                                            learning_rate=0.01,
                                            n_estimators=10000,
                                            min_data_in_leaf = 64,
                                            l2_leaf_reg = 0.01,
                                            subsample = 0.8,
                                            random_state=seed)
                0.743593216334238
                0.7436376662931555
                0.7435902893383509




        lgb:
            all lgb_v5: 0.7414399621379929

                # all      0.7414399621379929
                # 0  1080  0.7416374300092856 
                        ==> dart 0.7425594305829544 ==> 10fold dart:0.7431850958339178
                # 1  963   0.7415071379928511
                # 2  895   0.7414701773321343
                # 3  848   0.7416128771770132
                # 4  810   0.741520857322394
                # 5   775  0.7416578291775182
                # 6   750  0.7417024752765454 
                            ==> dart 0.7424520787314424 ==> 10fold dart 0.7428928827701371
                # 7   723  0.7416120303835705
                # 8  694   0.74164929177028   
                            ==>dart 0.7425022745252645 ==>10fold dart 0.7431227444896279
                # 9  669   0.7415229100918341
                # 10  637  0.741389388974043


            dart 6k足够

    myserver: lgb和cat都先用完整原始特征跑，看看结果怎么样？  看看各自选择能不能stacking出好的结果

    5fold的结果stacking：
            stacking_BR_auc_0.7448558073799789.csv  0.7455

            stacking_BR_auc_0.7452710127373581.csv 0.7459

            stacking_BR_auc_0.74529528214091.csv

    + 10fold:
        stacking  stacking_BR_auc_0.7453411941008917.csv   

        stacking_BR_auc_0.74543288287124.csv   0.7460

        br也是10fold
        10sub_1_stacking_BR_auc_0.7457619493976538.csv   0.7462

        5sub_1_stacking_BR_auc_0.7457866325171291.csv

        10sub_1_stacking_BR_auc_0.7457928830351728.csv

        10sub_1_stacking_BR_auc_0.7458714016056716.csv  0.7462

        纯10fold: 10sub_1_stacking_BR_auc_0.7456694461684129.csv

        10sub_1_stacking_BR_auc_0.7459973496782877.csv  0.7464

        10sub_1_stacking_BR_auc_0.7460731019668653.csv  0.7464

        10sub_1_stacking_BR_auc_0.7462084100552883.csv  0.7465

        10sub_1_stacking_BR_auc_0.7463919256121727.csv  0.7465

        10sub_1_stacking_BR_auc_0.7463923582494334.csv  0.7465

    gmeans
        sub_1 0.7447 胯

todo:
    nn:
    https://github.com/manujosephv/pytorch_tabular
    https://github.com/dreamquark-ai/tabnet

    loss:
    https://zhuanlan.zhihu.com/p/260569269 并不行


    xgb+ optuna 10 trials:
        Number of finished trials: 10
Best trial: {'lambda': 5.60362917448845, 'alpha': 0.014130230732570874, 'colsample_bytree': 0.5, 'subsample': 1.0, 'learning_rate': 0.008, 'max_depth': 12, 'min_child_weight': 80}


########
                    model = xgb.XGBClassifier(
                                            max_depth = 6,
                                            learning_rate=0.01,
                                            n_estimators=10000,
                                            tree_method='gpu_hist',
                                            random_state = seed,
                                            min_child_weight = 4,
                                            colsample_bytree = colsample_bytree,
                                            subsample = subsample
                                             )
# subsample : 0.7 colsample_bytree 0.9 cv_auc:  0.7435204457412359 

