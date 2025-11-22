from sklearn.feature_selection import RFECV
import modelprocessing
import matplotlib.pyplot as plt

X_train,X_test,y_train,y_test = modelprocessing.data_processing(func='square')

def remove():
    # 将xgb的最佳模型加载出来
    multi_model = modelprocessing.xgb_processing(func='square')

    multi_model.fit(X_train,y_train)
    models = multi_model.estimators_
    num = 0
    for model in models:
        rfecv = RFECV(estimator=model,step=1,cv=5,scoring='r2')
        rfecv.fit(X_train,y_train[:,num])

        print(f'在模型{num+1}中')
        print('Optimal number of features: {}'.format(rfecv.n_features_))
        print('被选中的特征为')
        for col in X_train.columns[rfecv.support_]:
            print('->{}'.format(col))
        rank = dict(zip(X_train.columns,rfecv.ranking_))
        rank_items = list(rank.items())
        rank_items.sort(key = lambda x:x[1],reverse=False)
        print('其排名为')
        for i in rank_items:
            print(f'{i[0]}->{i[1]}')
        num += 1




        plt.figure(figsize=(8, 4))
        plt.title(f'Recursive Feature Elimination with Cross-Validation{num}_square', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
        plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'], color='#303F9F', linewidth=3)

        plt.show()
    return None

multi_model = modelprocessing.xgb_processing(func='square')
multi_model.fit(X_train.loc[:,['PCP_WW','PCP_SM','IRR_SM'
                              ,'PCP_RY','TMP_AV','TMP_MX','TMP_MN']],y_train)
print('三次函数测试集得分为{:.4f}'.format(multi_model.score(X_test.loc[:,['PCP_WW','PCP_SM','IRR_SM'
                              ,'PCP_RY','TMP_AV','TMP_MX','TMP_MN']],y_test)))

multi_model = modelprocessing.xgb_processing(func='cubic')
multi_model.fit(X_train.loc[:,['PCP_WW','PCP_SM','IRR_SM'
                              ,'PCP_RY','TMP_AV','TMP_MX','TMP_MN']],y_train)
print('二次函数测试集得分为{:.4f}'.format(multi_model.score(X_test.loc[:,['PCP_WW','PCP_SM','IRR_SM'
                              ,'PCP_RY','TMP_AV','TMP_MX','TMP_MN']],y_test)))


'''['Clay_Up','Silt_Up','Sand_Up','Clay_Down'
                              ,'Silt_Down','Sand_Down','OC_Up','BD_Up'
                              ,'BD_Down','AWC_Up','AWC_Down','K_Up','K_Down'
                               ,'PCP_WW','PCP_SM','IRR_SM','PCP_RY','TMP_AV'
                               ,'TMP_MX','TMP_MN']'''