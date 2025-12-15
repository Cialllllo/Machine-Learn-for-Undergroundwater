import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
import json
import os
from modelprocessing import Encoder
import modelprocessing
from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV

# 仅对线性函数进行调参
names = ['GW_RCHG','SA_ST','SHGWT']
func = 'linear'

for name in names:
    print(f"Processing {name} with {func} function")
    X_train, X_test, y_train, y_test = modelprocessing.data_processing(relation=name, func=func)

    if os.path.exists(f'./RF_params/{name}_{func}_params.json'):
        model = modelprocessing.rf_processing(name,func)
        model.fit(X_train,y_train)
        score = model.score(X_test, y_test)
        print(f'测试集得分为{score:.4f}')
        print('各系数得分为：')
        for i in range(y_test.shape[1]):
            print(f'{r2_score(y_test[:,i],model.predict(X_test)[:,i]):.4f}')
    else:
        # 随机森林支持多输出
        base_model = RandomForestRegressor()
        model = MultiOutputRegressor(base_model,n_jobs=1)
        search_space = {
                'estimator__n_estimators': Integer(300, 1200),
                'estimator__max_depth': Integer(10, 100),
                'estimator__max_features': Real(0.1, 0.9),
                'estimator__min_samples_split': Integer(2, 20),
                'estimator__min_samples_leaf': Integer(1, 10),
                'estimator__ccp_alpha': Real(0.0, 0.5),
                'estimator__min_weight_fraction_leaf': Real(0.0, 0.2)
            }
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=search_space,
            n_iter=100,  # 迭代次数
            cv=5,  # 5 折交叉验证
            scoring='r2',
            verbose=2,
            n_jobs=-1,
            random_state=42
        )

        bayes_search.fit(X_train, y_train)
        print(bayes_search.best_params_)
        best_gbr_model = bayes_search.best_estimator_
        score = best_gbr_model.score(X_test,y_test)
        params = bayes_search.best_params_
        with open(f'./RF_params/{name}_{func}_params.json', 'w') as f:
            f.write(json.dumps(params,cls=Encoder))
            print('最佳参数保存为json文件格式')
        print(f'测试集得分为{score:.4f}')