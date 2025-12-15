import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
import json
import os
import modelprocessing
from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV

def searchCV(name,func):
    # 划分训练集
    X_train, X_test, y_train, y_test = modelprocessing.data_processing(relation=name, func=func)

    if os.path.exists(f'./GBR_params/{name}_{func}_params.json'):
        with open(f'./GBR_params/{name}_{func}_params.json', 'r', encoding='utf-8') as file:
            # 使用 json.load() 方法加载文件内容并转换为 Python 对象
            params = json.load(file)
        base_model = GradientBoostingRegressor()
        model = MultiOutputRegressor(
            base_model,
            n_jobs=-1
        ).set_params(**params)
        model.fit(X_train,y_train)
        score = model.score(X_test, y_test)
        print(f'测试集得分为{score:.4f}')
        print('各系数得分为：')
        for i in range(y_test.shape[1]):
            print(f'{r2_score(y_test[:,i],model.predict(X_test)[:,i]):.4f}')
    else:
        # GradientBoostingRegressor 原生不支持多输出回归
        base_model = GradientBoostingRegressor(n_estimators=300,random_state=42)
        model = MultiOutputRegressor(
            base_model,
            n_jobs=1
        )
        search_space = {
            'estimator__n_estimators': Integer(300, 1200),
            'estimator__learning_rate': Real(0.05, 0.1),
            'estimator__max_depth': Integer(3, 5),
            'estimator__min_samples_split': Integer(2, 5),
            'estimator__min_samples_leaf': Integer(1, 5)
        }
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=search_space,
            n_iter=100,  # 先少试 30 次，后面根据时间增加
            scoring='r2',
            cv=5,
            n_jobs=1,
            verbose=2,
            random_state=42
        )

        bayes_search.fit(X_train, y_train)
        print(bayes_search.best_params_)
        best_gbr_model = bayes_search.best_estimator_
        score = best_gbr_model.score(X_test,y_test)
        params = bayes_search.best_params_
        with open(f'./GBR_params/{name}_{func}_params.json', 'w') as f:
            json.dump(params, f, cls=modelprocessing.Encoder, ensure_ascii=True, indent=4)
        print(f'测试集得分为{score:.4f}')


if __name__ == '__main__':
    names = ['GW_RCHG','SA_ST','SHGWT']
    func = 'linear'
    for name in names:
        searchCV(name,func)