import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import json
import os
import modelprocessing
# 读取特征与标签数据集

name = 'SA_ST'
func = 'cubic'
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
    base_model = GradientBoostingRegressor(n_estimators=300)
    model = MultiOutputRegressor(
        base_model,
        n_jobs=-1
    )
    search_space = {
        'estimator__n_estimators': np.arange(300,1200,100),
        'estimator__learning_rate': np.arange(0.05,0.1,0.01),
        'estimator__max_depth': np.arange(3,5,1),
        'estimator__min_samples_split': np.arange(2,5),
        'estimator__min_samples_leaf': np.arange(1,5),
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=search_space,
        cv=5,
        n_jobs=1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    best_gbr_model = grid_search.best_estimator_
    score = best_gbr_model.score(X_test,y_test)
    params = grid_search.best_params_
    with open(f'./GBR_params/{name}_{func}_params.json', 'w') as f:
        json.dump(params, f, cls=modelprocessing.Encoder, ensure_ascii=True, indent=4)
    print(f'测试集得分为{score:.4f}')