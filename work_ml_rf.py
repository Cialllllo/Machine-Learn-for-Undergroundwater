import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
import json
import os
from modelprocessing import Encoder
import modelprocessing
# 读取特征与标签数据集

name = 'SHGWT'
func = 'quadratic'


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
        'estimator__n_estimators': [300, 500, 800],
        'estimator__max_depth': [None, 5, 10, 20],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__max_features': ['sqrt', 'log2'],
        'estimator__bootstrap': [True]
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=search_space,
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    best_gbr_model = grid_search.best_estimator_
    score = best_gbr_model.score(X_test,y_test)
    params = grid_search.best_params_
    with open(f'./RF_params/{name}_{func}_params.json', 'w') as f:
        f.write(json.dumps(params,cls=Encoder))
        print('最佳参数保存为json文件格式')
    print(f'测试集得分为{score:.4f}')