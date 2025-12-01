import numpy as np
import lightgbm as GBM
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
import json
import os
import modelprocessing


# 这里name和func需要改
name = 'GW_RCHG'
func = 'quadratic'
X_train, X_test, y_train, y_test = modelprocessing.data_processing(relation=name, func=func)

if os.path.exists(f'./LightGBM_params/{name}_{func}_params.json'): # 这里为保存路径修改一下，你要能找得到
    with open(f'./LightGBM_params/{name}_{func}_params.json', 'r', encoding='utf-8') as file:# 这里同理
        # 使用 json.load() 方法加载文件内容并转换为 Python 对象
        params = json.load(file)
    base_model = GBM.LGBMRegressor()
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
    # lightGBM 原生不支持多输出回归
    base_model = GBM.LGBMRegressor(n_estimators=300,verbose = -1,boosting_type='dart',reg_alpha=0,reg_lambda=0.1)
    model = MultiOutputRegressor(
        base_model,
        n_jobs=-1
    )
    search_space = {
        'estimator__num_leaves': [15, 31, 63],
        'estimator__max_depth': [-1,3,5,7],
        'estimator__learning_rate': [0.05, 0.1, 0.15],
        'estimator__n_estimators': [300, 500,800,1000,1200],
        'estimator__subsample': [0.8, 0.9, 1.0],
        'estimator__min_child_samples': [10, 20]
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
    # 将最佳参数进行保存，路径修改
    with open(f'./LightGBM_params/{name}_{func}_params.json', 'w') as f:
        f.write(json.dumps(params, ensure_ascii=True, indent=4,cls=modelprocessing.Encoder))
        print('最佳参数保存为json文件格式')
    print(f'测试集得分为{score:.4f}')