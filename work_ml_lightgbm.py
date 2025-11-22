import pandas as pd
import numpy as np
import lightgbm as GBM
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import json
import os


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()


name = 'GW_RCHG'
func = 'square'

features = pd.read_csv('features.csv')
label = pd.read_csv(f'./label/{name}_{func}_relation.csv')

# 对标签集进行标准化
y_standard = StandardScaler()
Y_scale = y_standard.fit_transform(label)

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(features, Y_scale
                                                    , test_size=0.3, random_state=30)

if os.path.exists(f'./LightGBM_params/{name}_{func}_params.json'):
    with open(f'./LightGBM_params/{name}_{func}_params.json', 'r', encoding='utf-8') as file:
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
    base_model = GBM.LGBMRegressor(n_estimators=300,verbose = -1)
    model = MultiOutputRegressor(
        base_model,
        n_jobs=-1
    )
    search_space = {
        'estimator__boosting_type': ['gbdt', 'dart', 'rf'],
        'estimator__num_leaves': np.arange(15, 60, 5),
        'estimator__max_depth': np.arange(2, 5, 1),
        'estimator__learning_rate': np.arange(0.05, 0.2, 0.01),
        'estimator__min_samples_leaf': np.arange(1, 5),
        'estimator__n_estimators': np.arange(100, 1200, 100),
        'estimator__reg_alpha': np.arange(0, 0.01, 0.001),
        'estimator__reg_lamba': np.arange(0, 0.01, 0.001)
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
    with open(f'./RF_params/{name}_{func}_params.json', 'w') as f:
        f.write(json.dumps(params, ensure_ascii=True, indent=4,cls=Encoder))
        print('最佳参数保存为json文件格式')
    print(f'测试集得分为{score:.4f}')