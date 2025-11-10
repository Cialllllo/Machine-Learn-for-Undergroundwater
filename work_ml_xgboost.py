import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import os
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer,r2_score
os.environ["OMP_NUM_THREADS"] = '6'




name = 'SHGWT'
func = 'cubic'

# 读取准备好的特征
Eigenvalue = pd.read_csv(r'features.csv')
data_label = pd.read_csv(fr'.\label\{name}_{func}_relation.csv') # 这里需要改成实际的标签文件名


# 对标签集进行标准化
y_standard = StandardScaler()
Y_scale = y_standard.fit_transform(data_label)

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(Eigenvalue, Y_scale
                                                    , test_size=0.3, random_state=30)

def params_get(model,your_own = True):
    if your_own:
        best_model = joblib.load(model)
        # print(f'5折R2为{cross_val_score(best_model,X_train,y_train,cv=5).mean()}')
        y_pred = best_model.predict(X_test)
        r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
        print(f"平均 R²:{best_model.score(X_test, y_test):.4f}")
        params_lst = best_model.get_params()
        for params in params_lst.items():
            print(f'{params[0]}的最优值为{params[1]}')
        # print(f'交叉验证R²为{cross_val_score(best_model, X_train, y_train, cv=10).mean():.4f}')
        print("每个系数的 R²:", r2_scores)
    else:
        print(f'{name}--{func}')
        best_params = joblib.load(model).get_params()
        # print(f'5折R2为{cross_val_score(best_model,X_train,y_train,cv=5).mean()}')
        # 确定基础模型
        base_model = xgb.XGBRegressor(early_stopping_rounds=100)
        # 导入参数
        best_model = MultiOutputRegressor(base_model,n_jobs=-1).set_params(**best_params)
        best_model.fit(X_train,y_train)
        r2_scores = [best_model.estimators_[i].score(X_test,y_test[:,i]) for i in range(y_test.shape[1])]
        print(f"平均 R²:{best_model.score(X_test, y_test):.4f}")
        print(f'训练集得分为{best_model.score(X_train,y_train)}')
        params_lst = best_model.get_params()
        # for params in params_lst.items():
            # print(f'{params[0]}的最优值为{params[1]}')
        # print(f'交叉验证R²为{cross_val_score(best_model, X_train, y_train, cv=10).mean():.4f}')
        print("每个系数的 R²:", r2_scores)





# 开始训练调参
if os.path.exists(f'model/{name}_{func}_relation_model.pkl'): # 这里保存的模型最好也改下，和label的文件名最好一致，下面也是同理
    params_get(f'model/{name}_{func}_relation_model.pkl',False)
else:
    base_model = xgb.XGBRegressor(
        n_estimators=1000,
        tree_method='hist',  # 改为 hist
        device='cuda',  # 启用 GPU
        random_state=0,
        verbosity=0
    )

    # MultiOutput 包裹器：并行策略由你控制（此处设置为并行训练每个目标）
    multi_model = MultiOutputRegressor(base_model, n_jobs=-1)

    # IMPORTANT: 这里的搜索空间键必须以 estimator__ 开头，指向 base estimator 的参数
    search_spaces = {
        'estimator__learning_rate': Real(0.01, 0.15, prior='log-uniform'),
        'estimator__max_depth': Integer(4, 8),
        'estimator__min_child_weight': Integer(1, 10),
        'estimator__subsample': Real(0.6, 1.0),
        'estimator__colsample_bytree': Real(0.5, 0.9),
        'estimator__gamma': Real(0, 5),
        'estimator__reg_alpha': Real(0, 20),
        'estimator__reg_lambda': Real(0, 20),
        # 树数量增加（建议搭配小学习率）
        'estimator__n_estimators': Integer(800, 2000),
        'estimator__colsample_bylevel': Real(0.5, 1.0),
        'estimator__colsample_bynode': Real(0.5, 1.0),
    }

    # BayesSearchCV：注意 n_jobs 设置，避免与 MultiOutput 的 n_jobs 嵌套冲突
    opt = BayesSearchCV(
        estimator=multi_model,
        search_spaces=search_spaces,
        n_iter=100,  # 先少试 30 次，后面根据时间增加
        cv=5,
        scoring='r2',
        n_jobs=1,  # IMPORTANT: 设为 1，避免嵌套并行（或改成 -1 并把 multi_model.n_jobs=1）
        verbose=2,
        random_state=0
    )

    # 运行调参（可能耗时）
    opt.fit(X_train, y_train)
    # 保存训练好的 multi-output 模型
    joblib.dump(opt.best_estimator_, f'model/{name}_{func}_relation_model.pkl')
    # 预测评估
    y_pred = opt.predict(X_test)
    r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
    mean_r2 = opt.best_estimator_.score(X_test,y_test)
    print(f'训练集R2为{opt.best_estimator_.score(X_train, y_train)}')
    print('在训练中最佳分数为')
    print("每个系数的 R²:", r2_scores)
    print("平均 R²:", mean_r2)
    print("最佳参数:", opt.best_params_)




