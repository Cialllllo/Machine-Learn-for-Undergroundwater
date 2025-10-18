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




# 读取准备好的特征
Eigenvalue = pd.read_csv(r'特征re.csv')
data_label = pd.read_csv(r'标签.csv')

# 对标签集进行标准化
y_standard = StandardScaler()
Y_scale = y_standard.fit_transform(data_label)

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(Eigenvalue, Y_scale
                                                    , test_size=0.3, random_state=30)



# 开始训练调参
if os.path.exists('best_multi_xgb_re.pkl'):
    best_model = joblib.load('best_multi_xgb_re.pkl')
    y_pred = best_model.predict(X_test)
    r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
    print(f"平均 R²:{r2_score(y_test[:, 1:], y_pred[:, 1:]):.4f}")
    print(f'交叉验证R²为{cross_val_score(best_model, X_train, y_train, cv=10).mean():.4f}')
    print("每个系数的 R²:", r2_scores)
else:
    base_model = xgb.XGBRegressor(
        n_estimators=1000,
        tree_method='hist',
        random_state=0,
        verbosity=0
    )

    # MultiOutput 包裹器：并行策略由你控制（此处设置为并行训练每个目标）
    multi_model = MultiOutputRegressor(base_model, n_jobs=-1)

    # IMPORTANT: 这里的搜索空间键必须以 estimator__ 开头，指向 base estimator 的参数
    search_spaces = {
        'estimator__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'estimator__max_depth': Integer(3, 12),
        'estimator__min_child_weight': Integer(1, 10),
        'estimator__subsample': Real(0.5, 1.0),
        'estimator__colsample_bytree': Real(0.5, 1.0),
        'estimator__gamma': Real(0, 5),
        'estimator__reg_alpha': Real(0, 10),
        'estimator__reg_lambda': Real(0, 10)
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
    joblib.dump(opt.best_estimator_, 'best_multi_xgb_re.pkl')
    # 预测评估
    y_pred = opt.predict(X_test)
    r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
    mean_r2 = sum(r2_scores) / len(r2_scores)

    print("每个系数的 R²:", r2_scores)
    print("平均 R²:", mean_r2)
    print("最佳参数:", opt.best_params_)




