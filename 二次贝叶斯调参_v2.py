# -*- coding: utf-8 -*-
"""
文件名: 二次贝叶斯调参_v2.py
功能: 读取上一轮最优 XGBoost 模型，缩小范围进行二次精调
作者: 欧佩奇 & ChatGPT
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import os
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

os.environ["OMP_NUM_THREADS"] = '6'

# ========== 1. 读取特征与标签 ==========
Eigenvalue = pd.read_csv(r'特征re.csv')
data_label = pd.read_csv(r'标签.csv')

# 标签标准化
y_standard = StandardScaler()
Y_scale = y_standard.fit_transform(data_label)

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    Eigenvalue, Y_scale, test_size=0.3, random_state=30
)

# ========== 2. 载入上一轮最优模型参数 ==========
if not os.path.exists('best_multi_xgb_re.pkl'):
    raise FileNotFoundError("未找到 best_multi_xgb_re.pkl，请先运行第一阶段调参脚本。")

old_model = joblib.load('best_multi_xgb_re.pkl')
best_params_old = old_model.get_params()
print("✅ 已加载第一轮最优模型参数：")
for k, v in best_params_old.items():
    if "estimator__" in k:
        print(f"{k}: {v}")

# ========== 3. 定义第二阶段更窄的搜索空间 ==========
# 以第一轮最优参数为中心，缩小范围
search_spaces_v2 = {
    'estimator__learning_rate': Real(
        max(0.02, best_params_old.get('estimator__learning_rate', 0.05) * 0.5),
        min(0.08, best_params_old.get('estimator__learning_rate', 0.05) * 1.5),
        prior='log-uniform'
    ),
    'estimator__n_estimators': Integer(1500, 2500),
    'estimator__max_depth': Integer(4, 6),
    'estimator__min_child_weight': Integer(1, 6),
    'estimator__subsample': Real(0.7, 0.9),
    'estimator__colsample_bytree': Real(0.45, 0.7),
    'estimator__reg_alpha': Real(0.0, 8.0),
    'estimator__reg_lambda': Real(5.0, 15.0),
    'estimator__colsample_bylevel': Real(0.8, 1.0),
    'estimator__colsample_bynode': Real(0.8, 1.0),
    'estimator__grow_policy': Categorical(['depthwise', 'lossguide']),
}

# ========== 4. 初始化基础模型 ==========
base_model = xgb.XGBRegressor(
    tree_method='hist',
    device='cuda',
    random_state=42,
    verbosity=0
)

multi_model = MultiOutputRegressor(base_model, n_jobs=-1)

# ========== 5. 二次贝叶斯调参 ==========
opt2 = BayesSearchCV(
    estimator=multi_model,
    search_spaces=search_spaces_v2,
    n_iter=60,
    cv=5,
    scoring='r2',
    n_jobs=1,
    verbose=2,
    random_state=42
)

print("\n🚀 开始第二阶段贝叶斯精调...")
opt2.fit(X_train, y_train)

# ========== 6. 保存模型 ==========
joblib.dump(opt2.best_estimator_, 'best_multi_xgb_re_v2.pkl')

# ========== 7. 结果评估 ==========
y_pred = opt2.predict(X_test)
r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
mean_r2 = r2_score(y_test, y_pred)
cv_r2 = cross_val_score(opt2.best_estimator_, X_train, y_train, cv=10).mean()

print("\n========== 精调结果 ==========")
print(f"平均 R²: {mean_r2:.4f}")
print(f"10 折交叉验证 R²: {cv_r2:.4f}")
print("每个输出维度 R²:", r2_scores)
print("最优参数:\n", opt2.best_params_)

# ========== 8. 参数对比报告 ==========
print("\n========== 参数对比报告 ==========")
for key, new_val in opt2.best_params_.items():
    old_val = best_params_old.get(key)
    if old_val != new_val:
        print(f"{key}: {old_val} → {new_val}")
    else:
        print(f"{key}: 保持不变 ({old_val})")

print("\n✅ 二次贝叶斯调参完成，模型已保存为 best_multi_xgb_re_v2.pkl")
