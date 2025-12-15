from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as GBM
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import shap
import gc
import warnings
warnings.filterwarnings("ignore")


# 在 modelprocessing.py 中添加以下函数

from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    """
    计算并返回常见回归指标

    参数:
        y_true (np.ndarray): 真实值
        y_pred (np.ndarray): 预测值

    返回:
        dict: 包含各种指标的字典
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2)
    }

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
def data_processing(relation = 'GW_RCHG',func = 'cubic',scale = True,back = False):
    # 读取准备好的特征
    features = pd.read_csv(r'features.csv')
    data_label = pd.read_csv(fr'.\label\{relation}_{func}_relation.csv')  # 这里需要改成实际的标签文件名

    # 正确做法：先划分，再分别标准化
    X_train, X_test, y_train, y_test = train_test_split(features, data_label, test_size=0.3, random_state=30)

    # 对标签标准化
    if scale:

        y_standard = StandardScaler()
        y_train_scaled = y_standard.fit_transform(y_train)  # 只用训练集拟合
        y_test_scaled = y_standard.transform(y_test)  # 用训练集的参数转换测试集
        if back:
            # 这个决定你是否返回标准缩放
            return X_train,X_test,y_train_scaled,y_test_scaled,y_standard
        else:
            return X_train,X_test,y_train_scaled,y_test_scaled
    else:
        if back:
            assert not back,'你明明不要标准化的，你要back干啥'
            return X_train, X_test, y_train, y_test
        else:
            return X_train, X_test, y_train, y_test



def rf_processing(relation = 'GW_RCHG',func = 'cubic'):
    with open(f'./RF_params/{relation}_{func}_params.json', 'r', encoding='utf-8') as file:
        # 使用 json.load() 方法加载文件内容并转换为 Python 对象
        params = json.load(file)
    # 返回最佳参数的模型
    base_model = RandomForestRegressor()
    model = MultiOutputRegressor(base_model,n_jobs=1).set_params(**params)
    return model

def xgb_processing(relation = 'GW_RCHG',func = 'cubic'):
    params = joblib.load(f'./XGB_model/{relation}_{func}_relation_model.pkl').get_params()
    base_model = xgb.XGBRegressor()
    model = MultiOutputRegressor(base_model,n_jobs=1).set_params(**params)
    return model

def gbr_processing(relation = 'GW_RCHG',func = 'cubic'):
    with open(f'./GBR_params/{relation}_{func}_params.json', 'r', encoding='utf-8') as file:
        params = json.load(file)
    base_model = GradientBoostingRegressor()
    # 修正参数传递
    model = MultiOutputRegressor(base_model,n_jobs=1)
    model.set_params(**params)  # 正确设置参数的方式
    return model

def gbm_processing(relation = 'GW_RCHG',func = 'cubic'):
    with open(f'./LightGBM_params/{relation}_{func}_params.json', 'r', encoding='utf-8') as file:
        params = json.load(file)
    base_model = GBM.LGBMRegressor(verbose = -1)
    # 修正参数传递
    model = MultiOutputRegressor(base_model,n_jobs=1)
    model.set_params(**params)  # 正确设置参数的方式
    return model


def multioutput_rfecv(model_process,
                             relation='GW_RCHG',
                             func='cubic',
                             min_features_to_select=1,
                             step=1,
                             cv_splits=5):

    # 数据准备
    X_train, X_test, y_train, y_test = data_processing(relation, func)

    # 初始化基础模型（只为了获取结构，不用于训练）
    base_model = model_process(relation, func)
    if hasattr(base_model, "n_jobs"):
        base_model.set_params(n_jobs=1)

    # 强制 base model 的子模型单线程
    for est in getattr(base_model, "estimators_", []):
        if hasattr(est, "n_jobs"):
            est.set_params(n_jobs=1)
        if hasattr(est, "nthread"):
            est.set_params(nthread=1)

    n_features = X_train.shape[1]
    current_features = np.arange(n_features)

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=30)

    # 🔥 历史记录
    feature_hist = []   # 每一轮的特征索引
    score_hist = []     # 每一轮的CV得分

    # ==========================
    #       RFECV 主循环
    # ==========================
    while len(current_features) >= min_features_to_select:

        X_sub = X_train.iloc[:, current_features]

        cv_scores = []

        # 手写 CV（避免 Windows 并行出错）
        for train_idx, val_idx in cv.split(X_sub):
            X_tr, X_val = X_sub.iloc[train_idx], X_sub.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # 每折一个新模型
            m = model_process(relation, func)

            # 强制单线程
            if hasattr(m, "n_jobs"):
                m.set_params(n_jobs=1)
            for est in getattr(m, "estimators_", []):
                if hasattr(est, "n_jobs"):
                    est.set_params(n_jobs=1)
                if hasattr(est, "nthread"):
                    est.set_params(nthread=1)

            m.fit(X_tr, y_tr)
            pred = m.predict(X_val)

            # 多输出统一 R2
            score = r2_score(y_val, pred, multioutput='uniform_average')
            cv_scores.append(score)


        mean_score = np.mean(cv_scores)

        # 记录历史
        feature_hist.append(current_features.copy())
        score_hist.append(mean_score)

        print(f"特征数 {len(current_features)} → CV 平均 R² = {mean_score:.4f}")

        # 拟合一次，用于特征重要性
        base_model.fit(X_sub, y_train)

        # 平均每个输出的重要性
        importances = np.mean(
            [est.feature_importances_ for est in base_model.estimators_],
            axis=0
        )

        # 删除 step 个最不重要特征
        least_k = np.argsort(importances)[:step]

        print(f"删除特征: {current_features[least_k]}\n")

        # 更新特征集合
        current_features = np.delete(current_features, least_k)

        gc.collect()

    # ==========================
    #      找到最佳 index
    # ==========================
    best_idx = np.argmax(score_hist)
    best_score = score_hist[best_idx]
    best_features = feature_hist[best_idx]

    print("============== 结果总结 ==============")
    print(f"最佳特征数量: {len(best_features)}")
    print(f"最佳平均 CV R²: {best_score:.6f}")
    print("=====================================")

    # 绘制曲线
    plt.figure(figsize=(8, 5))
    plt.plot([len(f) for f in feature_hist[::-1]], score_hist[::-1],
             marker='o', markersize=6, linewidth=2, color='#2E86AB')
    plt.xticks(range(1, 22))
    plt.xlabel("Number of Features", fontsize=12, fontweight='bold')
    plt.ylabel("Cross-Validation R² Score", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.title(f"Feature Selection Performance: {relation} {func}",
              fontsize=14, fontweight='bold', pad=20)

    # Add some styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{relation}_{func}_rfecv.png')

    return best_features, best_score, feature_hist, score_hist


# 替换或更新 modelprocessing.py 中的 R2_plot 函数为如下版本：

def R2_plot(Y_true,
            Y_pred,
            path='parity_plot_dynamic_limits.png',
            title="Model Performance For Param A"):
    """
    绘制真实值 vs. 预测值的散点图，并添加 y=x 基准线。
    轴的范围由数据的最小值和最大值动态确定。
    同时打印多个回归指标。

    Args:
        Y_true (array-like): 真实值 (X轴)。
        Y_pred (array-like): 预测值 (Y轴)。
        path (str): 图片保存路径。
        title (str): 图表标题。
    """

    # 1. 确保数据是 NumPy 数组
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)

    # 2. 设置图形尺寸和风格
    plt.figure(figsize=(8, 8))
    plt.style.use('default')

    # 3. 动态计算轴的范围
    global_min = min(Y_true.min(), Y_pred.min())
    global_max = max(Y_true.max(), Y_pred.max())
    padding = (global_max - global_min) * 0.05
    limit_min = global_min - padding
    limit_max = global_max + padding
    plot_range = np.linspace(limit_min, limit_max, 100)

    # 4. 绘制 y=x 基准线
    plt.plot(plot_range, plot_range,
             color='red',
             linestyle='--',
             linewidth=2.5,
             label='$y=x$ (Ideal Prediction)')

    # 5. 绘制预测散点图
    plt.scatter(Y_true, Y_pred,
                s=20,
                color='#1f77b4',
                alpha=0.7,
                label='Predicted Data')

    # 6. 设置坐标轴范围
    plt.xlim(limit_min, limit_max)
    plt.ylim(limit_min, limit_max)

    # 7. 添加标题、标签及样式
    metrics = calculate_metrics(Y_true, Y_pred)
    metric_text = '\n'.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    full_title = f"{title}\n{metric_text}"

    plt.title(full_title, fontsize=14, fontweight='bold')
    plt.xlabel('True Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"动态范围的预测 vs. 真实值图已保存为 {path}")
    print("模型性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")




