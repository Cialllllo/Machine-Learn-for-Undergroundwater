import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap

# ========== Step 1. 读取Excel文件 ==========
excel_file = "DATA-zhou2.xlsx"
data = pd.read_excel(excel_file)

# ========== Step 2. 数据检查与清理 ==========
print("原始数据维度:", data.shape)

# 检查 NaN / inf
print("各列缺失值数量:\n", data.isna().sum())
print("是否包含 inf:", np.isinf(data.values).any())

# 替换 inf -> NaN，再删除缺失值
data = data.replace([np.inf, -np.inf], np.nan).dropna()

print("清理后数据维度:", data.shape)

# ========== Step 3. 特征选择 ==========
X1 = data.iloc[:, 1:].values  # 第2列到最后一列作为特征
y = data.iloc[:, 0].values    # 第1列作为目标变量

# 建议不要人为添加常数列 (RandomForest 不需要截距)
X = X1

# ========== Step 4. 标准化 ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 保存反归一化所需的均值和标准差
X_mean = scaler.mean_
X_std = scaler.scale_

# ========== Step 5. 划分训练测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# 初始化 GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=5,  # 5折交叉验证
                           n_jobs=1,
                           verbose=2)

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 打印最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数训练模型
best_rf_model = grid_search.best_estimator_

# 在测试集和训练集上进行预测
y_test_pred = best_rf_model.predict(X_test)
y_train_pred = best_rf_model.predict(X_train)

# 计算R2值
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 计算其他评估指标
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_ev = explained_variance_score(y_train, y_train_pred)
test_ev = explained_variance_score(y_test, y_test_pred)

# 计算 NSE, NRMSE, 和 Relative Error
def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

def nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return rmse / (np.max(y_true) - np.min(y_true))

def relative_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

train_nse = nse(y_train, y_train_pred)
test_nse = nse(y_test, y_test_pred)

train_nrmse = nrmse(y_train, y_train_pred)
test_nrmse = nrmse(y_test, y_test_pred)

train_rel_error = relative_error(y_train, y_train_pred)
test_rel_error = relative_error(y_test, y_test_pred)

# 将评估指标名称和对应的值定义为字典
metric_names = ['R2', 'MAE', 'RMSE', 'Explained Variance', 'NSE', 'NRMSE', 'Relative Error']
train_metrics = [train_r2, train_mae, train_rmse, train_ev, train_nse, train_nrmse, train_rel_error]
test_metrics = [test_r2, test_mae, test_rmse, test_ev, test_nse, test_nrmse, test_rel_error]

# 创建评估指标 DataFrame
evaluation_data = pd.DataFrame({
    'Metric': metric_names,
    'Training': train_metrics,
    'Testing': test_metrics
})

# 保存到 Excel 文件
with pd.ExcelWriter("output_rf_with_combined_train_test_ZHOU0903.xlsx") as writer:
    pd.DataFrame({'y_test': y_test, 'y_pred': y_test_pred}).to_excel(writer, sheet_name='Test Results', index=False)
    pd.DataFrame({'y_train': y_train, 'y_pred': y_train_pred}).to_excel(writer, sheet_name='Train Results', index=False)
    evaluation_data.to_excel(writer, sheet_name='Evaluation Metrics', index=False)

# ========== 计算 SHAP 值 ==========
# 特征名称（不要加 Intercept，保持和 X_train 一致）
feature_names = list(data.columns[1:])

explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(X_train)

# 检查维度匹配
print("feature_names 数量:", len(feature_names))
print("shap_values 数量:", np.mean(np.abs(shap_values), axis=0).shape)

# 构建 SHAP 重要性 DataFrame
shap_importance = np.mean(np.abs(shap_values), axis=0)
feature_importance = pd.DataFrame({
    'Feature': feature_names[:len(shap_importance)],
    'SHAP Importance': shap_importance
})

# 排序
feature_importance = feature_importance.sort_values(
    by='SHAP Importance', ascending=False
).reset_index(drop=True)

print("Top features by SHAP importance:\n", feature_importance.head())

# 获取前3个重要特征
top_features = feature_importance.loc[:2, 'Feature'].values


# 反标准化X_train
X_train_original = X_train * X_std + X_mean  # 将归一化数据反归一化

# 使用反标准化后的数据绘制 SHAP 总结图
shap.summary_plot(shap_values, X_train_original, feature_names=feature_names)

# 保存 SHAP 总结图
shap.summary_plot(shap_values, X_train_original, feature_names=feature_names, show=False)
plt.savefig("shap_summary_plot_original_scale.png", dpi=300)
plt.show()

# 绘制 SHAP 总结的瀑布图
shap.summary_plot(shap_values, X_train_original, plot_type="bar", feature_names=feature_names)
plt.savefig("shap_bar_plot_summary.png", dpi=300)
plt.show()


# 按照 SHAP 重要性降序排序，获取排名前3的特征
top_features = feature_importance.sort_values(by='SHAP Importance', ascending=False).head(3)['Feature'].values

# 自动绘制 SHAP 依赖图的交互
# 第一幅图：排名第1和第2的交互
shap.dependence_plot(top_features[0], shap_values, X_train_original, feature_names=feature_names, interaction_index=top_features[1])
plt.savefig("shap_dependence_plot_interaction_1_2.png", dpi=300)
plt.show()

# 第二幅图：排名第2和第3的交互
shap.dependence_plot(top_features[1], shap_values, X_train_original, feature_names=feature_names, interaction_index=top_features[2])
plt.savefig("shap_dependence_plot_interaction_2_3.png", dpi=300)
plt.show()

# 第三幅图：排名第3和第1的交互
shap.dependence_plot(top_features[2], shap_values, X_train_original, feature_names=feature_names, interaction_index=top_features[0])
plt.savefig("shap_dependence_plot_interaction_3_1.png", dpi=300)
plt.show()

# 保存 SHAP 依赖图
plt.savefig("shap_dependence_plot_original_scale_3.png", dpi=300)
plt.show()


# 绘制1:1线图和误差分布
def plot_1to1_combined(y_train, y_train_pred, y_test, y_test_pred, train_r2, test_r2, filename):
    fig = plt.figure(figsize=(12, 12))

    # 使用gridspec设置子图位置
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 3, height_ratios=[1, 5, 1], width_ratios=[1, 5, 1])

    # 主绘图区域（1:1线图）
    ax_main = fig.add_subplot(gs[1, 1])
    ax_main.scatter(y_train, y_train_pred, alpha=0.6, edgecolor='k', label=f'Training Set ($R^2$ = {train_r2:.3f})',
                    s=50,
                    color='#3575A2')
    ax_main.scatter(y_test, y_test_pred, alpha=0.6, edgecolor='k', label=f'Test Set ($R^2$ = {test_r2:.3f})', s=50,
                    color='#E1822B')
    ax_main.plot([min(np.concatenate((y_train, y_test))), max(np.concatenate((y_train, y_test)))],
                 [min(np.concatenate((y_train, y_test))), max(np.concatenate((y_train, y_test)))],
                 color='black', linestyle='--', label='1:1 Line')
    ax_main.set_xlabel('Actual Values', fontsize=16)
    ax_main.set_ylabel('Predicted Values', fontsize=16)
    ax_main.set_title("1:1 Line Plot for Training and Test Sets", fontsize=18)
    ax_main.legend(loc='lower right', fontsize=14)
    ax_main.grid(True)
    ax_main.axis('equal')

    # 设置刻度标签字体大小
    ax_main.tick_params(axis='both', which='major', labelsize=12)  # 修改主刻度标签的字体大小

    # 上方的真实值分布柱状图
    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.hist(y_train, bins=20, alpha=0.6, label="Training Set", color='#3575A2', edgecolor='black')
    ax_top.hist(y_test, bins=20, alpha=0.6, label="Test Set", color='#E1822B', edgecolor='black')
    ax_top.set_ylabel('Frequency', fontsize=14)
    ax_top.set_title('Distribution of Actual Values', fontsize=14)
    ax_top.legend(loc='upper right', fontsize=10)

    # 设置刻度标签字体大小
    ax_top.tick_params(axis='both', which='major', labelsize=12)

    # 右侧的预测值分布柱状图
    ax_right = fig.add_subplot(gs[1, 2])
    ax_right.hist(y_train_pred, bins=20, alpha=0.6, label="Training Set", color='#3575A2', edgecolor='black',
                  orientation='horizontal')
    ax_right.hist(y_test_pred, bins=20, alpha=0.6, label="Test Set", color='#E1822B', edgecolor='black',
                  orientation='horizontal')
    ax_right.set_xlabel('Frequency', fontsize=14)
    ax_right.set_title('Distribution of Predicted Values', fontsize=14)
    ax_right.legend(loc='upper right', fontsize=10)

    # 设置刻度标签字体大小
    ax_right.tick_params(axis='both', which='major', labelsize=12)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


# 绘制合并的训练集和测试集的1:1线图
plot_1to1_combined(y_train, y_train_pred, y_test, y_test_pred, train_r2, test_r2, "combined_train_test_1to1_plot.png")