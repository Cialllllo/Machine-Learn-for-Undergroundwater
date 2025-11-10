import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import shap
import time

# ========= Step 1. 读取 Excel 数据 =========
excel_file = "DATA-zhou2.xlsx"
data = pd.read_excel(excel_file)

# ========= Step 2. 数据清洗（稳定性增强） =========
# 选取特征和目标
X1 = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 添加常数项列
X = np.c_[np.ones((X1.shape[0], 1)), X1]

# 合并方便处理缺失值
df_all = pd.DataFrame(X, columns=["Intercept"] + list(data.columns[1:]))
df_all.insert(0, "target", y)

# 替换 inf/-inf 为 NaN
df_all.replace([np.inf, -np.inf], np.nan, inplace=True)

# 打印缺失统计
print("缺失值统计（清洗前）:\n", df_all.isna().sum())

# 均值填充缺失值
imputer = SimpleImputer(strategy="mean")
df_all = pd.DataFrame(imputer.fit_transform(df_all), columns=df_all.columns)

# 拆分回 X 和 y
y = df_all["target"].values
X = df_all.drop(columns=["target"]).values

# ========= Step 3. 标准化 =========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 保存反标准化参数
X_mean = scaler.mean_
X_std = scaler.scale_

# ========= Step 4. 划分数据集 =========
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ========= Step 5. 定义网格搜索参数 =========
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# ========= Step 6. 训练模型 =========
start_time = time.time()
grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=1,
    verbose=2
)
grid_search.fit(X_train, y_train)
end_time = time.time()

print(f"GridSearchCV training time: {end_time - start_time:.2f} seconds")
print("Best parameters found:", grid_search.best_params_)

best_gbr_model = grid_search.best_estimator_

# ========= Step 7. 预测 =========
y_test_pred = best_gbr_model.predict(X_test)
y_train_pred = best_gbr_model.predict(X_train)

# ========= Step 8. 评估指标 =========
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_ev = explained_variance_score(y_train, y_train_pred)
test_ev = explained_variance_score(y_test, y_test_pred)

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

# 保存评估结果
metric_names = ['R2', 'MAE', 'RMSE', 'Explained Variance', 'NSE', 'NRMSE', 'Relative Error']
train_metrics = [train_r2, train_mae, train_rmse, train_ev, train_nse, train_nrmse, train_rel_error]
test_metrics = [test_r2, test_mae, test_rmse, test_ev, test_nse, test_nrmse, test_rel_error]
evaluation_data = pd.DataFrame({'Metric': metric_names, 'Training': train_metrics, 'Testing': test_metrics})

with pd.ExcelWriter("output_gbr_with_combined_train_test_zhou0906.xlsx") as writer:
    pd.DataFrame({'y_test': y_test, 'y_pred': y_test_pred}).to_excel(writer, sheet_name='Test Results', index=False)
    pd.DataFrame({'y_train': y_train, 'y_pred': y_train_pred}).to_excel(writer, sheet_name='Train Results', index=False)
    evaluation_data.to_excel(writer, sheet_name='Evaluation Metrics', index=False)

# ========= Step 9. SHAP 分析 =========
feature_names = ["Intercept"] + list(data.columns[1:])
explainer = shap.TreeExplainer(best_gbr_model)
shap_values = explainer.shap_values(X_train)

X_train_original = X_train * X_std + X_mean

shap.summary_plot(shap_values, X_train_original, feature_names=feature_names, plot_size=(10, 6))
plt.savefig("shap_summary_plot_original_scale.png", dpi=300)
plt.show()

shap.summary_plot(shap_values, X_train_original, plot_type="bar", feature_names=feature_names)
plt.savefig("shap_bar_plot_summary.png", dpi=300)
plt.show()

# ========= Step 10. SHAP Top 3 交互图 =========
shap_importance = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({'Feature': feature_names, 'SHAP Importance': shap_importance})
top_features = feature_importance.sort_values(by='SHAP Importance', ascending=False).head(3)['Feature'].values

shap.dependence_plot(top_features[0], shap_values, X_train_original, feature_names=feature_names, interaction_index=top_features[1])
plt.savefig("shap_dependence_plot_interaction_1_2.png", dpi=300)
plt.show()

shap.dependence_plot(top_features[1], shap_values, X_train_original, feature_names=feature_names, interaction_index=top_features[2])
plt.savefig("shap_dependence_plot_interaction_2_3.png", dpi=300)
plt.show()

shap.dependence_plot(top_features[2], shap_values, X_train_original, feature_names=feature_names, interaction_index=top_features[0])
plt.savefig("shap_dependence_plot_interaction_3_1.png", dpi=300)
plt.show()

# ========= Step 11. 1:1 线图 =========
def plot_1to1_combined(y_train, y_train_pred, y_test, y_test_pred, train_r2, test_r2, filename):
    fig = plt.figure(figsize=(12, 12))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 3, height_ratios=[1, 5, 1], width_ratios=[1, 5, 1])

    ax_main = fig.add_subplot(gs[1, 1])
    ax_main.scatter(y_train, y_train_pred, alpha=0.6, edgecolor='k', label=f'Training Set ($R^2$ = {train_r2:.3f})', s=50, color='#3575A2')
    ax_main.scatter(y_test, y_test_pred, alpha=0.6, edgecolor='k', label=f'Test Set ($R^2$ = {test_r2:.3f})', s=50, color='#E1822B')
    all_vals = np.concatenate((y_train, y_test))
    ax_main.plot([min(all_vals), max(all_vals)], [min(all_vals), max(all_vals)], color='black', linestyle='--', label='1:1 Line')
    ax_main.set_xlabel('Actual Values', fontsize=16)
    ax_main.set_ylabel('Predicted Values', fontsize=16)
    ax_main.set_title("1:1 Line Plot for Training and Test Sets", fontsize=18)
    ax_main.legend(loc='lower right', fontsize=14)
    ax_main.grid(True)
    ax_main.axis('equal')
    ax_main.tick_params(axis='both', which='major', labelsize=12)

    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.hist(y_train, bins=20, alpha=0.6, label="Training Set", color='#3575A2', edgecolor='black')
    ax_top.hist(y_test, bins=20, alpha=0.6, label="Test Set", color='#E1822B', edgecolor='black')
    ax_top.set_ylabel('Frequency', fontsize=14)
    ax_top.set_title('Distribution of Actual Values', fontsize=14)
    ax_top.legend(loc='upper right', fontsize=10)
    ax_top.tick_params(axis='both', which='major', labelsize=12)

    ax_right = fig.add_subplot(gs[1, 2])
    ax_right.hist(y_train_pred, bins=20, alpha=0.6, label="Training Set", color='#3575A2', edgecolor='black', orientation='horizontal')
    ax_right.hist(y_test_pred, bins=20, alpha=0.6, label="Test Set", color='#E1822B', edgecolor='black', orientation='horizontal')
    ax_right.set_xlabel('Frequency', fontsize=14)
    ax_right.set_title('Distribution of Predicted Values', fontsize=14)
    ax_right.legend(loc='upper right', fontsize=10)
    ax_right.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

plot_1to1_combined(y_train, y_train_pred, y_test, y_test_pred, train_r2, test_r2, "combined_train_test_1to1_plot.png")
