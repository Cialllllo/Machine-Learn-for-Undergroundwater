from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as GBM
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
def data_processing(name = 'GW_RCHG',func = 'cubic'):
    # 读取准备好的特征
    features = pd.read_csv(r'features.csv')
    data_label = pd.read_csv(fr'.\label\{name}_{func}_relation.csv')  # 这里需要改成实际的标签文件名

    # 正确做法：先划分，再分别标准化
    X_train, X_test, y_train, y_test = train_test_split(features, data_label, test_size=0.3, random_state=30)

    # 对标签标准化
    y_standard = StandardScaler()
    y_train = y_standard.fit_transform(y_train)  # 只用训练集拟合
    y_test = y_standard.transform(y_test)  # 用训练集的参数转换测试集
    return X_train,X_test,y_train,y_test


def rf_processing(relation = 'GW_RCHG',func = 'cubic'):
    with open(f'./RF_params/{relation}_{func}_params.json', 'r', encoding='utf-8') as file:
        # 使用 json.load() 方法加载文件内容并转换为 Python 对象
        params = json.load(file)
    # 返回最佳参数的模型
    model = RandomForestRegressor(**params)
    return model

def xgb_processing(relation = 'GW_RCHG',func = 'cubic'):
    params = joblib.load(f'./XGB_model/{relation}_{func}_relation_model.pkl').get_params()
    base_model = xgb.XGBRegressor()
    model = MultiOutputRegressor(base_model,n_jobs=-1).set_params(**params)
    return model

def gbr_processing(relation = 'GW_RCHG',func = 'cubic'):
    with open(f'./GBR_params/{relation}_{func}_params.json', 'r', encoding='utf-8') as file:
        params = json.load(file)
    base_model = GradientBoostingRegressor()
    # 修正参数传递
    model = MultiOutputRegressor(base_model, n_jobs=-1)
    model.set_params(**params)  # 正确设置参数的方式
    return model

def gbm_processing(relation = 'GW_RCHG',func = 'cubic'):
    with open(f'./LightGBM_params/{relation}_{func}_params.json', 'r', encoding='utf-8') as file:
        params = json.load(file)
    base_model = GBM.LGBMRegressor()
    # 修正参数传递
    model = MultiOutputRegressor(base_model, n_jobs=-1)
    model.set_params(**params)  # 正确设置参数的方式
    return model
