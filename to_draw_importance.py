import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from time import sleep


# 导入总模型
def draw(model_path = r'SA_ST_square_relation_model.pkl'
         ):
    # 读取保存的模型
    multi_model = joblib.load(model_path)
    # 提取其中的元模型
    estimators = multi_model.estimators_

    model = xgb.plot_importance(estimators[0])
    plt.savefig(f'Param ')
    plt.close()


if __name__ == '__main__':
    draw()