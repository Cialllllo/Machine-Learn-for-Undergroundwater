import modelprocessing
import pandas as pd

if __name__ == '__main__':
    relation = 'SHGWT'
    func = 'cubic'
    modelprocessing.multioutput_rfecv(model_process=modelprocessing.xgb_processing,relation=relation,func=func)
