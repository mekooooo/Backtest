import sys
sys.path.append('..')
from backtest import Backtest
import pandas as pd
import warnings
import os
import glob
import numpy as np
import scipy.io as mread

if __name__ == '__main__':
    # p_dir = os.path.abspath(os.path.join(os.getcwd(), './tests/'))
    # Factor = pd.read_csv('{}/test_factor.csv'.format(p_dir), index_col=0, header=0, parse_dates=True)

    pred_path = 'D:/2018/pinpoint/ml/py3/lightGBM/xgb_tune/'
    pred_files = np.array([os.path.basename(i) for i in glob.glob(pred_path + '*.csv')])
    print(len(pred_files))

    pred_date_data = pd.read_csv(pred_path + pred_files[0], delimiter=",", header=None)
    pred_date = pd.to_datetime([str(int(x)) for x in pred_date_data[500:][0]])

    path = 'D:/2018/pinpoint/data/factors'
    mat_raw = mread.loadmat('%s/ticker' % path)
    pred_ticker = mat_raw['ticker']
    pred_ticker = [str(x[0][0]) for x in pred_ticker]


    dictfilePath = 'D:/2018/pinpoint/ml/py3/lightGBM/xgb_tune/'

    temp = Backtest(temp_data=True)
    for i in pred_files:
        print(i)
        pred_temp = pd.read_csv(pred_path + i, delimiter=",", header=None)
        fileName = '{}'.format(i[41:-4])
        pred_temp = pred_temp.values
        pred_temp_file = pd.DataFrame(pred_temp[500:, 1:], index=pred_date, columns=pred_ticker)
        temp.TargetOn(pred_temp_file, scale_method=None)
        result = temp.Backtest(w_limit=0.05, top_num=200, risk_control=True, fir_line=-0.04, sec_line=-0.055,
                               fir_recov=-0.01, sec_recov=-0.03)
        summary = temp.Summary(result, plot=False)
        summary = (summary.astype(float)*100).round(2)
        summary.to_csv('{}xgb_summary_{}.txt'.format(dictfilePath, fileName), sep = ' ', mode='a')
        del summary, result, pred_temp_file


    pred = 0
    for i in pred_files:
        pred_temp = pd.read_csv(pred_path + i, delimiter=",", header=None)
        pred = pred_temp + pred

    pred = pred.values
    pred_final = pd.DataFrame(pred[500:,1:], index=pred_date, columns=pred_ticker)

    temp = Backtest(temp_data=True)
    temp.TargetOn(pred_final, scale_method=None)
    result = temp.Backtest(w_limit=0.05, top_num=200, risk_control=True, fir_line=-0.04, sec_line=-0.055, fir_recov=-0.01, sec_recov=-0.03)
    summary = temp.Summary(result)

