# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:42:59 2019

@author: vmuser20
"""

import sys
sys.path.append('..')
from backtest import Backtest
import pandas as pd
import warnings
import os

warnings.simplefilter('ignore', RuntimeWarning)

if __name__ == '__main__':
    
    p_dir = os.path.abspath(os.path.join(os.getcwd(), './tests/'))
    Factor = pd.read_csv('{}/test_factor.csv'.format(p_dir),
                         index_col = 0, header = 0, parse_dates=True)
    temp = Backtest(temp_data=True)
    temp.TargetOn(Factor, scale_method='standardize')        
    result = temp.Backtest(w_limit=0.05, risk_control=True)
    summary = temp.Summary(result)
