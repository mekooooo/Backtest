# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:06:18 2019

@author: vmuser20
"""

import pandas as pd
import numpy as np
import h5py
from matplotlib import pyplot as plt
from progressbar import ProgressBar
import io
import warnings

warnings.simplefilter('ignore', RuntimeWarning)

temp_dir = './/'
data_dir = '//'
h5_dir = '{}//'.format(data_dir)

def to_hdf(data, fname, key, string=False):
    h5 = h5py.File(fname, 'a')
    if key in list(h5.keys()):
        del h5[key]
    if string:
        h5.create_dataset(key, data=np.array(np.array(data,
                                                      dtype=np.dtype(str)), 
    dtype=h5py.special_dtype(vlen=str)), dtype=h5py.special_dtype(vlen=str))
    else:
        h5.create_dataset(key, data=data)
    h5.close()

def to_hdf_df(df, fname, key, string=False):
    to_hdf(df, fname, key, string=string)
    to_hdf(df.index, fname, 'Dates', string=True)
    to_hdf(df.columns, fname, 'Tickers', string=True)

def from_hdf(fname, key):
    h5 = h5py.File(fname, 'r')
    data = h5[key].value
    h5.close()
    return data

def from_hdf_df(fname, key):
    return pd.DataFrame(from_hdf(fname, key),
                        index=pd.to_datetime(from_hdf(fname, 'Dates')),
                        columns=from_hdf(fname, 'Tickers'))

class Backtest:
    
    def __init__(self, temp_data=False):
        """
        Initialize the class by importing below data,
            - Dates
            - Tickers
            - CSI500 Close
            - CSI500 Open
            - Close
            - High
            - Low
            - VWAP
            - Market Value
            - Check Matrix
            - Free Turnover
            - Free Market Value
        
        Parameters
        ----------
        temp_data : boolean, indicates whether importing data from temporary
                    data directory.
        
        Notes
        -----
            - CSI500 Open is used for futures trading, High and Low are used
              to indicate whether the stock can be longed this day.
            
            - At the present stage, data are imported from .mat files.
            
            - The Check matrix can prevent the factor from buying low value
              stocks.
        """
        data_dir = temp_dir if temp_data else h5_dir
        self.data_dir = data_dir
        
        if temp_data:
            self.Dates = pd.to_datetime([str(x[0][0]) for x in io.loadmat('{}date.mat'.format(data_dir))['date']])

            self.Tickers = [str(x[0][0]) for x in io.loadmat('{}ticker.mat'.format(data_dir))['ticker']]
            
            self.StartDate = self.Dates[0]
            self.EndDate = self.Dates[-1]
            
            CSI500 = pd.Series(io.loadmat('{}csi500_close.mat'.format(data_dir))['hs300_c'].reshape([-1]),
                    index=self.Dates)
            CSI500_open = pd.Series(io.loadmat(
                    '{}csi500_open.mat'.format(data_dir))['hs300_o'].reshape([-1]),
                    index=self.Dates)
            self.Close = pd.DataFrame(io.loadmat(
                    '{}data_close_price.mat'.format(data_dir))['close_price'].T,
                    index=self.Dates, columns=self.Tickers)
            self.VWAP = pd.DataFrame(io.loadmat(
                    '{}data_vwap_price.mat'.format(data_dir))['vwap_price'].T,
                    index=self.Dates, columns=self.Tickers)
            self.Check = pd.DataFrame(io.loadmat(
                    '{}data_tickercheck.mat'.format(data_dir))['tickercheck'].T,
                    index=self.Dates, columns=self.Tickers)
            self.MV = pd.DataFrame(io.loadmat(
                    '{}data_mv.mat'.format(data_dir))['mv'].T,
                    index=self.Dates, columns=self.Tickers)
            
            self.Close[self.Close == 0] = np.nan
            self.VWAP[self.VWAP == 0] = np.nan
            self.Close_fill = self.Close.ffill(axis=0)
            self.Ret = self.Close_fill/self.Close_fill.shift(axis=0)
            CSI500[CSI500 == 0] = np.nan
            CSI500_open[CSI500_open == 0] = np.nan                       
            self.CSI500 = CSI500.ffill().bfill()
            self.CSI500_open = CSI500_open.ffill().bfill()
            
            # Go up staying & Fall staying
            High = io.loadmat('{}data_high_price.mat'.format(data_dir))['high_price'].T
            Low = io.loadmat('{}data_low_price.mat'.format(data_dir))['low_price'].T
            cant = High == Low
            self.CantBuy = cant*(Low/self.Close_fill.shift(axis=1)-1 > 0.09)
            self.CantSell = cant*(High/self.Close_fill.shift(axis=1)-1 < -0.09)
            
            # Liquidity limit
            fturn = pd.DataFrame(io.loadmat(
                    '{}data_fturn.mat'.format(data_dir))['fturn'].T / 100,
                    index=self.Dates, columns=self.Tickers)
            fmv = pd.DataFrame(io.loadmat(
                    '{}data_fmv.mat'.format(data_dir))['fmv'].T,
                    index=self.Dates, columns=self.Tickers)
            self.Liq_top = (fturn * fmv).rolling(5, axis=0).mean()
            L_mask = pd.isnull(self.Liq_top)
            self.Liq_top[L_mask] = (fturn * fmv)[L_mask]
            
        else:
            self.Dates = pd.to_datetime(from_hdf('{}/Close.h5'.format(data_dir),
                                                 'Dates'))

            self.Tickers = from_hdf('{}/Close.h5'.format(data_dir), 'Tickers')
            
            self.StartDate = self.Dates[0]
            self.EndDate = self.Dates[-1]
            
            CSI500 = pd.Series(from_hdf('{}/CSI500.h5'.format(data_dir),
                                        'Close'),
                               index=pd.to_datetime(from_hdf(
                               '{}/CSI500.h5'.format(data_dir), 'Dates')))
            CSI500_open = pd.Series(from_hdf('{}/CSI500.h5'.format(data_dir),
                                             'Open'),
                                    index=pd.to_datetime(from_hdf(
                                    '{}/CSI500.h5'.format(data_dir), 'Dates')))
            self.Close = from_hdf_df('{}/Close.h5'.format(data_dir), 'Forward')
            self.VWAP = from_hdf_df('{}/VWAP.h5'.format(data_dir), 'Forward')
            
            # Check
#            self.Check = from_hdf_df('{}/Check.h5'.format(data_dir), 'Check')
            ST = from_hdf_df('{}/Check.h5'.format(data_dir), 'ST').fillna(1)
            NL = from_hdf_df('{}/Check.h5'.format(data_dir), 'NewListed_6m')
            NL[NL == 0] = np.nan
            NL = NL.bfill(axis=0).fillna(0)
            Sus = from_hdf_df('{}/Check.h5'.format(data_dir), 'Suspend').fillna(1)
            self.Check = ST + Sus + Sus.shift(axis=0) + NL

            self.Check[~(self.Check == 0)] = 1
        

            self.MV = from_hdf_df('{}/MV.h5'.format(data_dir), 'Raw')
            
            self.Close[self.Close == 0] = np.nan
            self.VWAP[self.VWAP == 0] = np.nan
            self.Close_fill = self.Close.ffill(axis=0)
            self.Ret = self.Close_fill/self.Close_fill.shift(axis=0)
            CSI500[CSI500 == 0] = np.nan
            CSI500_open[CSI500_open == 0] = np.nan                       
            self.CSI500 = CSI500.ffill().bfill()
            self.CSI500_open = CSI500_open.ffill().bfill()
            
            self.Tickers = np.array(sorted(list(set(self.Close.columns) &
                                                set(self.MV.columns))))
            
            # Go up staying & Fall staying
            High = from_hdf('{}/High.h5'.format(data_dir), 'Forward')
            Low = from_hdf('{}/Low.h5'.format(data_dir), 'Forward')
            cant = High == Low
            self.CantBuy = cant*(Low/self.Close_fill.shift(axis=0)-1 > 0.09)
            self.CantSell = cant*(High/self.Close_fill.shift(axis=0)-1 < -0.09)
            
            # Liquidity limit
            fturn = from_hdf_df('{}/fTurnover.h5'.format(data_dir), 'Raw') / 100.0
            fmv = from_hdf_df('{}/fMV.h5'.format(data_dir), 'Raw')
            
            self.Liq_top = (fturn * fmv).rolling(5, axis=0).mean()
            L_mask = pd.isnull(self.Liq_top)
            self.Liq_top[L_mask] = (fturn * fmv)[L_mask]
            
            if self.StartDate < self.CSI500.index[0]:
                self.StartDate = self.CSI500.index[0]
                
            if self.EndDate > self.CSI500.index[-1]:
                self.EndDate = self.CSI500.index[-1]
                
            self.Members = {'CSI300': {},
                            'CSI500': {},
                            'CSI1000': {}}
            
            for t_index in ['CSI300', 'CSI500', 'CSI1000']:
                t_file = h5py.File('{}/{}_Member.h5'.format(data_dir,
                                   t_index), 'r')
                keys = list(t_file.keys())
                t_file.close()
                for key in keys:
                    self.Members[t_index][key] = [x[0] for x in
                                 from_hdf('{}/{}_Member.h5'.format(data_dir,
                                          t_index), key)]
            
            self.Industry = from_hdf_df('{}/Industry_Citics.h5'.format(data_dir),
                                        'Citics').reindex(index=self.Dates,
                                                columns=self.Tickers).ffill(axis=0).fillna('None')

#Members
#MV = from_hdf_df('D:/share/Data/Wind/MV.h5','Raw')
#temp = MV[Members['CSI300']['201901']]['20190101':].iloc[0]
#
#last_q = -1
#mean = []
#std = []
#qt = pd.Series(temp.quantile([0.1*x for x in range(1,10)]).values)
#for q in temp.quantile([0.1*x for x in range(1,10)]):
#    mask = (temp >= last_q) & (temp < q)
#    mean.append(temp[mask].mean())
#    std.append(temp[mask].std())
#    temp.loc[mask] = temp[mask]
#mean.append(temp[temp >= q].mean())
#std.append(temp[temp >= q].std())
#pd.concat([pd.Series(mean) + pd.Series(std), pd.Series(mean),  pd.Series(mean) - pd.Series(std), pd.Series(std)], axis=1, keys=['upper', 'mean', 'lower', 'std'])

    
    def TargetOn(self, Factor, scale_method=None):
        """
        This function can adjust dates and tickers of data and factor to
        the same sets. And factor will be pre-scale before the backtest
        is run.
        
        Parameters
        ----------
        Factor : pandas.DataFrame, with dates as index and tickers as columns.
        
        scale_method : str, default None, only 'standardize' and 'normalize'
                       can be chosen.
        
        Notes
        -----
            - Scale or not does not affect backtest
              result, but may have some impact on some weighted
              methods planned to update in the future.
              
            - If tickers and dates are not matched, the function will raise
              a warning.
        """
        # Input control
        if not isinstance(Factor, pd.DataFrame):
            raise TypeError('Input factor should be pandas.DataFrame.')
        
        # Save original factor score
        self.Factor = Factor
        self.Factor.index = pd.to_datetime(self.Factor.index)
        # Adjust time
        adjust_flag = False
        start_date = Factor.index[0]
        end_date = Factor.index[-1]
        
        StartDate = self.StartDate
        EndDate = self.EndDate
        
        if start_date != StartDate:
            SDate = start_date if start_date > StartDate else StartDate
            warnings.warn(
            'Start date of input factor is {}, '.format(start_date.date()) +
            'but start date of data is {}.'.format(StartDate.date()))
            adjust_flag = True
        else:
            SDate = start_date
            
        if end_date != EndDate:
            EDate = end_date if end_date < EndDate else EndDate
            warnings.warn(
                'End date of input factor is {}, '.format(end_date.date()) +
                'but end date of data is {}.'.format(EndDate.date()))
            adjust_flag = True
        else:
            EDate = end_date
        
        # Adjust tickers
        cp_set1 = set(Factor.columns) - set(self.Tickers)
        cp_set2 = set(self.Tickers) - set(Factor.columns)
        if len(cp_set1) > 0:
            warnings.warn(
                    'Tickers {} in factor are not in data.'.format(cp_set1))
            adjust_flag = True
        elif len(cp_set2) > 0:
            warnings.warn(
                    'Tickers {} in data are not in factor.'.format(cp_set2))
            adjust_flag = True
        
        if adjust_flag:
            self.tickers = sorted(list(set(Factor.columns) 
                                       & set(self.Tickers)))
            self.csi500 = self.CSI500[SDate:EDate].values
            self.csi500_open = self.CSI500_open[SDate:EDate].values
            self.close = self.Close[SDate:EDate][self.tickers].values
            self.vwap = self.VWAP[SDate:EDate][self.tickers].values
            self.check = self.Check[SDate:EDate][self.tickers].values
            self.mv = self.MV[SDate:EDate][self.tickers].values
            self.cant_buy = self.CantBuy[SDate:EDate][self.tickers].values
            self.cant_sell = self.CantSell[SDate:EDate][self.tickers].values
            self.close_fill = self.Close_fill[SDate:EDate][self.tickers].values
            self.liq_top = self.Liq_top[SDate:EDate][self.tickers].values
            self.dates = self.CSI500[SDate:EDate].index
            factor = Factor[SDate:EDate][self.tickers]
            # Insert missing trading days in factor
            if factor.shape[0] != len(self.dates):
                factor = factor.reindex(self.dates)
                
        else:
            self.tickers = self.Tickers
            self.csi500 = self.CSI500.values
            self.csi500_open = self.CSI500_open.values
            self.close = self.Close.values
            self.vwap = self.VWAP.values
            self.check = self.Check.values
            self.mv = self.MV.values
            self.cant_buy = self.CantBuy.values
            self.cant_sell = self.CantSell.values
            self.close_fill = self.Close_fill.values
            self.liq_top = self.Liq_top.values
            self.dates = self.CSI500.index
            factor = Factor.copy()
            
        # Scale input factor
        if scale_method == 'standardize':
            factor = factor.sub(factor.mean(axis=1),axis=0).div(
                            factor.std(axis=1), axis=0)
        elif scale_method == 'normalize':
            temp_min = factor.min(axis=1)
            factor = factor.sub(temp_min, axis=0).div(
                            factor.max(axis=1)-temp_min, axis=0)
        elif scale_method == 'linear':
            for i in range(len(self.dates)):
                sort_c = factor.iloc[i,:].sort_values()
                diff = sort_c.diff()
                d_min = diff.min()
                new_diff = diff.sub(d_min).div(diff.max() - d_min)
                factor.iloc[i,:] = (new_diff.cumsum())[self.tickers]
        self.factor = factor.values
    
    def Backtest(self, init_cap=100000000, mv_weighted=False, top_num=100,
                 risk_control=False, fir_line=-0.02, sec_line=-0.04,
                 fir_recov=-0.01, sec_recov=-0.03, w_limit=0.05, 
                 pos_ratio=0.8, trade_complt=1, liq_limit=0.1,
                 turn_bot=0.1, cash_limit=None, commission=0.0002,
                 stamp_duty=0.001, mngm_fee=0.0001, w_method=None):
        """
        Main backtest function in this class, which has replicated the whole
        trading process.
        
        Parameters
        ----------
        init_cap : integer, initial capital for backtest.
        
        mv_weighted : boolean, whether use free market value to initialize
                      weights. If False, the weight matrix will be initialized
                      as equally weighted.
                      
        top_num : If integer, # of stocks on top will be picked. Else if float
                  in (0, 1], top % quantile will be picked. 
        
        risk_control : boolean, whether to control risk by two drawdown line.
        
        fir_line : float, maximum drawdown for the first risk control line.
        
        sec_line : float, maximum drawdown for the second risk control line.
        
        fir_recov : float, maximum drawdown for the first recovery line.
        
        sec_line : float, maximum drawdown for the second recovery line.
        
        w_limit : float, maximum weight for a single stock.
        
        pos_ratio : float, ratio of holding position and total AUM.
        
        trade_complt : float, proportion of trading volume will be completed.
        
        liq_limit : float, proportion of liquidity yesterday that values of 
                    today's trade orders should not exceed.
        
        turn_bot : float, minimum turnover rate that a trade will be ordered.
        
        cash_limit : float, minimum proportion of total AUM to be held in cash.
        
        commission : float, commission fee rate.
        
        stamp_duty : float, stamp_duty rate, which will be only occurred when
                     selling stocks.
        
        mngm_fee : float, management fee rate.
        
        w_method : str, indicating method that used for picking top stocks.
                   If None(default), stocks whose factor scores are in top
                   top_num will be picked. If 'lookback', stocks will be
                   splited into 10 group according to scores quantile, stocks
                   that are in the group performing best during last 3
                   months will be picked.
        
        
        Notes
        -----
            - For this stage, risk level can be recovered from 2 to 0. But
              liquidity limit and turnover limit can prevent the backtest
              from trading too large volume.
            
            - The output plot is using imperfect function to plot the
              background color, which stands for risk level that day.
              However, the vertical lines are not quite consistent with
              X axis.
            
            - w_method can be designed to be many other forms, and top 
              top_num to pick will be set to be a default setting.
        """
        res = {}
        fill_value = -100
        self.factor[self.check==1] = fill_value
        self.factor[np.isnan(self.factor)] = fill_value
        self.factor[np.isinf(self.factor)] = fill_value
        weights = np.zeros(self.factor.shape)
        valid_count = np.sum(self.factor > fill_value, axis=1)
        top_num_use = top_num
        pb = ProgressBar()
        for i in pb(range(len(self.dates))):
            if i < 1:
                continue
            if valid_count[i-1] < 100:
                continue
            if top_num <= 1:
                top_num_use = int(top_num*valid_count[i-1])
            if w_method is 'lookback':
                if i < 22 + 1:
                    continue
                temp_res = []
                for j in range(22):
                    t_res = []
                    check_pos = np.where(self.check[i-j-2,:] == 1)[0]
                    if len(pd.unique(self.factor[i-j-2,:][check_pos])) <= 1:
                        t_res.append(1)
                        continue
                    categ = pd.qcut(self.factor[i-j-2,:][check_pos], 10,
                                    duplicates='drop').codes
                    for k in range(10):
                        t_res.append(self.Ret.iloc[i-j-1,
                                                   check_pos[categ==k]].mean())
                    temp_res.append(t_res)
                pick = (-np.nanprod(np.array(temp_res), axis=0)).argsort()
                check_pos = np.where(self.check[i-1,:] == 1)[0]
                if len(pd.unique(self.factor[i-1,:][check_pos])) <= 1:
                    continue
                categ = pd.qcut(self.factor[i-1,:][check_pos], 10,
                                duplicates='drop').codes
                if np.sum(categ == pick[0]) < top_num_use:
                    pick = pick[:2]
                else:
                    pick = pick[0]
                pick = check_pos[np.in1d(categ, pick)]
            elif w_method is 'sector':
                t_dates = str(int(self.dates[i-1].year*100+self.dates[i-1].month))
                pick = []
                ind_dict = {}
                cnt_300 = 0
                cnt_500 = 0
                num_300 = int(top_num_use * 0.4)
                num_500 = int(top_num_use * 0.3)
                memb_300 = self.Members['CSI300']
                memb_500 = self.Members['CSI500']
                temp_Ind = self.Industry.loc[self.dates[i]]
                for tic in np.array(self.tickers)[np.argsort(-self.factor[i-1,:])]:
                    if (len(pick) > top_num_use) or (cnt_300 > num_300) or (cnt_500 > num_500):
                        break
                    ind_flag = False
                    index_flag = False
                    # industry
                    if tic not in temp_Ind.index:
                        if ind_dict['None'] < 5:
                            if t_dates in memb_300:
                                if tic in memb_300[t_dates]:
                                    cnt_300 += 1
                                    index_flag = True
                            
                            if t_dates in memb_500:
                                if tic in memb_500[t_dates]:
                                    cnt_500 += 1
                                    index_flag = True
                            if index_flag:
                                ind_flag = True
                    else:
                        if temp_Ind[tic] not in ind_dict:
                            ind_dict[temp_Ind[tic]] = 0
                            if t_dates in memb_300:
                                if tic in memb_300[t_dates]:
                                    cnt_300 += 1
                                    index_flag = True
                            
                            if t_dates in memb_500:
                                if tic in memb_500[t_dates]:
                                    cnt_500 += 1
                                    index_flag = True
                            if index_flag:
                                ind_flag = True
                        if ind_dict[temp_Ind[tic]] < 5:
                            if t_dates in memb_300:
                                if tic in memb_300[t_dates]:
                                    cnt_300 += 1
                                    index_flag = True
                            
                            if t_dates in memb_500:
                                if tic in memb_500[t_dates]:
                                    cnt_500 += 1
                                    index_flag = True
                            if index_flag:
                                ind_flag = True
                    
                    if ind_flag:
                        pick.append(self.tickers.index(tic))
                        ind_dict[temp_Ind[tic]] += 1

                
                # repeat   
                for tic in np.array(self.tickers)[np.argsort(-self.factor[i-1,:])]:
                    tic_pos = self.tickers.index(tic)
                    if tic_pos in pick:
                        continue
                    if len(pick) > top_num_use:
                        break
                    ind_flag = False
                    index_flag = False
                    # industry
                    if tic not in temp_Ind.index:
                        if ind_dict['None'] < 5:
                            ind_flag = True
                    else:
                        if temp_Ind[tic] not in ind_dict:
                            ind_dict[temp_Ind[tic]] = 0
                            ind_flag = True
                        if ind_dict[temp_Ind[tic]] < 5:
                            ind_flag = True
                        
                    
                    if ind_flag:
                        ind_dict[temp_Ind[tic]] += 1
                        pick.append(tic_pos)
                        
            elif w_method is 'index':
                pick = np.argpartition(self.factor[i-1,:],
                                       -top_num_use)[-top_num_use:]
                pick_num = int(top_num_use*0.3)
                t_dates = str(int(self.dates[i-1].year*100+self.dates[i-1].month))
                im = 'CSI300'
                flag = False
                if t_dates in self.Members[im]:
                    temp_memb = self.Members[im][t_dates]
                    flag = len(set(temp_memb) & set(pick)) < int(top_num_use*0.4)
                im = 'CSI500'
                if t_dates in self.Members[im]:
                    temp_memb = self.Members[im][t_dates]
                    flag = len(set(temp_memb) & set(pick)) < pick_num

                if flag:
                    scanned = []
                    pick = []
                    total_r = 1
                    im = 'CSI300'
                    if t_dates in self.Members[im]:
                        i_pos = [self.tickers.index(x) for x in 
                                 self.Members[im][t_dates]
                                 if x in self.tickers]
                        t_pick = np.argpartition(self.factor[i-1, i_pos],
                                                 -int(top_num_use*0.4))[-int(top_num_use*0.4):]
                        pick += np.array(i_pos)[t_pick].tolist()
                        scanned += i_pos
                        total_r -= 0.4
                    im = 'CSI500'
                    if t_dates in self.Members[im]:
                        i_pos = [self.tickers.index(x) for x in
                                 self.Members[im][t_dates] 
                                 if x in self.tickers]
                        t_pick = np.argpartition(self.factor[i-1, i_pos],
                                                 -pick_num)[-pick_num:]
                        pick += np.array(i_pos)[t_pick].tolist()
                        scanned += i_pos
                        total_r -= 0.3
                    i_pos = list(set(range(len(self.tickers))) - 
                                 set(scanned))
                    pick_num = int(top_num_use*total_r)
                    t_pick = np.argpartition(self.factor[i-1, i_pos], 
                                             -pick_num)[-pick_num:]
                    pick += np.array(i_pos)[t_pick].tolist()
            else:
                pick = np.argpartition(self.factor[i-1,:],
                                       -top_num_use)[-top_num_use:]
            weights[i,pick] = np.sqrt(self.mv[i-1,pick]) if mv_weighted else 1
        
        weights /= np.nansum(weights, axis=1).reshape([-1,1])
        weights = np.nan_to_num(weights)
        self.weights = weights
        year_point = np.where(pd.Series([x.year for x in
                                         self.dates]).diff() == 1)[0]
        res['year_point'] = year_point
        risk_level = np.zeros(len(self.dates))
        
        # Pure trading alpha and risk level
        if risk_control: 
            f_pnl = np.zeros(len(self.dates))
            p_ratio = np.zeros(len(self.dates))
            holding = np.zeros(self.factor.shape)
            future_holding = np.zeros(len(self.dates))
            cash_lend = 0
            topV = init_cap
            DD = np.zeros(len(self.dates))
            turnover = np.zeros(len(self.dates))
            cash = np.zeros(len(self.dates))
            aum = np.zeros(len(self.dates))
            for i in range(len(self.dates)):
            
                if i < 1:
                    f_pnl[i] = 0.0
                    p_ratio[i] = 0.0
                    cash[i] = init_cap
                    aum[i] = init_cap
                    continue
        
                cash[i] = cash[i-1]
                holding[i,:] = holding[i-1,:]

                # Lend
                if cash[i] < 0:
                    cash_lend -= cash[i]
                    cash[i] = 0
                
                # Repay
                if cash[i] > 0 and cash_lend > 0:
                    if cash[i] < cash_lend:
                        cash_lend -= cash[i]
                        cash[i] = 0
                    else:
                        cash[i] -= cash_lend
                        cash_lend = 0
                
                if risk_level[i-1] == 0:
                    risk_level[i] = (1*(DD[i-1] <= fir_line) +
                                     1*(DD[i-1] <= sec_line))
                else:
                    risk_level[i] = (risk_level[i-1] -
                                  (risk_level[i-1]-1)*(DD[i-1] >= sec_recov) -
                                  (DD[i-1] >= fir_recov)*1 +
                                  (2-risk_level[i-1])*(DD[i-1] <= sec_line))
                
                weight_v = weights[i,:].copy()
                liq_mask = self.liq_top[i-1,:] / (pos_ratio*aum[i-1])
                if w_limit is not None:
                    liq_mask[~(liq_mask < w_limit)] = w_limit
                    liq_mask[liq_mask == 0] = w_limit
                balance_mask = weight_v > liq_mask
                if balance_mask.any():
                    remain = (weight_v[balance_mask] -
                              liq_mask[balance_mask]).sum()
                    weight_v[balance_mask] = liq_mask[balance_mask]
                    weight_v[~balance_mask] *= (remain /
                            weight_v[~balance_mask].sum() + 1)
                balance_mask = weight_v > liq_mask
                if balance_mask.any():
                    remain = (weight_v[balance_mask] -
                              liq_mask[balance_mask]).sum()
                    weight_v[balance_mask] = liq_mask[balance_mask]
                    weight_v[~balance_mask] *= (remain /
                            weight_v[~balance_mask].sum() + 1)
                weight_v /= weight_v.sum()
                weight_v = np.nan_to_num(weight_v)
                
                h_last = holding[i-1,:] * self.close_fill[i-1,:]
                h_last[np.isnan(self.close_fill[i-1,:])] = 0
                build_speed = pos_ratio * trade_complt
                reb_value = build_speed*aum[i-1]*weight_v - h_last
                reb_value[reb_value < 0] *= ~self.cant_sell[i,reb_value < 0]
                reb_value[reb_value > 0] *= ~self.cant_buy[i,reb_value > 0]
                
                liq_cant = self.liq_top[i-1,:]*liq_limit
                liq_mask = np.abs(reb_value) >= liq_cant
                if np.sum(liq_mask) > 0:
                    reb_value[liq_mask] = (np.sign(reb_value[liq_mask])*
                                           liq_cant[liq_mask])
                
                turn_mask = np.abs(reb_value / h_last) < turn_bot
                reb_value[turn_mask] = 0
                
                future_holding[i] = np.round((reb_value + h_last).sum() / 
                                              self.csi500[i-1] / 200)
                
                # Filtering non-tradable stock today but unknown yesterday
                reb_value[np.isnan(self.vwap[i,:])] = 0
                
                reb_buy = (reb_value > 0) * (h_last > 0)
                reb_sell = (reb_value < 0) * (reb_value + h_last > 0)
                all_buy = (reb_value > 0) * (h_last == 0)
                all_sell = (reb_value < 0) * (reb_value + h_last == 0)
                
                if cash_limit is not None:
                    cash_line = cash_limit * aum[i-1]
                else:
                    cash_line = 0
                    
                # Rebalance sell
                if np.sum(reb_sell) > 0:
                    d_rb_s = np.round(reb_value[reb_sell] /
                                      self.close[i-1,reb_sell],-2)
                    d_rb_s[np.isnan(d_rb_s)] = 0
                    holding[i,reb_sell] += d_rb_s
                    cash[i] += (np.nansum(-d_rb_s * self.vwap[i,reb_sell])
                                * (1-commission-stamp_duty))            
        
                # Repay
                if cash[i] > 0 and cash_lend > 0:
                    if cash[i] < cash_lend:
                        cash_lend -= cash[i]
                        cash[i] = 0
                    else:
                        cash[i] -= cash_lend
                        cash_lend = 0
                
                # Rebalance buy
                if np.sum(reb_buy) > 0 and cash[i] - cash_line > 0:
                    reb_value[reb_buy] /= ((np.sum(reb_value[reb_buy]) /
                             (cash[i] - cash_line)) 
                        if np.sum(reb_value[reb_buy]) > (cash[i] - cash_line)
                        else 1)
                    d_rb_b = np.round(reb_value[reb_buy] /
                                      self.close[i-1,reb_buy],-2)
                    d_rb_b[np.isnan(d_rb_b)] = 0
                    holding[i,reb_buy] += d_rb_b
                    cash[i] -= (np.nansum(d_rb_b * self.vwap[i,reb_buy])
                                * (1+commission))
                
                # Lend
                if cash[i] < 0:
                    cash_lend -= cash[i]
                    cash[i] = 0
                
                # Sell
                if np.sum(all_sell) > 0:
                    cash[i] += (np.nansum(holding[i,all_sell] *
                                self.vwap[i,all_sell])
                                * (1-commission-stamp_duty))
                    holding[i,all_sell] = 0
                    
                # Repay
                if cash[i] > 0 and cash_lend > 0:
                    if cash[i] < cash_lend:
                        cash_lend -= cash[i]
                        cash[i] = 0
                    else:
                        cash[i] -= cash_lend
                        cash_lend = 0
                
                # buy
                if np.sum(all_buy) > 0 and cash[i] - cash_line > 0:
                    reb_value[all_buy] /= ((np.sum(reb_value[all_buy]) /
                             (cash[i] - cash_line))
                        if np.sum(reb_value[all_buy]) > (cash[i] - cash_line) 
                        else 1)
                    d_all_b = np.floor(reb_value[all_buy] /
                                       self.vwap[i-1,all_buy]/100)*100
                    d_all_b[np.isnan(d_all_b)] = 0
                    holding[i,all_buy] = d_all_b
                    cash[i] -= (np.nansum(d_all_b * 
                                self.vwap[i,all_buy]) 
                                * (1+commission))
            
                d_f = future_holding[i] - future_holding[i-1]
                f_dly_pnl = (200*d_f*((self.csi500_open[i]
                             - self.csi500[i])/2)) + (200*future_holding[i-1]
                            *(self.csi500[i-1] - self.csi500[i]))
                cash[i] += f_dly_pnl
                f_pnl[i] = f_dly_pnl
                
                holding_v = holding[i,:] * self.close_fill[i,:]
                holding_v[np.isnan(self.close_fill[i,:])] = 0
                
                cash[i] -= aum[i-1]* mngm_fee
                
                aum[i] = (holding_v.sum() + cash[i] - cash_lend) 
                p_ratio[i] = holding_v.sum() / aum[i]

                # Risk level for tommorrow
                if i in year_point:
                    topV = 0
                if aum[i] > topV:
                    topV = aum[i]
                else:
                    DD[i] = aum[i] / topV - 1

                            
                holding_v = np.sum(np.abs(reb_value))
                temp = aum[i]*pos_ratio
                if temp > 0:
                    turnover[i] = holding_v / temp
                elif holding_v == 0:
                    turnover[i] = 0.0
                else:
                    turnover[i] = 1.0
                    
            res['r_aum'] = aum
            res['r_turnover'] = turnover
            res['r_position'] = p_ratio
            res['r_cash'] = cash
            res['r_f_pnl'] = f_pnl
            res['r_f_holding'] = future_holding
            res['r_drawdown'] = DD
            res['risk_level'] = risk_level
            res['r_ret'] = np.r_[0, aum[1:]/aum[:-1] - 1]
        
        # Real trading
        f_pnl = np.zeros(len(self.dates))
        p_ratio = np.zeros(len(self.dates))
        holding = np.zeros(self.factor.shape)
        future_holding = np.zeros(len(self.dates))
        cash_lend = 0
        topV = init_cap
        DD = np.zeros(len(self.dates))
        turnover = np.zeros(len(self.dates))
        cash = np.zeros(len(self.dates))
        aum = np.zeros(len(self.dates))
        temp_reb_value = []
        for i in range(len(self.dates)):
            if i < 1:
                f_pnl[i] = 0.0
                p_ratio[i] = 0.0
                cash[i] = init_cap
                aum[i] = init_cap
                continue
    
            cash[i] = cash[i-1]
            holding[i,:] = holding[i-1,:]

            # Lend
            if cash[i] < 0:
                cash_lend -= cash[i]
                cash[i] = 0
            
            # Repay
            if cash[i] > 0 and cash_lend > 0:
                if cash[i] < cash_lend:
                    cash_lend -= cash[i]
                    cash[i] = 0
                else:
                    cash[i] -= cash_lend
                    cash_lend = 0
            
            weight_v = weights[i,:].copy()
            liq_mask = self.liq_top[i-1,:]/(pos_ratio*aum[i-1]*
                              (1-0.5*risk_level[i]))
            liq_mask[~(liq_mask < w_limit)] = w_limit
            liq_mask[liq_mask == 0] = w_limit
            balance_mask = weight_v > liq_mask
            if balance_mask.any():
                remain = (weight_v[balance_mask] -
                          liq_mask[balance_mask]).sum()
                weight_v[balance_mask] = liq_mask[balance_mask]
                weight_v[~balance_mask] *= (remain /
                        weight_v[~balance_mask].sum() + 1)
            balance_mask = weight_v > liq_mask
            if balance_mask.any():
                remain = (weight_v[balance_mask] -
                          liq_mask[balance_mask]).sum()
                weight_v[balance_mask] = liq_mask[balance_mask]
                weight_v[~balance_mask] *= (remain /
                        weight_v[~balance_mask].sum() + 1)
            weight_v /= weight_v.sum()
            weight_v = np.nan_to_num(weight_v)
            
            h_last = holding[i-1,:] * self.close_fill[i-1,:]
            h_last[np.isnan(self.close_fill[i-1,:])] = 0
            build_speed = pos_ratio*trade_complt
            reb_value = (build_speed*aum[i-1]*weight_v*(1-0.5*risk_level[i]) -
                         h_last)
            reb_value[reb_value < 0] *= ~self.cant_sell[i,reb_value < 0]
            reb_value[reb_value > 0] *= ~self.cant_buy[i,reb_value > 0]
    
            liq_cant = self.liq_top[i-1,:]*liq_limit
            liq_mask = np.abs(reb_value) >= liq_cant
            if np.sum(liq_mask) > 0:
                reb_value[liq_mask] = (np.sign(reb_value[liq_mask])*
                                       liq_cant[liq_mask])
            
            turn_mask = np.abs(reb_value / h_last) < turn_bot
            reb_value[turn_mask] = 0
            
            future_holding[i] = np.round((reb_value + h_last).sum() / 
                                          self.csi500[i-1] / 200)
            
            # Filtering non-tradable stock today but unknown yesterday
            reb_value[np.isnan(self.vwap[i,:])] = 0
            
            reb_buy = (reb_value > 0) * (h_last > 0)
            reb_sell = (reb_value < 0) * (reb_value + h_last > 0)
            all_buy = (reb_value > 0) * (h_last == 0)
            all_sell = (reb_value < 0) * (reb_value + h_last == 0)
            
            if cash_limit is not None:
                cash_line = cash_limit * aum[i-1]
            else:
                cash_line = 0
                        
                
            # Rebalance sell
            if np.sum(reb_sell) > 0:
                d_rb_s = np.round(reb_value[reb_sell] /
                                  self.close[i-1,reb_sell],-2)
                d_rb_s[np.isnan(d_rb_s)] = 0
                holding[i,reb_sell] += d_rb_s
                cash[i] += (np.nansum(-d_rb_s * self.vwap[i,reb_sell])
                            * (1-commission-stamp_duty))            
    
            # Repay
            if cash[i] > 0 and cash_lend > 0:
                if cash[i] < cash_lend:
                    cash_lend -= cash[i]
                    cash[i] = 0
                else:
                    cash[i] -= cash_lend
                    cash_lend = 0
            
            # Rebalance buy
            if np.sum(reb_buy) > 0 and cash[i] - cash_line > 0:
                reb_value[reb_buy] /= ((np.sum(reb_value[reb_buy]) /
                         (cash[i] - cash_line)) 
                    if np.sum(reb_value[reb_buy]) > (cash[i] - cash_line)
                    else 1)
                d_rb_b = np.round(reb_value[reb_buy] /
                                  self.close[i-1,reb_buy],-2)
                d_rb_b[np.isnan(d_rb_b)] = 0
                holding[i,reb_buy] += d_rb_b
                cash[i] -= (np.nansum(d_rb_b * self.vwap[i,reb_buy])
                            * (1+commission))
            
            # Lend
            if cash[i] < 0:
                cash_lend -= cash[i]
                cash[i] = 0
            
            # Sell
            if np.sum(all_sell) > 0:
                cash[i] += (np.nansum(holding[i,all_sell] *
                            self.vwap[i,all_sell])
                            * (1-commission-stamp_duty))
                holding[i,all_sell] = 0
                
            # Repay
            if cash[i] > 0 and cash_lend > 0:
                if cash[i] < cash_lend:
                    cash_lend -= cash[i]
                    cash[i] = 0
                else:
                    cash[i] -= cash_lend
                    cash_lend = 0
            
            # buy
            if np.sum(all_buy) > 0 and cash[i] - cash_line > 0:
                reb_value[all_buy] /= ((np.sum(reb_value[all_buy]) /
                         (cash[i] - cash_line))
                    if np.sum(reb_value[all_buy]) > (cash[i] - cash_line) 
                    else 1)
                d_all_b = np.floor(reb_value[all_buy] /
                                   self.vwap[i-1,all_buy]/100)*100
                d_all_b[np.isnan(d_all_b)] = 0
                holding[i,all_buy] = d_all_b
                cash[i] -= (np.nansum(d_all_b * 
                            self.vwap[i,all_buy]) 
                            * (1+commission))
            
            temp_reb_value.append(reb_value)
            
            d_f = future_holding[i] - future_holding[i-1]
            f_dly_pnl = (200*d_f*((self.csi500_open[i]
                         - self.csi500[i])/2)) + (200*future_holding[i-1]
                        *(self.csi500[i-1] - self.csi500[i]))
            cash[i] += f_dly_pnl
            f_pnl[i] = f_dly_pnl
            
            holding_v = holding[i,:] * self.close_fill[i,:]
            holding_v[np.isnan(self.close_fill[i,:])] = 0
            
            cash[i] -= aum[i-1] * mngm_fee
            
            aum[i] = (holding_v.sum() + cash[i] - cash_lend) 
            p_ratio[i] = holding_v.sum() / aum[i]

            # Risk level for tommorrow
            if i in year_point:
                topV = 0
            if aum[i] > topV:
                topV = aum[i]
            else:
                DD[i] = aum[i] / topV - 1
            
            holding_v = np.sum(np.abs(reb_value))
            temp = aum[i]*pos_ratio
            if temp > 0:
                turnover[i] = holding_v / temp
            elif holding_v == 0:
                turnover[i] = 0.0
            else:
                turnover[i] = 1.0
            
            
        res['aum'] = aum
        res['turnover'] = turnover
        res['position'] = p_ratio
        res['cash'] = cash
        res['f_pnl'] = f_pnl
        res['f_holding'] = future_holding
        res['drawdown'] = DD
        res['ret'] = np.r_[0, aum[1:]/aum[:-1] - 1]
        res['holding'] = holding
        res['reb_value'] = np.array(temp_reb_value)
        res['dd_dates'] = [str(x.date())
                           for x in self.dates[res['ret'] < -0.01]]
        self.backtest_res = res
        return res 
        
    @staticmethod
    def AnnRet(aum, year_point=None):
        if year_point is not None:
            y_point = np.r_[0, year_point, len(aum)-1]
            res = []
            for i in range(len(y_point)-1):
                temp = aum[y_point[i+1]]/aum[y_point[i]]
                res.append((temp**(1/(y_point[i+1] - y_point[i])))**244-1)
            res.append(((aum[-1]/aum[0])**(1/len(aum)))**244-1)
            return np.array(res)
        
        else:
            return ((aum[-1]/aum[0])**(1/len(aum)))**244-1
    
    @staticmethod
    def TotRet(aum, year_point=None):
        if year_point is not None:
            y_point = np.r_[0, year_point, len(aum)-1]
            res = []
            for i in range(len(y_point)-1):
                res.append(aum[y_point[i+1]]/aum[y_point[i]] - 1)
            res.append(aum[-1]/aum[0] - 1)
            return np.array(res)
        
        else:
            return aum[-1]/aum[0] - 1
    
    @staticmethod
    def Volatility(ret, year_point=None):
        ann_f = 252**0.5
        if year_point is not None:
            y_point = np.r_[0, year_point, len(ret)]
            res = []
            for i in range(len(y_point)-1):
                res.append(np.std(ret[y_point[i]:y_point[i+1]])*ann_f)
            res.append(np.nanstd(ret)*ann_f)
            return np.array(res)
        
        else:
            return np.nanstd(ret)*ann_f
    
    @staticmethod
    def WinRate(ret, year_point=None):
        if year_point is not None:
            y_point = np.r_[0, year_point, len(ret)]
            res = []
            for i in range(len(y_point)-1):
                res.append(np.sum(ret[y_point[i]:y_point[i+1]]>0)/
                           (y_point[i+1]-y_point[i]))
            res.append(np.sum(ret>0)/len(ret))
            return np.array(res)
        
        else:
            return np.sum(ret>0)/len(ret)
    
    @staticmethod
    def WinLossRatio(ret, year_point=None):
        if year_point is not None:
            y_point = np.r_[0, year_point, len(ret)]
            res = []
            for i in range(len(y_point)-1):
                temp_ret = ret[y_point[i]:y_point[i+1]]
                win = np.mean(temp_ret[temp_ret>0])
                loss = np.abs(np.mean(temp_ret[temp_ret<0]))
                res.append(win/loss if loss > 0 else 100)
            res.append(np.mean(ret[ret>0])/np.abs(np.mean(ret[ret<0])))
            return np.array(res)
        
        else:
            return np.mean(ret[ret>0])/np.abs(np.mean(ret[ret<0]))
            
    @staticmethod
    def MaxDrawDown(drawdown, year_point=None):
        if year_point is not None:
            y_point = np.r_[0, year_point, len(drawdown)]
            res = []
            for i in range(len(y_point)-1):
                res.append(np.min(drawdown[y_point[i]:y_point[i+1]]))
            res.append(np.min(drawdown))
            return np.array(res)
        
        else:
            return np.min(drawdown)
    
    @staticmethod
    def Turnover(turnover, year_point=None):
        if year_point is not None:
            y_point = np.r_[0, year_point, len(turnover)]
            res = []
            for i in range(len(y_point)-1):
                temp_to = turnover[y_point[i]:y_point[i+1]]
                res.append(np.mean(temp_to[temp_to > 0]))
            res.append(np.mean(turnover[turnover > 0]))
            return np.array(res)
        else:
            return np.np.mean(turnover[turnover > 0])
    
    @staticmethod
    def ROT(ret, turnover, year_point=None):
        if year_point is not None:
            y_point = np.r_[0, year_point, len(ret)]
            res = []
            for i in range(len(y_point)-1):
                s_ret = np.sum(ret[y_point[i]:y_point[i+1]])
                s_trn = np.sum(turnover[y_point[i]:y_point[i+1]])
                res.append(s_ret/s_trn * 1e4 if s_trn > 0 else 0)
            res.append(np.sum(ret)/np.sum(turnover) * 1e4)
            return np.array(res)
        
        else:
            return np.sum(ret) / np.sum(turnover) * 1e4
    
    def Summary(self, res=None, plot=True):
        """
        Summarize performance of the input backtest result.
        
        Parameters
        ----------
        res : result from self.Backtest, a dictionary which contains serveral
              backtest processes.
              
        plot : boolean, indicating whether to plot the curve.
        
        Returns
        -------
        A pandas.DataFrame which contains following criteria within 
        different year as well as the whole period:
            - Annualize Return
            - Total Return
            - Volatility
            - Win Rate
            - Win Loss Ratio
            - Maximum Drawdown
            - Sharpe Ratio
            - Sterling Ratio
            - Return over turnover
        """
        if res is None:
            res = self.backtest_res
        
        year_point = res['year_point']
        ann_ret = self.AnnRet(res['aum'], year_point)
        tot_ret = self.TotRet(res['aum'], year_point)
        vol = self.Volatility(res['ret'], year_point)
        win_rate = self.WinRate(res['ret'], year_point)
        win_loss = self.WinLossRatio(res['ret'], year_point)
        max_dd = self.MaxDrawDown(res['drawdown'], year_point)
        turnover = self.Turnover(res['turnover'], year_point)
        sharpe_r = ann_ret / vol
        sterling = ann_ret / -max_dd
        rot = self.ROT(res['ret'], res['turnover'], year_point)
        
        index_l = [x.year for x in self.dates[np.r_[0,year_point]]]+['Overall']
        col_l = ['AnnualReturn', 'TotalReturn', 'Volatility', 'WinRate',
                 'Win/Loss', 'MaxDrawDown', 'Turnover', 'SharpeRatio',
                 'SterlingRatio', 'Return/Turnover']
        summary = pd.DataFrame(np.c_[ann_ret, tot_ret, vol, win_rate,
                                     win_loss, max_dd, turnover, sharpe_r,
                                     sterling, rot],
                               index=index_l, columns=col_l)
        
        if plot:
            NAV = pd.Series(res['aum'],index=pd.to_datetime(self.dates))
            dd_dates = pd.to_datetime(res['dd_dates'])
            (NAV / NAV[0]).plot()
            for i in range(len(dd_dates)):
                plt.axvline(dd_dates[i], ls=':', c='k', alpha=0.5)
            if 'risk_level' in res:
                rNAV = pd.Series(res['r_aum'],index=pd.to_datetime(self.dates))
                ax = (rNAV / rNAV[0]).plot()
                ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                              res['risk_level'][np.newaxis],
                              cmap='Reds', alpha=0.2)
        return summary
