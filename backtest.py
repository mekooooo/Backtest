# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:41:42 2019

@author: Brian
"""

import pandas as pd
import numpy as np
from scipy import io
from progressbar import ProgressBar
import warnings

class Backtest:
    
    def __init__(self, temp_data=False):
        
        data_dir = './temp_data/' if temp_data else './Data/'
        self.data_dir = data_dir
        if temp_data:
            self.Dates = pd.to_datetime([str(x[0][0]) for x in io.loadmat(
                    f'{data_dir}date.mat').popitem()[1]])
            self.Tickers = [str(x[0][0]) for x in io.loadmat(
                    f'{data_dir}ticker.mat').popitem()[1]]
            
            self.StartDate = self.Dates[0]
            self.EndDate = self.Dates[-1]
            
            CSI500 = pd.Series(io.loadmat(
                    f'{data_dir}csi500_close.mat').popitem()[1].reshape([-1]),
                    index=self.Dates)
            CSI500_open = pd.Series(io.loadmat(
                    f'{data_dir}csi500_open.mat').popitem()[1].reshape([-1]),
                    index=self.Dates)
            self.Close = pd.DataFrame(io.loadmat(
                    f'{data_dir}data_close_price.mat').popitem()[1].T,
                    index=self.Dates, columns=self.Tickers)
            self.VWAP = pd.DataFrame(io.loadmat(
                    f'{data_dir}data_vwap_price.mat').popitem()[1].T, 
                    index=self.Dates, columns=self.Tickers)
            self.Check = pd.DataFrame(io.loadmat(
                    f'{data_dir}data_tickercheck.mat').popitem()[1].T,
                    index=self.Dates, columns=self.Tickers)
            self.MV = pd.DataFrame(io.loadmat(
                    f'{data_dir}data_mv.mat').popitem()[1].T,
                    index=self.Dates, columns=self.Tickers)
            
            self.Close[self.Close == 0] = np.nan
            self.VWAP[self.VWAP == 0] = np.nan
            self.Close_fill = self.Close.ffill(axis=0) 
            CSI500[CSI500 == 0] = np.nan
            CSI500_open[CSI500_open == 0] = np.nan                       
            self.CSI500 = CSI500.ffill().bfill()
            self.CSI500_open = CSI500_open.ffill().bfill()
            
            # Go up staying & Fall staying
            High = io.loadmat(f'{data_dir}data_high_price.mat').popitem()[1].T
            Low = io.loadmat(f'{data_dir}data_low_price.mat').popitem()[1].T
            cant = High == Low
            self.CantBuy = cant*(Low/self.Close_fill.shift(axis=1)-1 > 0.09)
            self.CantSell = cant*(High/self.Close_fill.shift(axis=1)-1 < -0.09)
            
            # Liquidity limit
            fturn = pd.DataFrame(io.loadmat(
                    f'{data_dir}data_fturn.mat').popitem()[1].T / 100,
                    index=self.Dates, columns=self.Tickers)
            fmv = pd.DataFrame(io.loadmat(
                    f'{data_dir}data_fmv.mat').popitem()[1].T, 
                    index=self.Dates, columns=self.Tickers)
            self.Liq_top = (fturn * fmv).rolling(5, axis=0).mean()
            L_mask = pd.isnull(self.Liq_top)
            self.Liq_top[L_mask] = (fturn * fmv)[L_mask]
    
    def TargetOn(self, Factor, scale_method=None):
        
        # Input control
        if not isinstance(Factor, pd.DataFrame):
            raise TypeError('Input factor should be pandas.DataFrame.')
        
        # Save original factor score
        self.Factor = Factor
        
        # Adjust time
        adjust_flag = False
        start_date = Factor.index[0]
        end_date = Factor.index[-1]
        
        StartDate = self.StartDate
        EndDate = self.EndDate
        
        if start_date != StartDate:
            SDate = start_date if start_date > StartDate else StartDate
            warnings.warn('Start date of input factor is ' +
                          f'{start_date.date()}, but start date of data is ' +
                          f'{StartDate.date()}.')
            adjust_flag = True
        else:
            SDate = start_date
            
        if end_date != EndDate:
            EDate = end_date if end_date < EndDate else EndDate
            warnings.warn(f'End date of input factor is {end_date.date()}, ' +
                          f'but end date of data is {EndDate.date()}.')
            adjust_flag = True
        else:
            EDate = end_date
        
        # Adjust tickers
        cp_set1 = set(Factor.columns) - set(self.Tickers)
        cp_set2 = set(self.Tickers) - set(Factor.columns)
        if len(cp_set1) > 0:
            warnings.warn(f'Tickers {cp_set1} in factor are not in data.')
            adjust_flag = True
        elif len(cp_set2) > 0:
            warnings.warn(f'Tickers {cp_set2} in data are not in factor.')
            adjust_flag = True
    
        if adjust_flag:
            ticker_set = sorted(list(set(Factor.columns) 
                                     and set(self.Tickers)))
            self.csi500 = self.CSI500[SDate:EDate].values
            self.csi500_open = self.CSI500_open[SDate:EDate].values
            self.close = self.Close[SDate:EDate][ticker_set].values
            self.vwap = self.VWAP[SDate:EDate][ticker_set].values
            self.check = self.Check[SDate:EDate][ticker_set].values
            self.mv = self.MV[SDate:EDate][ticker_set].values
            self.cant_buy = self.CantBuy[SDate:EDate][ticker_set].values
            self.cant_sell = self.CantSell[SDate:EDate][ticker_set].values
            self.close_fill = self.Close_fill[SDate:EDate][ticker_set].values
            self.liq_top = self.Liq_top[SDate:EDate][ticker_set].values
            self.dates = self.CSI500[SDate:EDate].index
            factor = Factor[SDate:EDate][ticker_set]
        else:
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
        if scale_method is None:
            raise ValueError('Please input scale method.')
        elif scale_method == 'standardize':
            factor = factor.sub(factor.mean(axis=1),axis=0).div(
                            factor.std(axis=1), axis=0)
        elif scale_method == 'normalize':
            temp_min = factor.min(axis=1)
            factor = factor.sub(temp_min, axis=0).div(
                            factor.max(axis=1)-temp_min, axis=0)
        
        self.factor = factor.values
    
    def Backtest(self, init_cap=100000000, mv_weighted=False, top_num=100,
                 risk_control=False, fir_line=-0.02, sec_line=-0.04,
                 fir_recov=-0.01, sec_recov=-0.03, w_limit=None, 
                 pos_ratio=0.8, trade_complt=1, liq_limit=0.1,
                 turn_bot=0.1, cash_limit=None, commission=0.0002,
                 stamp_duty=0.001, mngm_fee=0.0001):
        """
        Main backtest function in this class, which has replicated the whole
        trading process.
        
        Parameters
        ----------
        init_cap : integer, initial capital for backtest.
        
        mv_weighted : boolean, whether use free market value to initialize
                      weights. If False, the weight matrix will be initialized
                      as equally weighted.
                      
        top_num : integer, # of stocks on top will be picked.
        
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
        
        Notes
        -----
            - For this stage, risk level can be recover from 2 to 0. But
              liquidity limit and turnover limit can prevent the backtest
              from trading too large volume.
            
            - The output plot is using imperfect function to plot the
              background color, which stands for risk level that day. However,
              the vertical lines are not quite consistent with X axis.
        """
        self.factor[self.Check==0] = -10000
        weights = np.zeros(self.factor.shape)
        for i in range(len(self.dates)):
            if i < 1:
                continue
            top = np.argpartition(self.factor[i-1,:], -top_num)[-top_num:]
            weights[i,top] = mv[i-1,top] if mv_weighted else 1
        weights /= np.sum(weights, axis=1).reshape([-1,1])
        weights[np.isnan(weights)] = 0

        year_point = np.where(pd.Series([x.year for x in
                                         self.dates]).diff() == 1)[0]
        risk_level = np.zeros(len(self.dates))
        
        # Pure trading alpha and risk level
        if risk_control: 
            f_pnl = []
            p_ratio = []
            holding = np.zeros(self.factor.shape)
            future_holding = np.zeros(len(self.dates))
            cash_lend = 0
            topV = init_cap
            DD = np.zeros(len(self.dates))
            maxDD = 0
            cash = np.zeros(len(self.dates))
            cash[0] = init_cap
            aum = np.zeros(len(self.dates))
            aum[0] = init_cap
            pb = ProgressBar()
            for i in pb(range(len(self.dates))):
            
                if i < 1:
                    f_pnl.append(0)
                    p_ratio.append(0)
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
                
                weight_v = weights[i,:]
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
#                    cansell = np.where(~np.isnan(self.vwap[i,all_sell]))[0]
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
                f_pnl.append(f_dly_pnl)
                
                holding_v = holding[i,:] * self.close_fill[i,:]
                holding_v[np.isnan(self.close_fill[i,:])] = 0
                p_ratio.append(holding_v.sum() / aum[i])
                
                cash[i] -= aum[i-1]* 0.0001
                
                aum[i] = (holding_v.sum() + cash[i] - cash_lend) 
                
                # Risk level for tommorrow
                if i in year_point:
                    topV = 0
                    maxDD = 0
                if aum[i] > topV:
                    topV = aum[i]
                else:
                    DD[i] = aum[i] / topV - 1
                    if DD[i] < maxDD:
                        maxDD = DD[i]
            
            temp_DD = DD.copy()
            temp_aum = aum.copy()
            temp_hocc = np.array(p_ratio)
        
        # Real trading
        f_pnl = []
        p_ratio = []
        holding = np.zeros(self.factor.shape)
        future_holding = np.zeros(len(self.dates))
        cash_lend = 0
        topV = init_cap
        DD = np.zeros(len(self.dates))
        maxDD = 0
        cash = np.zeros(len(self.dates))
        cash[0] = init_cap
        aum = np.zeros(len(self.dates))
        aum[0] = init_cap
        pb = ProgressBar()
        for i in pb(range(len(self.dates))):
            if i < 1:
                f_pnl.append(0)
                p_ratio.append(0)
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
            
            weight_v = weights[i,:]
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
#                cansell = np.where(~np.isnan(self.vwap[i,all_sell]))[0]
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
            f_pnl.append(f_dly_pnl)
            
            holding_v = holding[i,:] * self.close_fill[i,:]
            holding_v[np.isnan(self.close_fill[i,:])] = 0
            p_ratio.append(holding_v.sum() / aum[i])
            
            cash[i] -= aum[i-1] * mngm_fee
            
            aum[i] = (holding_v.sum() + cash[i] - cash_lend) 
            
            # Risk level for tommorrow
            if i in year_point:
                topV = 0
                maxDD = 0
            if aum[i] > topV:
                topV = aum[i]
            else:
                DD[i] = aum[i] / topV - 1
                if DD[i] < maxDD:
                    maxDD = DD[i]
        
        dps = pd.Series(aum,index=pd.to_datetime(self.dates))
        temp_dps = pd.Series(temp_aum,index=pd.to_datetime(self.dates))
        ax = (dps / dps[0]).plot()
        (temp_dps / temp_dps[0]).plot()
        ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                      risk_level[np.newaxis], cmap='Reds', alpha=0.2)
        
            
            

if __name__ == '__main__':
    
    Factor = pd.read_csv('D:/Projects/Technical/Factors/RSI_bot_10_22_5.csv',
                         index_col = 0, header = 0, parse_dates=True)
    temp = Backtest(temp_data=True)
    temp.TargetOn(Factor, scale_method='normalize')        
    temp.Backtest(w_limit=0.05, risk_control=True)
    
    

    #%% __init__
    data_dir = './temp_data/'
    Dates = pd.to_datetime([str(x[0][0]) for x in io.loadmat(f'{data_dir}date.mat').popitem()[1]])
    Tickers = [str(x[0][0]) for x in io.loadmat(f'{data_dir}ticker.mat').popitem()[1]]
    
    StartDate = Dates[0]
    EndDate = Dates[-1]
    
    # backtest
    CSI500 = pd.Series(io.loadmat(f'{data_dir}csi500_close.mat').popitem()[1].reshape([-1]),index=Dates)
    CSI500_open = pd.Series(io.loadmat(f'{data_dir}csi500_open.mat').popitem()[1].reshape([-1]),index=Dates)
    Close = pd.DataFrame(io.loadmat(f'{data_dir}data_close_price.mat').popitem()[1].T, index=Dates, columns=Tickers)
    VWAP = pd.DataFrame(io.loadmat(f'{data_dir}data_vwap_price.mat').popitem()[1].T, index=Dates, columns=Tickers)
    Check = pd.DataFrame(io.loadmat(f'{data_dir}data_tickercheck.mat').popitem()[1].T, index=Dates, columns=Tickers)
    MV = pd.DataFrame(io.loadmat(f'{data_dir}data_mv.mat').popitem()[1].T, index=Dates, columns=Tickers)
    
    Close[Close == 0] = np.nan
    VWAP[VWAP == 0] = np.nan
    CSI500[CSI500 == 0] = np.nan
    CSI500_open[CSI500_open == 0] = np.nan
    Close_fill = Close.ffill(axis=0)
    High = io.loadmat(f'{data_dir}data_high_price.mat').popitem()[1].T
    Low = io.loadmat(f'{data_dir}data_low_price.mat').popitem()[1].T
    cant = High == Low
    CantBut = cant * (Low / Close_fill.shift(axis=1) - 1 > 0.09)
    CantSell = cant * (High / Close_fill.shift(axis=1) - 1 < -0.09)
#    Close_fill = Close_fill.values
#    Close = Close.values
#    VWAP = VWAP.values
    CSI500 = CSI500.ffill().bfill()
    CSI500_open = CSI500_open.ffill().bfill()
#    MV_weight = MV / np.sum(MV, axis=0)
    
    fturn = pd.DataFrame(io.loadmat(f'{data_dir}data_fturn.mat').popitem()[1].T / 100, index=Dates, columns=Tickers)
    fmv = pd.DataFrame(io.loadmat(f'{data_dir}data_fmv.mat').popitem()[1].T, index=Dates, columns=Tickers)
    Liq_top = (fturn * fmv).rolling(5,axis=0).mean()
    Liq_top[pd.isnull(Liq_top)] = (fturn * fmv)[pd.isnull(Liq_top)]
    
    #%% config
    Factor = pd.read_csv('D:/Projects/Technical/Factors/RSI_bot_10_22_5.csv',
                         index_col = 0, header = 0, parse_dates=True)
    
    adjust_time = False
    scale_method = 'normalize'
    # input to config
    start_date = Factor.index[0]
    end_date = Factor.index[-1]
    
    if start_date != StartDate:
        SDate = start_date if start_date > StartDate else StartDate
        warnings.warn(f'''Start date of input factor is {start_date.date()},
                          but start date of data is {StartDate.date()}.''')
        adjust_time = True
    else:
        SDate = start_date
        
    if end_date != EndDate:
        EDate = end_date if end_date < EndDate else EndDate
        warnings.warn(f'''End date of input factor is {end_date.date()},
                          but end date of data is {EndDate.date()}.''')
        adjust_time = True
    else:
        EDate = end_date
    
    csi500 = CSI500[SDate:EDate].values
    csi500_open = CSI500_open[SDate:EDate].values
    close = Close[SDate:EDate].values
    vwap = VWAP[SDate:EDate].values
    check = Check[SDate:EDate].values
    mv = MV[SDate:EDate].values
    cant_buy = CantBut[SDate:EDate].values
    cant_sell = CantSell[SDate:EDate].values
    close_fill = Close_fill[SDate:EDate].values
    liq_top = Liq_top[SDate:EDate].values
    dates = CSI500[SDate:EDate].index
    
    # scale
    if scale_method == 'standardize':
        factor = Factor.sub(Factor.mean(axis=1), axis=0).div(Factor.std(axis=1), axis=0)
    elif scale_method == 'normalize':
        temp_min = Factor.min(axis=1)
        factor = Factor.sub(temp_min, axis=0).div(Factor.max(axis=1)-temp_min, axis=0)
    factor = factor.values
    
    #%% backtest
    factor[Check==0] = -10000
    weights = np.zeros(factor.shape)
    for i in range(factor.shape[0]):
        if i < 1:
            continue
        top = np.argpartition(factor[i-1,:], -50)[-50:]
        weights[i,top] = 1#mv[top,i-1]
    weights /= np.sum(weights, axis=1).reshape([-1,1])
    weights[np.isnan(weights)] = 0
    
    bm = CSI500 / CSI500.shift(1)
    cash_total = 100000000

    year_point = np.where(pd.Series([x.year for x in dates]).diff() == 1)[0]

    occ = []
    hocc = []
    holding = np.zeros(close.shape)
    future_holding = np.zeros(close.shape[0])
    cash_lend = 0
    topV = cash_total
    DD = np.zeros(close.shape[0])
    maxDD = 0
    risk_level = np.zeros(close.shape[0])
    cash = np.zeros(close.shape[0])
    cash[0] = cash_total
    aum = np.zeros(close.shape[0])
    aum[0] = cash_total
    model_line = aum.copy()
    NV_today = np.zeros(close.shape[1])
    pb = ProgressBar()
    for i in pb(range(close.shape[0])):
    
        if i < 1:
            occ.append(0)
            hocc.append(0)
            continue

        cash[i] = cash[i-1]
        holding[i,:] = holding[i-1,:]
#        if i == 2318:
#            break
        # lend
        if cash[i] < 0:
            cash_lend -= cash[i]
            cash[i] = 0
        
        # repay
        if cash[i] > 0 and cash_lend > 0:
            if cash[i] < cash_lend:
                cash_lend -= cash[i]
                cash[i] = 0
            else:
                cash[i] -= cash_lend
                cash_lend = 0
        
        if risk_level[i-1] == 0:
            risk_level[i] = 1*(DD[i-1] <= -0.02) + 1*(DD[i-1] <= -0.04)
        else:
            risk_level[i] = risk_level[i-1] - (risk_level[i-1]-1)*(DD[i-1] >= -0.03) - 1*(DD[i-1] >= -0.01) + (2-risk_level[i-1])*(DD[i-1] <= -0.04)
        
        weight_v = weights[i,:]
        liq_mask = liq_top[i,:]/(0.8*aum[i-1])
        liq_mask[~(liq_mask < 0.05)] = 0.05
        liq_mask[liq_mask == 0] = 0.05
        balance_mask = weight_v > liq_mask
        if balance_mask.any():
            remain = (weight_v[balance_mask] - liq_mask[balance_mask]).sum()
            weight_v[balance_mask] = liq_mask[balance_mask]
            weight_v[~balance_mask] *= (remain / weight_v[~balance_mask].sum() + 1)
        balance_mask = weight_v > liq_mask
        if balance_mask.any():
            remain = (weight_v[balance_mask] - liq_mask[balance_mask]).sum()
            weight_v[balance_mask] = liq_mask[balance_mask]
            weight_v[~balance_mask] *= (remain / weight_v[~balance_mask].sum() + 1)
        weight_v /= weight_v.sum()
        
        h_yes = holding[i-1,:] * close_fill[i-1,:]
        h_yes[np.isnan(close_fill[i-1,:])] = 0
        build_speed = 0.8
        reb_value = build_speed*aum[i-1]*weight_v - h_yes
        reb_value[reb_value < 0] *= ~cant_sell[i,reb_value < 0]
        reb_value[reb_value > 0] *= ~cant_buy[i,reb_value > 0]

        liq_mask = np.abs(reb_value) >= liq_top[i-1,:] * 0.1
        if np.sum(liq_mask) > 0:
            reb_value[liq_mask] = np.sign(reb_value[liq_mask])*liq_top[i-1,liq_mask]*0.1
        
        turn_mask = np.abs(reb_value / h_yes) < 0.1
        reb_value[turn_mask] = 0
        
        future_holding[i] = np.round((reb_value+h_yes).sum() / csi500[i-1] / 200)
        
        reb_buy_mask = (reb_value > 0) * (h_yes > 0)
        reb_sell_mask = (reb_value < 0) * (reb_value + h_yes > 0)
        all_buy_mask = (reb_value > 0) * (h_yes == 0)
        all_sell_mask = (reb_value < 0) * (reb_value + h_yes == 0)
        
#        cash_line = 0.15 * aum[i-1]
        
        trn = []
        
        # rebalance sell
        if np.sum(reb_sell_mask) > 0:
            d_rb_s = np.round(reb_value[reb_sell_mask] / close[i-1,reb_sell_mask],-2)
            d_rb_s[np.isnan(d_rb_s)] = 0
            holding[i,reb_sell_mask] += d_rb_s
            cash[i] += np.nansum(-d_rb_s * vwap[i,reb_sell_mask]) * (1-0.0002-0.001) # 72984458            

        # repay
        if cash[i] > 0 and cash_lend > 0:
            if cash[i] < cash_lend:
                cash_lend -= cash[i]
                cash[i] = 0
            else:
                cash[i] -= cash_lend
                cash_lend = 0
        
        # rebalance buy
        if np.sum(reb_buy_mask) > 0 and cash[i] > 0:
            reb_value[reb_buy_mask] /= ((np.sum(reb_value[reb_buy_mask]) / (cash[i])) if np.sum(reb_value[reb_buy_mask]) > (cash[i]) else 1)
            d_rb_b = np.round(reb_value[reb_buy_mask] / close[i-1,reb_buy_mask],-2)
            d_rb_b[np.isnan(d_rb_b)] = 0
            holding[i,reb_buy_mask] += d_rb_b
            cash[i] -= np.nansum(d_rb_b * vwap[i,reb_buy_mask]) * (1+0.0002)
            
        # lend
        if cash[i] < 0:
            cash_lend -= cash[i]
            cash[i] = 0
        
        # sell
        if np.sum(all_sell_mask) > 0:
            cash[i] += np.nansum(holding[i,all_sell_mask] * vwap[i,all_sell_mask]) * (1-0.0002-0.001) # 111391565
            holding[i,all_sell_mask] = 0
            

        # repay
        if cash[i] > 0 and cash_lend > 0:
            if cash[i] < cash_lend:
                cash_lend -= cash[i]
                cash[i] = 0
            else:
                cash[i] -= cash_lend
                cash_lend = 0
        
        # buy
        if np.sum(all_buy_mask) > 0 and cash[i] > 0:
            reb_value[all_buy_mask] /= ((np.sum(reb_value[all_buy_mask]) / (cash[i])) if np.sum(reb_value[all_buy_mask]) > (cash[i]) else 1)
            d_all_b = np.floor(reb_value[all_buy_mask] / vwap[i-1,all_buy_mask]/100)*100
            d_all_b[np.isnan(d_all_b)] = 0
            holding[i,all_buy_mask] = d_all_b
            cash[i] -= np.nansum(d_all_b * vwap[i,all_buy_mask]) * (1+0.0002) # 61828839
        
    
        d_f = future_holding[i] - future_holding[i-1]
        cash[i] += (200*d_f*((csi500_open[i] + csi500[i])/2 - csi500[i]))
        cash[i] += (200*future_holding[i-1]*(csi500[i-1] - csi500[i]))
        
        holding_v = holding[i,:] * close_fill[i,:]
        holding_v[np.isnan(close_fill[i,:])] = 0
        cash[i] -= aum[i-1]* 0.0001
        aum[i] = (holding_v.sum() + cash[i] - cash_lend) 
        occ.append(200*d_f*((csi500_open[i] + csi500[i])/2 - csi500[i])+200*future_holding[i-1]*(csi500[i-1] - csi500[i]))
        hocc.append(holding_v.sum() / aum[i])
        if i in year_point:
            topV = 0
            maxDD = 0
        if aum[i] > topV:
            topV = aum[i]
        else:
            DD[i] = aum[i] / topV - 1
            if DD[i] < maxDD:
                maxDD = DD[i]
    
    temp_DD = DD.copy()
    temp_aum = aum.copy()
    temp_hocc = np.array(hocc)

    occ = []
    hocc = []
    holding = np.zeros(close.shape)
    future_holding = np.zeros(close.shape[0])
    cash_lend = 0
    topV = cash_total
    DD = np.zeros(close.shape[0])
    maxDD2 = 0
    cash = np.zeros(close.shape[0])
    cash[0] = cash_total
    aum = np.zeros(close.shape[0])
    aum[0] = cash_total
    NV_today = np.zeros(close.shape[1])
    pb = ProgressBar()
    for i in pb(range(close.shape[0])):
        if i < 1:
            occ.append(0)
            hocc.append(0)
            continue

        cash[i] = cash[i-1]
        holding[i,:] = holding[i-1,:]
#        if i == 1688:
#            break
        # lend
        if cash[i] < 0:
            cash_lend -= cash[i]
            cash[i] = 0
        
        # repay
        if cash[i] > 0 and cash_lend > 0:
            if cash[i] < cash_lend:
                cash_lend -= cash[i]
                cash[i] = 0
            else:
                cash[i] -= cash_lend
                cash_lend = 0
        
        weight_v = weights[i,:]
        liq_mask = liq_top[i,:]/(0.8*aum[i-1]*(1-0.5*risk_level[i]))
        liq_mask[~(liq_mask < 0.05)] = 0.05
        liq_mask[liq_mask == 0] = 0.05
        balance_mask = weight_v > liq_mask
        if balance_mask.any():
            remain = (weight_v[balance_mask] - liq_mask[balance_mask]).sum()
            weight_v[balance_mask] = liq_mask[balance_mask]
            weight_v[~balance_mask] *= (remain / weight_v[~balance_mask].sum() + 1)
        balance_mask = weight_v > liq_mask
        if balance_mask.any():
            remain = (weight_v[balance_mask] - liq_mask[balance_mask]).sum()
            weight_v[balance_mask] = liq_mask[balance_mask]
            weight_v[~balance_mask] *= (remain / weight_v[~balance_mask].sum() + 1)
        weight_v /= weight_v.sum()
        
        h_yes = holding[i-1,:] * close_fill[i-1,:]
        h_yes[np.isnan(close_fill[i-1,:])] = 0
        build_speed = 0.8
        reb_value = build_speed*aum[i-1]*weight_v*(1-0.5*risk_level[i]) - h_yes
        reb_value[reb_value < 0] *= ~cant_sell[i,reb_value < 0]
        reb_value[reb_value > 0] *= ~cant_buy[i,reb_value > 0]

        liq_mask = np.abs(reb_value) >= liq_top[i-1,:] * 0.1
        if np.sum(liq_mask) > 0:
            reb_value[liq_mask] = np.sign(reb_value[liq_mask])*liq_top[i-1,liq_mask]*0.1
        
        turn_mask = np.abs(reb_value / h_yes) < 0.1
        reb_value[turn_mask] = 0
        
        future_holding[i] = np.round((reb_value+h_yes).sum() / csi500[i-1] / 200)
        
        reb_buy_mask = (reb_value > 0) * (h_yes > 0)
        reb_sell_mask = (reb_value < 0) * (reb_value + h_yes > 0)
        all_buy_mask = (reb_value > 0) * (h_yes == 0)
        all_sell_mask = (reb_value < 0) * (reb_value + h_yes == 0)
        
#        cash_line = 0.15 * aum[i-1]
        
        trn = []
        
        # rebalance sell
        if np.sum(reb_sell_mask) > 0:
            d_rb_s = np.round(reb_value[reb_sell_mask] / close[i-1,reb_sell_mask],-2)
            d_rb_s[np.isnan(d_rb_s)] = 0
            holding[i,reb_sell_mask] += d_rb_s
            cash[i] += np.nansum(-d_rb_s * vwap[i,reb_sell_mask]) * (1-0.0002-0.001) # 72984458
            

        # repay
        if cash[i] > 0 and cash_lend > 0:
            if cash[i] < cash_lend:
                cash_lend -= cash[i]
                cash[i] = 0
            else:
                cash[i] -= cash_lend
                cash_lend = 0
        
        # rebalance buy
        if np.sum(reb_buy_mask) > 0 and cash[i] > 0:
            reb_value[reb_buy_mask] /= ((np.sum(reb_value[reb_buy_mask]) / (cash[i])) if np.sum(reb_value[reb_buy_mask]) > (cash[i]) else 1)
            d_rb_b = np.round(reb_value[reb_buy_mask] / close[i-1,reb_buy_mask],-2)
            d_rb_b[np.isnan(d_rb_b)] = 0
            holding[i,reb_buy_mask] += d_rb_b
            cash[i] -= np.nansum(d_rb_b * vwap[i,reb_buy_mask]) * (1+0.0002)
            
        # lend
        if cash[i] < 0:
            cash_lend -= cash[i]
            cash[i] = 0
        
        # sell
        if np.sum(all_sell_mask) > 0:
            cash[i] += np.nansum(holding[i,all_sell_mask] * vwap[i,all_sell_mask]) * (1-0.0002-0.001) # 111391565
            holding[i,all_sell_mask] = 0
            

        # repay
        if cash[i] > 0 and cash_lend > 0:
            if cash[i] < cash_lend:
                cash_lend -= cash[i]
                cash[i] = 0
            else:
                cash[i] -= cash_lend
                cash_lend = 0
        
        # buy
        if np.sum(all_buy_mask) > 0 and cash[i] > 0:
            reb_value[all_buy_mask] /= ((np.sum(reb_value[all_buy_mask]) / (cash[i])) if np.sum(reb_value[all_buy_mask]) > (cash[i]) else 1)
            d_all_b = np.floor(reb_value[all_buy_mask] / vwap[i-1,all_buy_mask]/100)*100
            d_all_b[np.isnan(d_all_b)] = 0
            holding[i,all_buy_mask] = d_all_b
            cash[i] -= np.nansum(d_all_b * vwap[i,all_buy_mask]) * (1+0.0002) # 61828839
        
    
        d_f = future_holding[i] - future_holding[i-1]
        cash[i] += (200*d_f*((csi500_open[i] + csi500[i])/2 - csi500[i]))
        cash[i] += (200*future_holding[i-1]*(csi500[i-1] - csi500[i]))
        
        holding_v = holding[i,:] * close_fill[i,:]
        holding_v[np.isnan(close_fill[i,:])] = 0
        cash[i] -= aum[i-1]* 0.0001
        aum[i] = (holding_v.sum() + cash[i] - cash_lend) 
        occ.append(200*d_f*((csi500_open[i] + csi500[i])/2 - csi500[i])+200*future_holding[i-1]*(csi500[i-1] - csi500[i]))
        hocc.append(holding_v.sum() / aum[i])
        if i in year_point:
            topV = 0
            maxDD2 = 0
        if aum[i] > topV:
            topV = aum[i]
        else:
            DD[i] = aum[i] / topV - 1
            if DD[i] < maxDD2:
                maxDD2 = DD[i]
        
    
    dps = pd.Series(aum,index=pd.to_datetime(dates))
    temp_dps = pd.Series(temp_aum,index=pd.to_datetime(dates))
    ax = (dps / dps[0]).plot()
    (temp_dps / temp_dps[0]).plot()
    ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                  risk_level[np.newaxis], cmap='Reds', alpha=0.2)
    
    
    




