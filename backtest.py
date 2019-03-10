# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:41:42 2019

@author: Brian
"""

import pandas as pd
import numpy as np
from scipy import io
import os
from progressbar import ProgressBar

class S_FactorAnalysis:
    
    def __init__(self, temp_data=False):
        
        data_dir = './temp_data/' if temp_data else './Data/'
        
        if temp_data:
            self.Dates = [str(x[0][0]) for x in
                          io.loadmat(f'{data_dir}date.mat').popitem()[1]]
            self.Tickers = [str(x[0][0]) for x in
                            io.loadmat(f'{data_dir}ticker.mat').popitem()[1]]
            
            CSI500 = pd.Series(io.loadmat(f'{data_dir}csi500_close.mat'
                                          ).popitem()[1].reshape([-1]))
            CSI500_open = pd.Series(io.loadmat(f'{data_dir}csi500_open.mat'
                                               ).popitem()[1].reshape([-1]))
            Close = pd.DataFrame(io.loadmat(f'{data_dir}data_close_price.mat'
                                            ).popitem()[1])
            VWAP = pd.DataFrame(io.loadmat(f'{data_dir}data_vwap_price.mat'
                                           ).popitem()[1])
            self.Check = io.loadmat(f'{data_dir}data_tickercheck.mat'
                                    ).popitem()[1].T
            self.MV = io.loadmat(f'{data_dir}data_mv.mat').popitem()[1].T
            
            Close[Close == 0] = np.nan
            VWAP[VWAP == 0] = np.nan
            self.CSI500[CSI500 == 0] = np.nan
            self.CSI500_open[CSI500_open == 0] = np.nan
            self.High = io.loadmat(f'{data_dir}data_high_price.mat'
                                   ).popitem()[1].T
            self.Low = io.loadmat(f'{data_dir}data_low_price.mat'
                                  ).popitem()[1].T
            
            Close_fill = Close.ffill(axis=1)
            cant = High == Low
            self.cant_buy = cant * (Low / Close_fill.shift(axis=1) - 1 > 
                                    0.09).values
            self.cant_sell = cant * (High / Close_fill.shift(axis=1) - 1
                                     < -0.09).values
            
            self.Close_fill = Close_fill.values.T
            self.Close = Close.values.T
            self.VWAP = VWAP.values.T
            self.CSI500 = CSI500.ffill().bfill().values
            self.CSI500_open = CSI500_open.ffill().bfill().values
            self.MV_weight = self.MV / np.sum(self.MV, axis=0).T
            
            fturn = io.loadmat(f'{data_dir}data_fturn.mat').popitem()[1] / 100
            fmv = io.loadmat(f'{data_dir}data_fmv.mat').popitem()[1]
            fliq_top = pd.DataFrame(fturn * fmv).rolling(5,axis=1).mean()
            fliq_top[pd.isnull(fliq_top)] = pd.DataFrame(fturn *
                     fmv)[pd.isnull(fliq_top)]
            self.fliq_top = fliq_top.values.T
    
    def config(self, scale_method='standardize'):
        self.scale_method = scale_method
    
    def Backtest(self, factor, pick, init_cap=100000000, risk_control=True, scale=True):
        
        factor[self.Check==0] = -100
        weights = np.zeros(factor.shape)
        for i in range(factor.shape[1]):
            if i < 1:
                continue
            top = np.argpartition(factor[i-1:], -50)[-50:]
            weights[i,top] = 1 # self.MV[i-1, top]
        weights /= np.sum(weights,axis=1)
        weights[np.isnan(weights)] = 0
    
        bm = CSI500 / CSI500.shift()        
        year_point = np.where(pd.Series([int(x[:4]) for x
                                         in self.Dates]).diff() == 1)[0]
        
        occ = []
        hocc = []
        holding = np.zeros(factor)
        future_holding = np.zeros(len(self.Dates))
        cash_lend = 0
        topV = init_cap
        DD = np.zeros(len(self.Dates))
        maxDD = 0
        risk_level = np.zeros(len(self.Dates))
        cash = np.zeros(len(self.Dates))
        cash[0] = init_cap
        aum = np.zeros(len(self.Dates))
        aum[0] = init_cap
        model_line = aum.copy()
        NV_today = np.zeros(len(self.Tickers))
        pb = ProgressBar()
        for i in pb(range(len(self.Dates))):
        
            if i < 1:
                occ.append(0)
                hocc.append(0)
                continue
    
            cash[i] = cash[i-1]
            holding[:,i] = holding[:,i-1]
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
            
            weight_v = weights[:,i]
            liq_mask = fliq_top[:,i]/(0.8*aum[i-1])
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
            
            h_yes = holding[:,i-1] * Close_fill[:,i-1]
            h_yes[np.isnan(Close_fill[:,i-1])] = 0
            build_speed = 0.8
            reb_value = build_speed*aum[i-1]*weight_v - h_yes
            reb_value[reb_value < 0] *= ~cant_sell[reb_value < 0,i]
            reb_value[reb_value > 0] *= ~cant_buy[reb_value > 0,i]
    
            liq_mask = np.abs(reb_value) >= fliq_top[:,i-1] * 0.1
            if np.sum(liq_mask) > 0:
                reb_value[liq_mask] = np.sign(reb_value[liq_mask])*fliq_top[liq_mask,i-1]*0.1
            
            turn_mask = np.abs(reb_value / h_yes) < 0.1
            reb_value[turn_mask] = 0
            
            future_holding[i] = np.round((reb_value+h_yes).sum() / CSI500[i-1] / 200)
            
            reb_buy_mask = (reb_value > 0) * (h_yes > 0)
            reb_sell_mask = (reb_value < 0) * (reb_value + h_yes > 0)
            all_buy_mask = (reb_value > 0) * (h_yes == 0)
            all_sell_mask = (reb_value < 0) * (reb_value + h_yes == 0)
            
    #        cash_line = 0.15 * aum[i-1]
            
            trn = []
            
            # rebalance sell
            if np.sum(reb_sell_mask) > 0:
                d_rb_s = np.round(reb_value[reb_sell_mask] / Close[reb_sell_mask,i-1],-2)
                d_rb_s[np.isnan(d_rb_s)] = 0
                holding[reb_sell_mask,i] += d_rb_s
                cash[i] += np.nansum(-d_rb_s * VWAP[reb_sell_mask,i]) * (1-0.0002-0.001) # 72984458            
    
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
                d_rb_b = np.round(reb_value[reb_buy_mask] / Close[reb_buy_mask,i-1],-2)
                d_rb_b[np.isnan(d_rb_b)] = 0
                holding[reb_buy_mask,i] += d_rb_b
                cash[i] -= np.nansum(d_rb_b * VWAP[reb_buy_mask,i]) * (1+0.0002)
                
            # lend
            if cash[i] < 0:
                cash_lend -= cash[i]
                cash[i] = 0
            
            # sell
            if np.sum(all_sell_mask) > 0:
                cash[i] += np.nansum(holding[all_sell_mask,i] * VWAP[all_sell_mask,i]) * (1-0.0002-0.001) # 111391565
                holding[all_sell_mask,i] = 0
                
    
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
                d_all_b = np.floor(reb_value[all_buy_mask] / VWAP[all_buy_mask,i-1]/100)*100
                d_all_b[np.isnan(d_all_b)] = 0
                holding[all_buy_mask,i] = d_all_b
                cash[i] -= np.nansum(d_all_b * VWAP[all_buy_mask,i]) * (1+0.0002) # 61828839
            
        
            d_f = future_holding[i] - future_holding[i-1]
            cash[i] += (200*d_f*((CSI500_open[i] + CSI500[i])/2 - CSI500[i]))
            cash[i] += (200*future_holding[i-1]*(CSI500[i-1] - CSI500[i]))
            
            holding_v = holding[:,i] * Close_fill[:,i]
            holding_v[np.isnan(Close_fill[:,i])] = 0
            cash[i] -= aum[i-1]* 0.0001
            aum[i] = (holding_v.sum() + cash[i] - cash_lend) 
            occ.append(200*d_f*((CSI500_open[i] + CSI500[i])/2 - CSI500[i])+200*future_holding[i-1]*(CSI500[i-1] - CSI500[i]))
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
        holding = np.zeros(Close.shape)
        future_holding = np.zeros(Close.shape[1])
        cash_lend = 0
        topV = cash_total
        DD = np.zeros(Close.shape[1])
        maxDD2 = 0
        cash = np.zeros(Close.shape[1])
        cash[0] = cash_total
        aum = np.zeros(Close.shape[1])
        aum[0] = cash_total
        NV_today = np.zeros(Close.shape[0])
        pb = ProgressBar()
        for i in pb(range(Close.shape[1])):
            if i < 1:
                occ.append(0)
                hocc.append(0)
                continue
    
            cash[i] = cash[i-1]
            holding[:,i] = holding[:,i-1]
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
            
            weight_v = weights[:,i]
            liq_mask = fliq_top[:,i]/(0.8*aum[i-1]*(1-0.5*risk_level[i]))
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
            
            h_yes = holding[:,i-1] * Close_fill[:,i-1]
            h_yes[np.isnan(Close_fill[:,i-1])] = 0
            build_speed = 0.8
            reb_value = build_speed*aum[i-1]*weight_v*(1-0.5*risk_level[i]) - h_yes
            reb_value[reb_value < 0] *= ~cant_sell[reb_value < 0,i]
            reb_value[reb_value > 0] *= ~cant_buy[reb_value > 0,i]
    
            liq_mask = np.abs(reb_value) >= fliq_top[:,i-1] * 0.1
            if np.sum(liq_mask) > 0:
                reb_value[liq_mask] = np.sign(reb_value[liq_mask])*fliq_top[liq_mask,i-1]*0.1
            
            turn_mask = np.abs(reb_value / h_yes) < 0.1
            reb_value[turn_mask] = 0
            
            future_holding[i] = np.round((reb_value+h_yes).sum() / CSI500[i-1] / 200)
            
            reb_buy_mask = (reb_value > 0) * (h_yes > 0)
            reb_sell_mask = (reb_value < 0) * (reb_value + h_yes > 0)
            all_buy_mask = (reb_value > 0) * (h_yes == 0)
            all_sell_mask = (reb_value < 0) * (reb_value + h_yes == 0)
            
    #        cash_line = 0.15 * aum[i-1]
            
            trn = []
            
            # rebalance sell
            if np.sum(reb_sell_mask) > 0:
                d_rb_s = np.round(reb_value[reb_sell_mask] / Close[reb_sell_mask,i-1],-2)
                d_rb_s[np.isnan(d_rb_s)] = 0
                holding[reb_sell_mask,i] += d_rb_s
                cash[i] += np.nansum(-d_rb_s * VWAP[reb_sell_mask,i]) * (1-0.0002-0.001) # 72984458
                
    
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
                d_rb_b = np.round(reb_value[reb_buy_mask] / Close[reb_buy_mask,i-1],-2)
                d_rb_b[np.isnan(d_rb_b)] = 0
                holding[reb_buy_mask,i] += d_rb_b
                cash[i] -= np.nansum(d_rb_b * VWAP[reb_buy_mask,i]) * (1+0.0002)
                
            # lend
            if cash[i] < 0:
                cash_lend -= cash[i]
                cash[i] = 0
            
            # sell
            if np.sum(all_sell_mask) > 0:
                trn.append(np.nansum(holding[all_sell_mask,i] * VWAP[all_sell_mask,i]) * (1-0.0002-0.001)-np.sum(holding[all_sell_mask,i] * Close[all_sell_mask,i]))
                cash[i] += np.nansum(holding[all_sell_mask,i] * VWAP[all_sell_mask,i]) * (1-0.0002-0.001) # 111391565
                holding[all_sell_mask,i] = 0
                
    
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
                d_all_b = np.floor(reb_value[all_buy_mask] / VWAP[all_buy_mask,i-1]/100)*100
                d_all_b[np.isnan(d_all_b)] = 0
                holding[all_buy_mask,i] = d_all_b
                cash[i] -= np.nansum(d_all_b * VWAP[all_buy_mask,i]) * (1+0.0002) # 61828839
                trn.append(np.sum(d_all_b * Close[all_buy_mask,i])-np.nansum(d_all_b * VWAP[all_buy_mask,i]) * (1+0.0002))
            
        
            d_f = future_holding[i] - future_holding[i-1]
            cash[i] += (200*d_f*((CSI500_open[i] + CSI500[i])/2 - CSI500[i]))
            cash[i] += (200*future_holding[i-1]*(CSI500[i-1] - CSI500[i]))
            
            holding_v = holding[:,i] * Close_fill[:,i]
            holding_v[np.isnan(Close_fill[:,i])] = 0
            cash[i] -= aum[i-1]* 0.0001
            aum[i] = (holding_v.sum() + cash[i] - cash_lend) 
            occ.append(200*d_f*((CSI500_open[i] + CSI500[i])/2 - CSI500[i])+200*future_holding[i-1]*(CSI500[i-1] - CSI500[i]))
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
                
def t1(line):
    return (line-np.nanmean(line))/np.nanstd(line)

def t2(s):
    return np.apply_along_axis(t1, 1, s)

%timeit t3 = pd.DataFrame(siw);t3 = t3.mean(axis=1) / t3.std(axis=1)
        

if __name__ == '__main__':
    
    data_dir = './temp_data/'
    Dates = [str(x[0][0]) for x in io.loadmat(f'{data_dir}date.mat').popitem()[1]]
    Tickers = [str(x[0][0]) for x in io.loadmat(f'{data_dir}ticker.mat').popitem()[1]]
    
    Data = []
    data_col = []
    for data in os.listdir(data_dir):
        if 'data' not in data:
            continue
        temp = io.loadmat(f'{data_dir}{data}').popitem()
        data_col.append(temp[0])
        Data.append(temp[1].tolist())
    
    Data = np.array(Data)
    
    # 
    CSI500_ret = io.loadmat(f'{data_dir}csi500_dy_return.mat').popitem()[1].reshape([-1])
    
    price_col = [i for i in range(len(data_col)) if 'price' in data_col[i]]
    Data_ret = Data[price_col,:,1:]/Data[price_col,:,:-1] - CSI500_ret[1:]
    mask = np.isnan(Data)
    Data_ret[np.isnan(Data_ret)] = 1   
    Data_ret = np.cumprod(Data_ret, axis=2)
    Data[price_col,:,1:] = Data_ret
    
    Data[mask] = np.nan
    del Data_ret
    
    # backtest
    CSI500 = pd.Series(io.loadmat(f'{data_dir}csi500_close.mat').popitem()[1].reshape([-1]))
    CSI500_open = pd.Series(io.loadmat(f'{data_dir}csi500_open.mat').popitem()[1].reshape([-1]))
    Close = pd.DataFrame(io.loadmat(f'{data_dir}data_close_price.mat').popitem()[1])
    VWAP = pd.DataFrame(io.loadmat(f'{data_dir}data_vwap_price.mat').popitem()[1])
    Check = io.loadmat(f'{data_dir}data_tickercheck.mat').popitem()[1]
    MV = io.loadmat(f'{data_dir}data_mv.mat').popitem()[1]
    
    Close[Close == 0] = np.nan
    VWAP[VWAP == 0] = np.nan
    CSI500[CSI500 == 0] = np.nan
    CSI500_open[CSI500_open == 0] = np.nan
    Close_fill = Close.ffill(axis=1)
    High = io.loadmat(f'{data_dir}data_high_price.mat').popitem()[1]
    Low = io.loadmat(f'{data_dir}data_low_price.mat').popitem()[1]
    cant = High == Low
    cant_buy = cant * (Low / Close_fill.shift(axis=1) - 1 > 0.09).values
    cant_sell = cant * (High / Close_fill.shift(axis=1) - 1 < -0.09).values
    Close_fill = Close_fill.values
    Close = Close.values
    VWAP = VWAP.values
    CSI500 = CSI500.ffill().bfill()
    CSI500_open = CSI500_open.ffill().bfill()
    MV_weight = MV / np.sum(MV, axis=0)
    
    fturn = io.loadmat(f'{data_dir}data_fturn.mat').popitem()[1] / 100
    fmv = io.loadmat(f'{data_dir}data_fmv.mat').popitem()[1]
    fliq_top = pd.DataFrame(fturn * fmv).rolling(5,axis=1).mean()
    fliq_top[pd.isnull(fliq_top)] = pd.DataFrame(fturn * fmv)[pd.isnull(fliq_top)]
    fliq_top = fliq_top.values
    
    siw = pd.read_csv('D:/Projects/Technical/Factors/RSI_bot_10_22_5_em.csv',
                      index_col = 0, header = 0)
    siw = siw.T.values
    
    #%%
    siw[Check==0] = -10000
    weights = np.zeros(siw.shape)
    for i in range(siw.shape[1]):
        if i < 1:
            continue
        top = np.argpartition(siw[:,i-1], -50)[-50:]
        weights[top,i] = 1#MV[top,i-1]
    weights /= np.sum(weights,axis=0)
    weights[np.isnan(weights)] = 0

    bm = CSI500 / CSI500.shift(1)
    cash_total = 100000000

    year_point = np.where(pd.Series([int(x[:4]) for x in Dates]).diff() == 1)[0]

    occ = []
    hocc = []
    holding = np.zeros(Close.shape)
    future_holding = np.zeros(Close.shape[1])
    cash_lend = 0
    topV = cash_total
    DD = np.zeros(Close.shape[1])
    maxDD = 0
    risk_level = np.zeros(Close.shape[1])
    cash = np.zeros(Close.shape[1])
    cash[0] = cash_total
    aum = np.zeros(Close.shape[1])
    aum[0] = cash_total
    model_line = aum.copy()
    NV_today = np.zeros(Close.shape[0])
    pb = ProgressBar()
    for i in pb(range(Close.shape[1])):
    
        if i < 1:
            occ.append(0)
            hocc.append(0)
            continue

        cash[i] = cash[i-1]
        holding[:,i] = holding[:,i-1]
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
        
        weight_v = weights[:,i]
        liq_mask = fliq_top[:,i]/(0.8*aum[i-1])
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
        
        h_yes = holding[:,i-1] * Close_fill[:,i-1]
        h_yes[np.isnan(Close_fill[:,i-1])] = 0
        build_speed = 0.8
        reb_value = build_speed*aum[i-1]*weight_v - h_yes
        reb_value[reb_value < 0] *= ~cant_sell[reb_value < 0,i]
        reb_value[reb_value > 0] *= ~cant_buy[reb_value > 0,i]

        liq_mask = np.abs(reb_value) >= fliq_top[:,i-1] * 0.1
        if np.sum(liq_mask) > 0:
            reb_value[liq_mask] = np.sign(reb_value[liq_mask])*fliq_top[liq_mask,i-1]*0.1
        
        turn_mask = np.abs(reb_value / h_yes) < 0.1
        reb_value[turn_mask] = 0
        
        future_holding[i] = np.round((reb_value+h_yes).sum() / CSI500[i-1] / 200)
        
        reb_buy_mask = (reb_value > 0) * (h_yes > 0)
        reb_sell_mask = (reb_value < 0) * (reb_value + h_yes > 0)
        all_buy_mask = (reb_value > 0) * (h_yes == 0)
        all_sell_mask = (reb_value < 0) * (reb_value + h_yes == 0)
        
#        cash_line = 0.15 * aum[i-1]
        
        trn = []
        
        # rebalance sell
        if np.sum(reb_sell_mask) > 0:
            d_rb_s = np.round(reb_value[reb_sell_mask] / Close[reb_sell_mask,i-1],-2)
            d_rb_s[np.isnan(d_rb_s)] = 0
            holding[reb_sell_mask,i] += d_rb_s
            cash[i] += np.nansum(-d_rb_s * VWAP[reb_sell_mask,i]) * (1-0.0002-0.001) # 72984458            

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
            d_rb_b = np.round(reb_value[reb_buy_mask] / Close[reb_buy_mask,i-1],-2)
            d_rb_b[np.isnan(d_rb_b)] = 0
            holding[reb_buy_mask,i] += d_rb_b
            cash[i] -= np.nansum(d_rb_b * VWAP[reb_buy_mask,i]) * (1+0.0002)
            
        # lend
        if cash[i] < 0:
            cash_lend -= cash[i]
            cash[i] = 0
        
        # sell
        if np.sum(all_sell_mask) > 0:
            cash[i] += np.nansum(holding[all_sell_mask,i] * VWAP[all_sell_mask,i]) * (1-0.0002-0.001) # 111391565
            holding[all_sell_mask,i] = 0
            

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
            d_all_b = np.floor(reb_value[all_buy_mask] / VWAP[all_buy_mask,i-1]/100)*100
            d_all_b[np.isnan(d_all_b)] = 0
            holding[all_buy_mask,i] = d_all_b
            cash[i] -= np.nansum(d_all_b * VWAP[all_buy_mask,i]) * (1+0.0002) # 61828839
        
    
        d_f = future_holding[i] - future_holding[i-1]
        cash[i] += (200*d_f*((CSI500_open[i] + CSI500[i])/2 - CSI500[i]))
        cash[i] += (200*future_holding[i-1]*(CSI500[i-1] - CSI500[i]))
        
        holding_v = holding[:,i] * Close_fill[:,i]
        holding_v[np.isnan(Close_fill[:,i])] = 0
        cash[i] -= aum[i-1]* 0.0001
        aum[i] = (holding_v.sum() + cash[i] - cash_lend) 
        occ.append(200*d_f*((CSI500_open[i] + CSI500[i])/2 - CSI500[i])+200*future_holding[i-1]*(CSI500[i-1] - CSI500[i]))
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
    holding = np.zeros(Close.shape)
    future_holding = np.zeros(Close.shape[1])
    cash_lend = 0
    topV = cash_total
    DD = np.zeros(Close.shape[1])
    maxDD2 = 0
    cash = np.zeros(Close.shape[1])
    cash[0] = cash_total
    aum = np.zeros(Close.shape[1])
    aum[0] = cash_total
    NV_today = np.zeros(Close.shape[0])
    pb = ProgressBar()
    for i in pb(range(Close.shape[1])):
        if i < 1:
            occ.append(0)
            hocc.append(0)
            continue

        cash[i] = cash[i-1]
        holding[:,i] = holding[:,i-1]
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
        
        weight_v = weights[:,i]
        liq_mask = fliq_top[:,i]/(0.8*aum[i-1]*(1-0.5*risk_level[i]))
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
        
        h_yes = holding[:,i-1] * Close_fill[:,i-1]
        h_yes[np.isnan(Close_fill[:,i-1])] = 0
        build_speed = 0.8
        reb_value = build_speed*aum[i-1]*weight_v*(1-0.5*risk_level[i]) - h_yes
        reb_value[reb_value < 0] *= ~cant_sell[reb_value < 0,i]
        reb_value[reb_value > 0] *= ~cant_buy[reb_value > 0,i]

        liq_mask = np.abs(reb_value) >= fliq_top[:,i-1] * 0.1
        if np.sum(liq_mask) > 0:
            reb_value[liq_mask] = np.sign(reb_value[liq_mask])*fliq_top[liq_mask,i-1]*0.1
        
        turn_mask = np.abs(reb_value / h_yes) < 0.1
        reb_value[turn_mask] = 0
        
        future_holding[i] = np.round((reb_value+h_yes).sum() / CSI500[i-1] / 200)
        
        reb_buy_mask = (reb_value > 0) * (h_yes > 0)
        reb_sell_mask = (reb_value < 0) * (reb_value + h_yes > 0)
        all_buy_mask = (reb_value > 0) * (h_yes == 0)
        all_sell_mask = (reb_value < 0) * (reb_value + h_yes == 0)
        
#        cash_line = 0.15 * aum[i-1]
        
        trn = []
        
        # rebalance sell
        if np.sum(reb_sell_mask) > 0:
            d_rb_s = np.round(reb_value[reb_sell_mask] / Close[reb_sell_mask,i-1],-2)
            d_rb_s[np.isnan(d_rb_s)] = 0
            holding[reb_sell_mask,i] += d_rb_s
            cash[i] += np.nansum(-d_rb_s * VWAP[reb_sell_mask,i]) * (1-0.0002-0.001) # 72984458
            

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
            d_rb_b = np.round(reb_value[reb_buy_mask] / Close[reb_buy_mask,i-1],-2)
            d_rb_b[np.isnan(d_rb_b)] = 0
            holding[reb_buy_mask,i] += d_rb_b
            cash[i] -= np.nansum(d_rb_b * VWAP[reb_buy_mask,i]) * (1+0.0002)
            
        # lend
        if cash[i] < 0:
            cash_lend -= cash[i]
            cash[i] = 0
        
        # sell
        if np.sum(all_sell_mask) > 0:
            trn.append(np.nansum(holding[all_sell_mask,i] * VWAP[all_sell_mask,i]) * (1-0.0002-0.001)-np.sum(holding[all_sell_mask,i] * Close[all_sell_mask,i]))
            cash[i] += np.nansum(holding[all_sell_mask,i] * VWAP[all_sell_mask,i]) * (1-0.0002-0.001) # 111391565
            holding[all_sell_mask,i] = 0
            

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
            d_all_b = np.floor(reb_value[all_buy_mask] / VWAP[all_buy_mask,i-1]/100)*100
            d_all_b[np.isnan(d_all_b)] = 0
            holding[all_buy_mask,i] = d_all_b
            cash[i] -= np.nansum(d_all_b * VWAP[all_buy_mask,i]) * (1+0.0002) # 61828839
            trn.append(np.sum(d_all_b * Close[all_buy_mask,i])-np.nansum(d_all_b * VWAP[all_buy_mask,i]) * (1+0.0002))
        
    
        d_f = future_holding[i] - future_holding[i-1]
        cash[i] += (200*d_f*((CSI500_open[i] + CSI500[i])/2 - CSI500[i]))
        cash[i] += (200*future_holding[i-1]*(CSI500[i-1] - CSI500[i]))
        
        holding_v = holding[:,i] * Close_fill[:,i]
        holding_v[np.isnan(Close_fill[:,i])] = 0
        cash[i] -= aum[i-1]* 0.0001
        aum[i] = (holding_v.sum() + cash[i] - cash_lend) 
        occ.append(200*d_f*((CSI500_open[i] + CSI500[i])/2 - CSI500[i])+200*future_holding[i-1]*(CSI500[i-1] - CSI500[i]))
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
        
    
    dps = pd.Series(aum,index=pd.to_datetime(Dates))
    temp_dps = pd.Series(temp_aum,index=pd.to_datetime(Dates))
    ax = (dps / dps[0]).plot()
    (temp_dps / temp_dps[0]).plot()
    ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                  risk_level[np.newaxis], cmap='Reds', alpha=0.2)
    
    
    




