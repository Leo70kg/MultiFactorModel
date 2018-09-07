# _*_ coding:utf-8 _*_
# Leo70kg
from __future__ import division
from WindPy import w
import pandas as pd
import numpy as np
from Database import Util
import datetime
from FutureOption import VolCal
from QuoteBase import QuoteBase
from OptionPricing import BsModel
import time
from scipy.optimize import curve_fit


def batch_vol_lis(code, maturity_lis, sell):
    """针对一系列到期时间对最终报价波动率进行线性拟合
       sell指作为期权的卖方，为True or False
    """

    def func(x, c, d):
        return c * x + d

    pricing_vol_lis = []
    today_daily_vol_lis = []
    base_vol_lis = []
    vol_batch_lis = []

    for t in maturity_lis:

        vol = VolCal(code, t)
        s = vol.get_close_price()

        hedge_vol = vol.hedge_vol()
        base_vol = vol.base_vol()
        batch_vol = vol.vol_batch()
        _vol_lis = batch_vol[1] + [s]
        _vol_lis.insert(5, base_vol)
        _vol_lis.insert(6, hedge_vol)
        _vol_lis.insert(7, batch_vol[0][0])

        vol_lis = _vol_lis
        diff_vol = base_vol - vol_lis[4]
        if sell:
            if 0.03 < diff_vol < 0.06:
                pricing_vol = base_vol + hedge_vol + batch_vol[0][0]

            elif diff_vol >= 0.06:
                pricing_vol = base_vol + hedge_vol

            else:
                pricing_vol = (base_vol + hedge_vol + batch_vol[0][0]) * 1.1

        else:
            pricing_vol = np.minimum(base_vol, vol_lis[4]) - hedge_vol - batch_vol[0][1]

        pricing_vol_lis.append(pricing_vol)
        today_daily_vol_lis.append(vol_lis[4])
        base_vol_lis.append(base_vol)

        vol_batch_lis.append(vol_lis)

    popt, pcov = curve_fit(func, np.array(maturity_lis), np.array(pricing_vol_lis))
    c1 = popt[0]
    c2 = popt[1]

    f_lis = []
    for k in range(len(maturity_lis)):

        if k > 5:
            new_vol = c1 * maturity_lis[k] + c2
        else:
            new_vol = pricing_vol_lis[k]

        # pricing_vol_lis[k] = np.minimum(new_vol, today_daily_vol_lis[k] + 0.03)
        pricing_vol_lis[k] = np.minimum(new_vol, (today_daily_vol_lis[k] + base_vol_lis[k]) / 2 + 0.03)
        lis = [pricing_vol_lis[k]] + vol_batch_lis[k]
        f_lis.append(lis)

    return f_lis


def price_batch_vol(code, maturity_lis, r, q, sell=True):
    """s为标的现价，r为无风险利率，q为红利率"""
    batch_vol = batch_vol_lis(code, maturity_lis, sell)
    result = []
    for i in range(len(maturity_lis)):

        _price = BsModel.bsPrice(batch_vol[i][-1], batch_vol[i][-1], r, q, maturity_lis[i], batch_vol[i][0], 1,
                                 'trading') / batch_vol[i][-1]

        lis = [_price] + batch_vol[i]
        result.append(lis)

    return result


def quote_df(name_lis):

    wind_price_df = pd.DataFrame(columns=[u'公司名称', u'品种', u'品种代码',
                                          u'期权类型', u'行权价', u'到期日/交易期限',
                                          u'最小交易单位', u'买价', u'卖价',
                                          u'标的价格', u'报价日期'], index=range(36 * 2))

    official_price_df1 = pd.DataFrame(columns=[u'主力合约名称', u'次主力合约名称', u'1天权利金比例',
                                               u'3天权利金比例', u'1周权利金比例', u'2周权利金比例',
                                               u'1月权利金比例'], index=name_lis)

    official_price_df2 = pd.DataFrame(columns=[u'主力合约名称', u'次主力合约名称', u'1天权利金比例',
                                               u'3天权利金比例',
                                               u'1周权利金比例', u'2周权利金比例',
                                               u'3周权利金比例', u'4周权利金比例',
                                               u'5周权利金比例', u'6周权利金比例'], index=name_lis)

    official_price_df3 = pd.DataFrame(columns=[u'主力合约名称', u'次主力合约名称', u'1天权利金比例',
                                               u'3天权利金比例', u'1周权利金比例', u'2周权利金比例',
                                               u'1月权利金比例'], index=name_lis)

    official_price_df4 = pd.DataFrame(columns=[u'主力合约名称', u'次主力合约名称', u'1天权利金比例',
                                               u'2天权利金比例', u'5天权利金比例'], index=name_lis)

    official_vol_df = pd.DataFrame(columns=[u'主力合约名称', u'次主力合约名称', u'1天报价波动率',
                                            u'2天报价波动率', u'3天报价波动率', u'5天报价波动率',
                                            u'1周报价波动率', u'2周报价波动率', u'3周报价波动率',
                                            u'4周报价波动率', u'1月报价波动率', u'5周报价波动率',
                                            u'6周报价波动率'], index=name_lis)

    internal_price_df = pd.DataFrame(columns=[u'主力合约名称', u'次主力合约名称',
                                              u'价格百分比(1天)', u'报价波动率(1天)', u'25%波动率(1天)', u'50%波动率(1天)',
                                              u'75%波动率(1天)', u'今日分钟波动率(1天)', u'今日日波动率(1天)',
                                              u'基准波动率(1天)', u'对冲波动率(1天)',
                                              u'溢价波动率(1天)', u'溢价倍数(1天)', u'平均波动率的波动率(1天)',

                                              u'价格百分比(2天)', u'报价波动率(2天)', u'25%波动率(2天)', u'50%波动率(2天)',
                                              u'75%波动率(2天)', u'今日分钟波动率(2天)', u'今日日波动率(2天)',
                                              u'基准波动率(2天)', u'对冲波动率(2天)',
                                              u'溢价波动率(2天)', u'溢价倍数(2天)', u'平均波动率的波动率(2天)',

                                              u'价格百分比(3天)', u'报价波动率(3天)', u'25%波动率(3天)', u'50%波动率(3天)',
                                              u'75%波动率(3天)', u'今日分钟波动率(3天)', u'今日日波动率(3天)',
                                              u'基准波动率(3天)', u'对冲波动率(3天)',
                                              u'溢价波动率(3天)', u'溢价倍数(3天)', u'平均波动率的波动率(3天)',

                                              u'价格百分比(5天)', u'报价波动率(5天)', u'25%波动率(5天)', u'50%波动率(5天)',
                                              u'75%波动率(5天)', u'今日分钟波动率(5天)', u'今日日波动率(5天)',
                                              u'基准波动率(5天)', u'对冲波动率(5天)',
                                              u'溢价波动率(5天)', u'溢价倍数(5天)', u'平均波动率的波动率(5天)',

                                              u'价格百分比(1周)', u'报价波动率(1周)', u'25%波动率(1周)', u'50%波动率(1周)',
                                              u'75%波动率(1周)', u'今日分钟波动率(1周)', u'今日日波动率(1周)',
                                              u'基准波动率(1周)', u'对冲波动率(1周)',
                                              u'溢价波动率(1周)', u'溢价倍数(1周)', u'平均波动率的波动率(1周)',

                                              u'价格百分比(2周)', u'报价波动率(2周)', u'25%波动率(2周)', u'50%波动率(2周)',
                                              u'75%波动率(2周)', u'今日分钟波动率(2周)', u'今日日波动率(2周)',
                                              u'基准波动率(2周)', u'对冲波动率(2周)',
                                              u'溢价波动率(2周)', u'溢价倍数(2周)', u'平均波动率的波动率(2周)',

                                              u'价格百分比(3周)', u'报价波动率(3周)', u'25%波动率(3周)', u'50%波动率(3周)',
                                              u'75%波动率(3周)', u'今日分钟波动率(3周)', u'今日日波动率(3周)',
                                              u'基准波动率(3周)', u'对冲波动率(3周)',
                                              u'溢价波动率(3周)', u'溢价倍数(3周)', u'平均波动率的波动率(3周)',

                                              u'价格百分比(4周)', u'报价波动率(4周)', u'25%波动率(4周)', u'50%波动率(4周)',
                                              u'75%波动率(4周)', u'今日分钟波动率(4周)', u'今日日波动率(4周)',
                                              u'基准波动率(4周)', u'对冲波动率(4周)',
                                              u'溢价波动率(4周)', u'溢价倍数(4周)', u'平均波动率的波动率(4周)',

                                              u'价格百分比(1月)', u'报价波动率(1月)', u'25%波动率(1月)', u'50%波动率(1月)',
                                              u'75%波动率(1月)', u'今日分钟波动率(1月)', u'今日日波动率(1月)',
                                              u'基准波动率(1月)', u'对冲波动率(1月)',
                                              u'溢价波动率(1月)', u'溢价倍数(1月)', u'平均波动率的波动率(1月)',

                                              u'价格百分比(5周)', u'报价波动率(5周)', u'25%波动率(5周)', u'50%波动率(5周)',
                                              u'75%波动率(5周)', u'今日分钟波动率(5周)', u'今日日波动率(5周)',
                                              u'基准波动率(5周)', u'对冲波动率(5周)',
                                              u'溢价波动率(5周)', u'溢价倍数(5周)', u'平均波动率的波动率(5周)',

                                              u'价格百分比(6周)', u'报价波动率(6周)', u'25%波动率(6周)', u'50%波动率(6周)',
                                              u'75%波动率(6周)', u'今日分钟波动率(6周)', u'今日日波动率(6周)',
                                              u'基准波动率(6周)', u'对冲波动率(6周)',
                                              u'溢价波动率(6周)', u'溢价倍数(6周)', u'平均波动率的波动率(6周)'], index=name_lis)

    return official_price_df1, official_price_df2, official_price_df3, official_price_df4, \
        official_vol_df, internal_price_df, wind_price_df


class Quote1(QuoteBase):

    def quote(self):

        r = 0.06
        q = 0.06

        data = Util.load_future_code()
        code_lis = data.keys()
        name_lis = data.values()

        main_code_lis = []
        semi_code_lis = []

        today_date = datetime.date(2018, 9, 4)
        for i in range(len(code_lis)):
            main_code = w.wss(code_lis[i], "trade_hiscode", "tradeDate={:%Y-%m-%d}".format(today_date)).Data[0][0]
            code_split = code_lis[i].split('.')
            code = code_split[0] + '_S' + '.' + code_split[1]
            semi_code = w.wss(code, "trade_hiscode", "tradeDate={:%Y-%m-%d}".format(today_date)).Data[0][0]
            main_code_lis.append(main_code)
            semi_code_lis.append(semi_code)

        official_price_df1, official_price_df2, official_price_df3, official_price_df4, official_vol_df, \
            internal_price_df, wind_price_df = quote_df(name_lis)

        T = [1, 2, 4, 5, 6, 11, 16, 21, 23, 26, 31]

        T_lis1 = [2, 4, 6, 11, 23]
        T_lis2 = [2, 4, 6, 11, 16, 21, 26, 31]
        T_lis3 = [1, 2, 5]

        sc_index = code_lis.index('SC.INE')

        for i in range(len(code_lis)):

            official_price_df1.iloc[i, 0] = main_code_lis[i]
            official_price_df2.iloc[i, 0] = main_code_lis[i]
            official_price_df3.iloc[i, 0] = main_code_lis[i]
            official_price_df4.iloc[i, 0] = main_code_lis[i]
            internal_price_df.iloc[i, 0] = main_code_lis[i]
            official_vol_df.iloc[i, 0] = main_code_lis[i]

            official_price_df1.iloc[i, 1] = semi_code_lis[i]
            official_price_df2.iloc[i, 1] = semi_code_lis[i]
            official_price_df3.iloc[i, 1] = semi_code_lis[i]
            official_price_df4.iloc[i, 1] = semi_code_lis[i]
            internal_price_df.iloc[i, 1] = semi_code_lis[i]
            official_vol_df.iloc[i, 1] = semi_code_lis[i]

            if code_lis[i] == 'SC.INE':
                pricing_vol = 0.3

                for j in range(len(T_lis1)):

                    official_price_df1.iloc[i, j + 2] = BsModel.bsPrice(1, 1, r, q, T_lis1[j], pricing_vol, 1,
                                                                        'trading') / 1

                    official_price_df3.iloc[i, j + 2] = BsModel.bsPrice(1, 1, r, q, T_lis1[j], pricing_vol, 1,
                                                                        'trading') / 1 + 0.0005

                for m in range(len(T_lis2)):
                    official_price_df2.iloc[i, m + 2] = BsModel.bsPrice(1, 1, r, q, T_lis2[m], pricing_vol, 1,
                                                                        'trading') / 1

                for n in range(len(T_lis3)):
                    official_price_df4.iloc[i, n + 2] = BsModel.bsPrice(1, 1, r, q, T_lis3[n], pricing_vol, 1,
                                                                        'trading') / 1

                for k in range(len(T)):
                    official_vol_df.iloc[i, k + 2] = 0.3

            else:

                _price_vol_lis = price_batch_vol(code_lis[i], T, r, q)
                _price_vol_lis1 = price_batch_vol(code_lis[i], T, r, q, False)

                for k in range(len(T)):
                    official_vol_df.iloc[i, k + 2] = _price_vol_lis[k][1]

                    for l in range(12):
                        internal_price_df.iloc[i, 12 * k + l + 2] = _price_vol_lis[k][l]

                for j in range(len(T_lis1)):

                    official_price_df1.iloc[i, j + 2] = _price_vol_lis[T.index(T_lis1[j])][0]
                    official_price_df3.iloc[i, j + 2] = _price_vol_lis[T.index(T_lis1[j])][0] + 0.0005

                for m in range(len(T_lis2)):
                    official_price_df2.iloc[i, m + 2] = _price_vol_lis[T.index(T_lis2[m])][0]

                for n in range(len(T_lis3)):
                    official_price_df4.iloc[i, n + 2] = _price_vol_lis[T.index(T_lis3[n])][0]

                _id = i-1 if i > sc_index else i
                for p in range(2):

                    wind_price_df.iloc[2 * _id + p, 0] = u'渤海融盛资本管理有限公司'
                    wind_price_df.iloc[2 * _id + p, 1] = name_lis[i]
                    wind_price_df.iloc[2 * _id + p, 2] = main_code_lis[i]
                    if p == 0:
                        wind_price_df.iloc[2 * _id + p, 3] = 'C'
                    else:
                        wind_price_df.iloc[2 * _id + p, 3] = 'P'

                    wind_price_df.iloc[2 * _id + p, 4] = _price_vol_lis1[0][-1]
                    wind_price_df.iloc[2 * _id + p, 5] = '1M'
                    wind_price_df.iloc[2 * _id + p, 6] = ' '
                    wind_price_df.iloc[2 * _id + p, 7] = '%.2f %%' % (_price_vol_lis1[8][0] * 100)

                    wind_price_df.iloc[2 * _id + p, 8] = '%.2f %%' % (_price_vol_lis[8][0] * 100)
                    wind_price_df.iloc[2 * _id + p, 9] = _price_vol_lis[8][-1]
                    wind_price_df.iloc[2 * _id + p, 10] = '{:%Y/%m/%d}'.format(
                        w.tdaysoffset(1, datetime.date.today(), '').Data[0][0])

            print(time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time())), ": calculation [", code_lis[i], "]")

        return official_price_df1, official_price_df2, official_price_df3, official_price_df4, \
            internal_price_df, official_vol_df, wind_price_df

    def df2excel(self):

        df = self.quote()
        date = w.tdaysoffset(1, datetime.date.today(), '').Data[0][0]
        writer1 = pd.ExcelWriter(u'D:\Quote\FutureOptionQuote\商品期权报价({:%Y-%m-%d}).xlsx'.format(date))

        writer2 = pd.ExcelWriter(u'D:\Quote\FutureOptionQuote\商品期权报价_1({:%Y-%m-%d}).xlsx'.format(date))

        writer3 = pd.ExcelWriter(u'D:\Quote\FutureOptionQuote\商品期权报价_2({:%Y-%m-%d}).xlsx'.format(date))

        writer4 = pd.ExcelWriter(u'D:\Quote\FutureOptionQuote\商品期权报价_3({:%Y-%m-%d}).xlsx'.format(date))

        writer5 = pd.ExcelWriter(r'D:\Quote\FutureOptionQuote\FutureOptionInternalQuote{:%Y-%m-%d}.xlsx'.format(date))

        writer6 = pd.ExcelWriter(u'D:\Quote\FutureOptionQuote\报价波动率({:%Y-%m-%d}).xlsx'.format(date))

        writer7 = pd.ExcelWriter(u'D:\Quote\FutureOptionQuote\场外期权报价-渤海融盛({:%Y-%m-%d}).xlsx'.format(date))

        df[0].to_excel(writer1, 'Sheet1', na_rep='NA', engine='io.excel.xlsx.writer')
        df[1].to_excel(writer2, 'Sheet1', na_rep='NA', engine='io.excel.xlsx.writer')
        df[2].to_excel(writer3, 'Sheet1', na_rep='NA', engine='io.excel.xlsx.writer')
        df[3].to_excel(writer4, 'Sheet1', na_rep='NA', engine='io.excel.xlsx.writer')
        df[4].to_excel(writer5, 'Sheet1', na_rep='NA', engine='io.excel.xlsx.writer')
        df[5].to_excel(writer6, 'Sheet1', na_rep='NA', engine='io.excel.xlsx.writer')
        df[6].to_excel(writer7, 'Sheet1', na_rep='NA', index=False, engine='io.excel.xlsx.writer')
        writer1.save()
        writer2.save()
        writer3.save()
        writer4.save()
        writer5.save()
        writer6.save()
        writer7.save()


if __name__ == '__main__':
    w.start()
    a = Quote1()
    a.df2excel()

