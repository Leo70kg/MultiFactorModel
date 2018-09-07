# -*- coding: utf-8 -*-
# Leo70kg
from __future__ import division
import numpy as np
import pandas as pd
from Util import UtilDF


class Alpha(object):

    def __init__(self, data_df):
        self.data = data_df

    def alpha001(self, n=6):

        c1 = self.data.volume.apply(np.log).diff(1)
        c2 = (self.data.close - self.data.open) / self.data.open
        rank1 = c1.rank()
        rank2 = c2.rank()

        corr = pd.rolling_corr(rank1, rank2, n)
        result = -corr

        return result

    def alpha002(self):

        c1 = (self.data.close - self.data.low) - (self.data.high - self.data.close)
        c2 = self.data.high - self.data.low
        c3 = c1 / c2

        result = -(c3.diff(1))

        return result

    def alpha003(self):

        pre_close = self.data.shift(1)
        c1 = np.minimum(self.data.low, pre_close)
        c2 = np.maximun(self.data.high, pre_close)

        c3 = UtilDF.if_else_func(self.data.close > pre_close, c1, c2)
        result = UtilDF.if_else_func(self.data.close == pre_close, 0, self.data.close - c3)

        return result

    def alpha004(self):

        c1 = self.data.close.rolling(8).sum() / 8
        c2 = self.data.close.rolling(2).sum() / 2
        c3 = self.data.close.rolling(8).std()
        c4 = self.data.volume / self.data.volume.rolling(20).mean()

        con1 = c4 >= 1
        c5 = UtilDF.if_else_func(con1, 1, -1)

        con2 = c2 < (c1 - c3)
        c6 = UtilDF.if_else_func(con2, 1, c5)

        con3 = (c1 + c3) < c2
        result = UtilDF.if_else_func(con3, -1, c6)

        return result

    def alpha005(self):

        length = len(self.close_series)

        volume_rank_lis = []
        high_rank_lis = []
        for i in range(length):

            if i < (5 - 1):
                volume_rank_lis.append(np.nan)
                high_rank_lis.append(np.nan)

            elif i == (length - 1):
                s1 = self.volume_series[-5:].rank().iloc[-1]
                s2 = self.high_series[-5:].rank().iloc[-1]

                volume_rank_lis.append(s1)
                high_rank_lis.append(s2)

            else:
                s1 = self.volume_series[i-4:i+1].rank().iloc[-1]
                s2 = self.high_series[i-4:i+1].rank().iloc[-1]

                volume_rank_lis.append(s1)
                high_rank_lis.append(s2)

        volume_rank_series = pd.Series(volume_rank_lis)
        high_rank_series = pd.Series(high_rank_lis)

        corr = pd.rolling_corr(volume_rank_series, high_rank_series, 5)
        result = -(corr.rolling(3).max())

        return result

    def alpha006(self):

        c1 = self.open_series * 0.85
        c2 = self.high_series * 0.15

        c3 = np.sign((c1 + c2).diff(4))

    def alpha007(self):

        c1 = np.maximum(self.vwap_series - self.close_series, 3)
        c2 = np.minimum(self.vwap_series - self.close_series, 3)
        c3 = self.volume_series.diff(3)

        result = (c1.rank() + c2.rank()) * c3.rank()

        return result

    def alpha008(self):

        c1 = (self.high_series + self.low_series) / 2 * 0.2 + (self.vwap_series * 0.8)
        c2 = c1.diff(4) * -1

        result = c2.rank()

        return result

    def alpha009(self):

        delay_high = self.high_series[:-1].reset_index(drop=True)
        delay_low = self.low_series[:-1].reset_index(drop=True)
        high = self.high_series[1:].reset_index(drop=True)
        low = self.low_series[1:].reset_index(drop=True)
        volume = self.volume_series[1:].reset_index(drop=True)

        c1 = (high + low) / 2 - (delay_high + delay_low) / 2
        c2 = (high - low) / volume

        ta.SMA()

    def alpha010(self):

        c1 = self.ret_series.rolling(20).std()

        c2 = (function.if_or_not_func(self.ret_series < 0, c1, self.close_series)) ** 2
        c3 = c2.rolling(5).max()

        c4 = c3.rank()

        return c4

    def alpha011(self):

        c1 = (self.close_series - self.low_series) - (self.high_series - self.close_series)
        c2 = c1 / (self.high_series - self.low_series) * self.volume_series
        c3 = c2.rolling(6).sum()

        return c3

    def alpha012(self):

        c1 = self.vwap_series.rolling(10).sum() / 10
        c2 = self.open_series - c1
        c3 = c2.rank()

        c4 = np.abs(self.close_series - self.vwap_series)
        c5 = -(c4.rank())

        result = c4 * c5

        return result

    def alpha013(self):

        c1 = self.high_series - self.low_series
        c2 = c1 ** 0.5 - self.vwap_series

        return c2

    def alpha014(self):

        c1 = function.delay(self.close_series, 5)
        c2 = function.advance(self.close_series, 5)

        result = c2 - c1

        return result

    def alpha015(self):

        c1 = function.delay(self.close_series, 1)
        c2 = self.open_series / c1 - 1

        return c2

    def alpha016(self):

        c1 = self.volume_series.rank()
        c2 = self.vwap_series.rank()

        c3 = c1.rolling(5).corr(c2)
        c4 = c3.rank()
        c5 = -(c4.rolling(5).max())

        return c5

    def alpha017(self):

        c1 = np.maximun(self.vwap_series, 15)
        c2 = (self.vwap_series - c1).rank()
        c3 = self.close_series.diff(5)

        result = c2 ** c3

        return result

    def alpha018(self):

        c1 = function.delay(self.close_series, 5)
        c2 = function.advance(self.close_series, 5)

        result = c2 / c1

        return result

    def alpha019(self):

        c1 = function.delay(self.close_series, 5)
        c2 = function.advance(self.close_series, 5)

        condition1 = (c2 == c1)
        c3 = (c2 - c1) / c2
        c4 = function.if_or_not_func(condition1, 0, c3)

        condition2 = (c2 < c1)
        c5 = (c2 - c1) / c1
        c6 = function.if_or_not_func(condition2, c5, c4)

        return c6

    def alpha020(self):

        c1 = function.delay(self.close_series, 6)
        c2 = function.advance(self.close_series, 6)

        c3 = (c2 - c1) / c1 * 100

        return c3

    def alpha021(self):

        c1 = self.close_series.rolling(6).mean()
        c2 = pd.Series(range(1, 7))
        beta_lis = []
        for i in range(len(c1)-6):

            c3 = c1[i:i+6]
            beta = function.linear_reg(c3, c2)[0]
            beta_lis.append(beta)

        return pd.Series(beta_lis)

    def alpha022(self):

        c1 = (self.close_series - self.close_series.rolling(6).mean()) / self.close_series.rolling(6).mean()
        c2 = function.delay(c1, 3)
        c3 = function.advance(c1, 3)

        result = function.sma(c3 - c2, 12, 1)

        return result

    def alpha023(self):

        c1 = function.delay(self.close_series, 1)
        c2 = function.advance(self.close_series, 1)
        c3 = c2.rolling(20).std()

        condition1 = c2 > c1
        condition2 = c2 <= c1

        c4 = function.if_or_not_func(condition1, c3, 0)
        c5 = function.if_or_not_func(condition2, c3, 0)

        c6 = function.sma(c4, 20, 1)
        c7 = function.sma(c5, 20, 1)

        result = c6 / (c6 + c7) * 100

        return result

    def alpha024(self):

        c1 = function.delay(self.close_series, 5)
        c2 = function.advance(self.close_series, 5)

        result = function.sma(c2 - c1, 5, 1)

        return result

    def alpha025(self):

        c1 = self.volume_series / self.volume_series.rolling(20).mean()
        c2 = function.wma(c1, 9)
        c3 = 1 - c2.rank()
        c4 = -1 * (self.close_series.diff(7).rank()) * c3
        c5 = 1 + self.ret_series.rolling(250).sum().rank()

        result = c4 * c5

        return result

    def alpha026(self):

        delay_close = function.delay(self.close, 5)
        close = function.advance(self.close_series, 5)
        vwap = function.advance(self.vwap_series, 5)

        c1 = close.rolling(7).sum() / 7
        c2 = vwap.rolling(230).corr(delay_close)

        return c1 + c2

    def alpha027(self):

        delay_close1 = function.delay(self.close_series, 3)
        delay_close2 = function.delay(self.close_series, 6)
        close1 = function.advance(self.close_series, 3)
        close2 = function.advance(self.close_series, 6)

        c1 = (close1 - delay_close1) / delay_close1 * 100
        c2 = (close2 - delay_close2) / delay_close2 * 100

        result = function.wma(c1 + c2, 12)

        return result