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