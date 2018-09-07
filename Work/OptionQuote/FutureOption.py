# -*- coding: utf-8 -*-
# Leo70kg
from __future__ import division
from dateutil.relativedelta import relativedelta
from OptionCalBase import OptionCalBase
import numpy as np
import datetime
import pymysql
from Database import Util


class VolCal(OptionCalBase):

    freq = 15  # 选取分钟级别数据的频率
    m = 6  # m是选取数据时间段的大小，单位为月

    EndDate = datetime.date(2018, 8, 31)
    StartDate = EndDate - relativedelta(months=+m)

    def __init__(self, code, t):

        self.code = code
        self.db = self.connect2db()
        self.cursor = self.db.cursor()
        self.T = t

        self.num = self.get_num_of_data_per_day()
        window = self.num * t
        self.tick = Util.get_tick_size(self.cursor, self.code)
        self.S = Util.get_close_price(self.cursor, self.code, VolCal.EndDate)
        self.K = self.S
        self.baseT = np.maximum(11, t)

        """计算分钟级别数据波动率"""
        minute_ret = Util.get_min_ret(self.cursor, code, VolCal.freq, VolCal.StartDate, VolCal.EndDate)
        self.minute_vol_series = minute_ret.rolling(window=window, center=False).std().dropna() * np.sqrt(252 * self.num
                                                                                                          )

        """计算日级别波动率"""
        daily_ret = Util.get_daily_ret(self.cursor, code, VolCal.StartDate, VolCal.EndDate)
        self.daily_vol_series = daily_ret.rolling(window=self.baseT, center=False).std().dropna() * np.sqrt(252)

    def connect2db(self):

        db = pymysql.connect(host='192.168.16.23', user='zuoyou', password='bhrsysp', db='future_information',
                             charset='utf8')
        return db

    def get_close_price(self):

        result = Util.get_close_price(self.cursor, self.code, VolCal.EndDate)

        return result

    def get_num_of_data_per_day(self):
        """获取每日分钟级别数据的个数"""
        code_split = Util.get_code_split(self.code)
        future_info = Util.load_future_trade_time()
        num = future_info[code_split[1]][code_split[0]][3][2]

        return num

    def base_vol(self):
        """以当日最新的已实现波动率作为基准波动率"""
        param = self.num * 5

        today_daily_vol = self.daily_vol_series.iloc[-1]
        mean_daily_vol = self.daily_vol_series[-252:].ewm(span=126).mean().iloc[-1]

        base_vol = (self.minute_vol_series[(-126*self.num):].ewm(span=param).mean().iloc[-1] + today_daily_vol +
                    mean_daily_vol) / 3

        return base_vol

    def hedge_vol(self):

        vol = 16.74 * self.tick / self.S + 0.0033

        return vol

    def vol_batch(self, param1=1.5, param2=1):
        """最终写入报价excel文件中所需要的各个波动率数值"""
        today_minute_vol = self.minute_vol_series.iloc[-1]

        vol_of_vol = self.daily_vol_series.rolling(window=self.baseT, center=False).std().dropna()
        today_daily_vol = self.daily_vol_series.iloc[-1]
        today_daily_vol_pct = self.daily_vol_series.rank().iloc[-1] / len(self.daily_vol_series)

        today_vol_of_vol_pct = vol_of_vol.rank().iloc[-1] / len(vol_of_vol)
        mean_vol_of_vol = np.mean(vol_of_vol)

        """*********************************确定spread宽度以及偏度***************************************"""
        """最大宽度为2倍的fixedCoff2, 最小宽度为1倍的最大宽度为2倍的fixedCoff2"""
        per75vol = np.percentile(self.daily_vol_series, 75)  # 波动率75%分位
        per50vol = np.percentile(self.daily_vol_series, 50)  # 波动率50%分位
        per25vol = np.percentile(self.daily_vol_series, 25)  # 波动率25%分位

        """根据当前波动率的波动率所处的水平确定溢出系数"""
        param = (param1 - param2) * today_vol_of_vol_pct + param2

        fixed_coff = param * mean_vol_of_vol

        if today_daily_vol >= per75vol:
            ask_extra_vol = 0
            bid_extra_vol = -fixed_coff / 2

        elif (today_daily_vol < per75vol) & (today_daily_vol >= per50vol):
            ask_extra_vol = (2 * today_daily_vol_pct - 0.5) * (0.75 - today_daily_vol_pct) / \
                            (0.75 - 0.5) * fixed_coff / 4
            bid_extra_vol = -(2 * today_daily_vol_pct - 0.5) * (2 - (0.75 - today_daily_vol_pct) /
                                                                (0.75 - 0.5)) * fixed_coff / 4

        elif (today_daily_vol < per50vol) & (today_daily_vol >= per25vol):
            ask_extra_vol = (-2 * today_daily_vol_pct + 1.5) * (2 - (today_daily_vol_pct - 0.25) / (0.5 - 0.25)) * \
                            fixed_coff / 4
            bid_extra_vol = -(-2 * today_daily_vol_pct + 1.5) * (today_daily_vol_pct - 0.25) / \
                            (0.5 - 0.25) * fixed_coff / 4

        else:
            ask_extra_vol = fixed_coff / 2
            bid_extra_vol = 0

        return [ask_extra_vol, bid_extra_vol], [per25vol, per50vol, per75vol, today_minute_vol, today_daily_vol,
                                                param, mean_vol_of_vol]



