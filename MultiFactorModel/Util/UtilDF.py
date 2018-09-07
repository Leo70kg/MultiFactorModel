# -*- coding: utf-8 -*-
# Leo70kg
from __future__ import division
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
"""处理pandas中DataFrame数据格式，尤其针对pandas自带的rolling函数中没有的计算进行添加"""


def if_else_func(condition, pd_s1, pd_s2):
    """形如：若condition成立，则等于a；若condition不成立，则等于b。且condition，a，b均为pandas series,长度相同"""

    if isinstance(pd_s1, pd.Series) and isinstance(pd_s2, pd.Series):

        c1 = pd_s1[condition]
        c2 = pd_s2[~condition]
        c3 = pd.concat([c1, c2]).sort_index()

    elif ~isinstance(pd_s1, pd.Series) and isinstance(pd_s2, pd.Series):

        c3 = pd_s2.replace(pd_s2[condition], pd_s1)

    elif isinstance(pd_s1, pd.Series) and ~isinstance(pd_s2, pd.Series):

        c3 = pd_s1.replace(pd_s1[~condition], pd_s2)

    else:

        l1 = len(condition)
        c1 = pd.Series([0] * l1, index=condition.index)
        c2 = c1.replace(c1[condition], pd_s1)
        c3 = c2.replace(c2[~condition], pd_s2)

    return c3


def delay(a, n):
    """a为pandas series，n为向前推的周期数，e.g. n=2为取向前滚动2个周期的数据"""

    result = a[:-n].reset_index(drop=True)

    return result


def advance(a, n):
    """为了匹配delay函数结果的series长度，需要截取series a"""

    result = a[n:].reset_index(drop=True)

    return result


def linear_reg(x, y):
    """进行线性拟合，x，y均为pandas series"""

    regr = LinearRegression()

    try:
        regr.fit(x.reshape(-1, 1), y)
        beta, residual = regr.coef_[0], regr.intercept_

    except ValueError as e:
        print e
        beta, residual = np.nan, np.nan

    return beta, residual


def sma(s, n, m):
    """计算SMA均线，n为周期，m为权重，s为时间序列"""

    i = 0
    while s.iloc[i] is np.nan:
        i = i + 1

    count_lis = []
    for j in range(i, len(s)):

        if j+i < n-1+i:
            count_lis.append(np.nan)
        else:
            s1 = s[j-n+i+1:j+i+1]
            count = s1.iloc[0]

            for k in range(len(s1)):

                count = (s1.iloc[k] * m + count * (n - m)) / n

            count_lis.append(count)

    return pd.Series(count_lis)


def wma(s, n):
    """计算加权移动平均"""

    i = 0
    while s.iloc[i] is np.nan:
        i = i + 1

    count_lis = []
    for j in range(i, len(s)):

        if j < n:
            count_lis.append(np.nan)
        else:
            s1 = s[j-n+1:j]
            count = np.average(s1, weights=range(n, 0, -1))

            count_lis.append(count)

    return pd.Series(count_lis)


if __name__ == '__main__':

    a1 = pd.Series([2,6,4,3,1,7], index=[datetime.date(2018,1,i) for i in range(1,7)])
    b1 = pd.Series([1,9,5,2,1,5], index=[datetime.date(2018,1,i) for i in range(1,7)])
    a = 1
    b = 3
    print if_else_func(a1>b1,a,b)