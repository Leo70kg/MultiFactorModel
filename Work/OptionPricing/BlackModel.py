# -*- coding: utf-8 -*-
# Leo70kg
from __future__ import division
from __future__ import division
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


"""*********************用于计算股票期权的Black Model************************"""


# ----------------------------------------------------------------------
def dOne(F, k, r, t, v, date='calendar'):
    """计算bl模型中的d1"""
    if date == 'calendar':
        T = t / 365
    else:
        T = t / 252

    return (np.log(F / k) + 0.5 * v ** 2 * T) / (v * np.sqrt(T))


# ----------------------------------------------------------------------
def blPrice(F, k, r, t, v, cp, date='calendar'):
    """使用bl模型计算期权的价格"""
    if date == 'calendar':
        T = t / 365
    else:
        T = t / 252

    d1 = dOne(F, k, r, t, v, date)
    d2 = d1 - v * np.sqrt(T)
    price = np.exp(-r * T) * cp * (F * norm.cdf(cp * d1) - k * norm.cdf(cp * d2))
    return price


# ----------------------------------------------------------------------
def blDelta(F, k, r, t, v, cp, date='calendar'):
    """使用bl模型计算期权的Delta"""

    if date == 'calendar':
        T = t / 365
    else:
        T = t / 252

    def indice(x):
        if x == 1:
            return 0
        else:
            return -1

    d1 = dOne(F, k, r, t, v, date)
    delta = (norm.cdf(d1) + indice(cp)) * np.exp(-r * T)
    return delta


# ----------------------------------------------------------------------
def blGamma(F, k, r, t, v, cp, date='calendar'):
    """使用bl模型计算期权的Gamma"""

    if date == 'calendar':
        T = t / 365
    else:
        T = t / 252
    d1 = dOne(F, k, r, t, v, date)
    gamma = norm.pdf(d1) / (F * v * np.sqrt(T)) * np.exp(-r * T)
    return gamma


# ----------------------------------------------------------------------
def blVega(F, k, r, t, v, cp, date='calendar'):
    """使用bl模型计算期权的Vega"""
    if date == 'calendar':
        T = t / 365
    else:
        T = t / 252
    d1 = dOne(F, k, r, t, v, date)
    vega = F * norm.pdf(d1) * np.sqrt(T) * np.exp(-r * T)
    return vega


# ----------------------------------------------------------------------
def blTheta(F, k, r, t, v, cp, date='calendar'):
    """使用bl模型计算期权的Theta"""
    if date == 'calendar':
        T = t / 365
    else:
        T = t / 252
    d1 = dOne(F, k, r, t, v, date)
    d2 = d1 - v * np.sqrt(T)
    theta = -(F * v * norm.pdf(d1) * np.exp(-r * T)) / (2 * np.sqrt(T)) - \
            cp * r * k * np.exp(-r * T) * norm.cdf(cp * d2) + \
            cp * r * F * np.exp(-r * T) * norm.cdf(cp * d1)

    return theta


def impliedVolatility(F0, K, r, T, cp, price, date='calendar'):
    def func(sigma):
        return blPrice(F0, K, r, T, sigma, cp, date) - price

    impliedSigma = brentq(func, 0.001, 1.5)
    return impliedSigma


if __name__ == '__main__':
    F = 2660.41
    K = F
    sigma = 0.5
    r = 0.03
    t = 22
    cp = 1
    price = 3.33 * 0.01 * F
    print impliedVolatility(F, K, r, t, cp, price, date='calar')