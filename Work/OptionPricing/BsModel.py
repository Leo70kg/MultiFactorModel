# -*- coding: utf-8 -*-
# Leo70kg
from __future__ import division
from __future__ import division
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


"""*********************用于计算股票期权的Black Scholes model************************"""


# ----------------------------------------------------------------------
def dOne(s, k, r, q, t, v, date='calendar'):
    """计算bs模型中的d1"""
    if date == 'calendar':
        T = t / 365
    else:
        T = t / 252

    return (np.log(s / k) + (0.5 * v ** 2 + r - q) * T) / (v * np.sqrt(T))


# ----------------------------------------------------------------------
def bsPrice(s, k, r, q, t, v, cp, date='calendar'):
    """使用bs模型计算期权的价格"""
    if date == 'calendar':
        T = t / 365
    else:
        T = t / 252

    d1 = dOne(s, k, r, q, t, v, date)
    d2 = d1 - v * np.sqrt(T)
    price = cp * (s * np.exp(-q * T) * norm.cdf(cp * d1) - k * np.exp(-r * T) * norm.cdf(cp * d2))
    return price


# ----------------------------------------------------------------------
def bsDelta(s, k, r, q, t, v, cp, date='calendar'):
    """使用bs模型计算期权的Delta"""

    if date == 'calendar':
        T = t / 365
    else:
        T = t / 252

    d1 = dOne(s, k, r, q, t, v, date)
    delta = cp * norm.cdf(cp * d1) * np.exp(-q * T)
    return delta


# ----------------------------------------------------------------------
def bsGamma(s, k, r, q, t, v, cp, date='calendar'):
    """使用bs模型计算期权的Gamma"""

    if date == 'calendar':
        T = t / 365
    else:
        T = t / 252
    d1 = dOne(s, k, r, q, t, v, date)
    gamma = norm.pdf(d1) / (s * v * np.sqrt(T)) * np.exp(-q * T)
    return gamma


# ----------------------------------------------------------------------
def bsVega(s, k, r, q, t, v, cp, date='calendar'):
    """使用bs模型计算期权的Vega"""
    if date == 'calendar':
        T = t / 365
    else:
        T = t / 252
    d1 = dOne(s, k, r, q, t, v, date)
    vega = s * norm.pdf(d1) * np.sqrt(T) * np.exp(-q * T)
    return vega


# ----------------------------------------------------------------------
def bsTheta(s, k, r, q, t, v, cp, date='calendar'):
    """使用bs模型计算期权的Theta"""
    if date == 'calendar':
        T = t / 365
    else:
        T = t / 252
    d1 = dOne(s, k, r, q, t, v, date)
    d2 = d1 - v * np.sqrt(T)
    theta = -(s * v * norm.pdf(d1) * np.exp(-q * T)) / (2 * np.sqrt(T)) - \
            cp * r * k * np.exp(-r * T) * norm.cdf(cp * d2) + \
            cp * q * s * norm.cdf(cp * d1) * np.exp(-q * T)

    return theta


def impliedVolatility(S0, K, r, q, T, cp, price, date='calendar'):
    def func(sigma):
        return bsPrice(S0, K, r, q, T, sigma, cp, date) - price

    impliedSigma = brentq(func, 0.001, 1.5)
    return impliedSigma


def FXoptionPrice(S, K, rd, rf, t, sigma, cp):
    T = t / 365
    d1 = (np.log(S / K) + (rd - rf + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = cp * S * np.exp(-rf * T) * norm.cdf(cp * d1) - cp * K * np.exp(-rd * T) * norm.cdf(cp * d2)

    return price


def bmPriceWithDepositRatio(s, k, r, t, v, cp, dr, cl):
    T = t / 252
    d1 = (np.log(s / k) + (cl * dr + v ** 2 / 2) * T) / (v * np.sqrt(T))
    d2 = (np.log(s / k) + (cl * dr - v ** 2 / 2) * T) / (v * np.sqrt(T))

    price = cp * np.exp(-(r - cp * cl * dr) * T) * s * norm.cdf(cp * d1) - cp * np.exp(-r * T) * k * norm.cdf(cp * d2)

    return price


def numericalDerivative(f, x0, h):
    if x0 == 0:
        derivative = (f(h) - f(-h)) / (2 * h)
    else:
        derivative = (f(x0 * (1 + h)) - f(x0 * (1 - h))) / (2 * x0 * h)
    return derivative


def Delta(S0, k, r, t, v, cp, dr, cl):
    h = 1e-5

    def fPrice(s):
        return bmPriceWithDepositRatio(s, k, r, t, v, cp, dr, cl)

    delta = numericalDerivative(fPrice, S0, h)

    return delta


if __name__ == '__main__':
    S = 524.1
    K = S
    sigma = 0.5
    r = 0.03
    t = 11
    cp = 1
    q = 0.03
    price = 11.47
    print impliedVolatility(S, K, r, q, t, cp, price, date='calar')



