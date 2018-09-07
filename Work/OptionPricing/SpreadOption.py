# -*- coding: utf-8 -*-
# Leo70kg
from __future__ import division
from __future__ import division
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from numpy import random
import matplotlib.pyplot as plt

"""*********************************************************************************************************
"""


class KirkMethod(object):

    def __init__(self, S1, S2, K, r, T, sigma1, sigma2, rho, cp):

        self.S1 = S1
        self.S2 = S2
        self.K = K
        self.r = r
        self.T = T / 252
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        self.cp = cp

        self.a = S2 + K
        self.b = S2 / (S2 + K)

        self.sigma = np.sqrt(sigma1 ** 2 - 2 * self.b * rho * sigma1 * sigma2 + self.b ** 2 * sigma2 ** 2)

        self.d1 = (np.log(S1 / self.a) + 0.5 * self.sigma ** 2 * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def OptionPrice(self):

        price = np.exp(-self.r * self.T) * (self.cp * self.S1 * norm.cdf(self.cp * self.d1) - self.cp * \
                                            (self.S2 + self.K) * norm.cdf(self.cp * self.d2))

        return price

    def OptionDelta(self):

        def IfElse(x):
            if x == 1:
                return 0
            else:
                return x

        beta = self.sigma2 * self.S2 / self.a
        sigma_pd_S2 = self.sigma2 * self.K / self.sigma * (beta - self.rho * self.sigma1) \
                      / self.a ** 2

        delta1 = np.exp(-self.r * self.T) * (norm.cdf(self.d1) + IfElse(self.cp))
        delta2 = np.exp(-self.r * self.T) * ((-norm.cdf(self.d2) + \
                                              self.a * norm.pdf(self.d2) * \
                                              np.sqrt(self.T) * sigma_pd_S2) \
                                             - IfElse(self.cp))

        return delta1, delta2

    def OptionDeltaPdRho(self):

        beta = self.sigma2 * self.S2 / self.a

        d1_pd_sigma = np.sqrt(self.T) - (np.log(self.S1 / self.a) + 0.5 * self.sigma ** 2 * self.T) / \
                      (self.sigma ** 2 * np.sqrt(self.T))

        d2_pd_sigma = d1_pd_sigma - np.sqrt(self.T)

        sigma_pd_rho = -self.b * self.sigma1 * self.sigma2 / self.sigma

        deltaPDrho1 = np.exp(-self.r * self.T) * norm.pdf(self.d1) * d1_pd_sigma * sigma_pd_rho

        deltaPDrho2 = np.exp(-self.r * self.T) * \
                      (-norm.pdf(self.d2) * d2_pd_sigma * sigma_pd_rho + \
                       self.sigma2 * self.K * np.sqrt(self.T) * norm.pdf(self.d2) / self.a * \
                       (((self.rho - self.sigma) * self.sigma1 - beta) / self.sigma ** 2 - \
                        self.d2 * d2_pd_sigma * sigma_pd_rho * (beta - self.rho * self.sigma1) / self.sigma) \
                       )

        return deltaPDrho1, deltaPDrho2

    def OptionVega(self):

        vega = self.S1 * np.exp(-self.r * self.T) * norm.pdf(self.d1) * np.sqrt(self.T)

        return vega

    def OptionGamma(self):
        beta = self.sigma2 * self.S2 / self.a
        sigma_pd_S2 = self.sigma2 * self.K / self.sigma * (beta - self.rho * self.sigma1) \
                      / self.a ** 2

        sigma_pd_S2_pd_S2 = self.sigma2 * self.K / (self.sigma ** 2 * self.a ** 3) * \
                            (self.sigma * self.sigma2 * self.K / self.a + \
                             (self.rho * self.sigma1 - beta) * (2 * self.sigma + \
                                                                self.a * sigma_pd_S2))

        d2_pd_S2 = -1 / (self.sigma) * sigma_pd_S2 * self.d1 - 1 / (self.sigma * \
                                                                    np.sqrt(self.T) * self.a)

        gamma1 = np.exp(-self.r * self.T) * norm.pdf(self.d1) / (self.S1 * self.sigma * np.sqrt(self.T))
        gamma2 = np.exp(-self.r * self.T) * norm.pdf(self.d2) * (-d2_pd_S2 + \
                                                                 np.sqrt(self.T) * (
                                                                             sigma_pd_S2 + self.a * (sigma_pd_S2_pd_S2 - \
                                                                                                     self.d2 * d2_pd_S2 * sigma_pd_S2)))

        return gamma1, gamma2

    def OptionChi(self):

        chi = np.exp(-self.r * self.T) * self.S1 / self.a * self.S2 * np.sqrt(self.T) * \
              norm.pdf(self.d1) * self.sigma1 * self.sigma2 / self.sigma

        return chi


def ImpliedVol(S1, S2, K, r, T, sigma1, rho, cp, price):
    def func(sigma2):
        PricingEngine = KirkMethod(S1, S2, K, r, T, sigma1, sigma2, rho, cp)
        return PricingEngine.OptionPrice() - price

    impliedSigma = brentq(func, 0.01, 0.99)
    return impliedSigma


class KirkMethodIV(object):
    def __init__(self):
        pass

    def combinedSigma(self, S2, K, sigma1, sigma2, rho):
        result = np.sqrt(sigma1 ** 2 - 2 * S2 / (S2 + K) * rho * sigma1 * sigma2 + (S2 / (S2 + K)) ** 2 * sigma2 ** 2)

        return result

    def d(self, S1, S2, K, r, T, sigma):
        d1 = (np.log(S1 / (S2 + K)) + 0.5 * sigma ** 2 * T / 252) / (sigma * np.sqrt(T / 252))
        d2 = d1 - sigma * np.sqrt(T / 252)
        return d1, d2

    def Price(self, S1, S2, K, r, T, sigma):
        d1, d2 = self.d(S1, S2, K, r, T, sigma)
        price = np.exp(-r * T / 252) * (S1 * norm.cdf(d1) - (S2 + K) * norm.cdf(d2))

        return price

    def Vega(self, S1, S2, K, r, T, sigma):

        d1, d2 = self.d(S1, S2, K, r, T, sigma)
        vega = S1 * np.exp(-r * T / 252) * norm.pdf(d1) * np.sqrt(T / 252)

        return vega

    def impliedVolatility(self, fPrice, fVega, price):

        TOLABS = 1e-6
        MAXITER = 100

        sigma = 0.3  # intial estimate
        dSigma = 10 * TOLABS  # enter loop for the first time
        nIter = 0
        while ((nIter < MAXITER) & (abs(dSigma) > TOLABS)):
            nIter = nIter + 1
            # Newton-Raphson method
            dSigma = (fPrice(sigma) - price) / fVega(sigma)
            sigma = sigma - dSigma

        if (nIter == MAXITER):
            warning('Newton-Raphson not converged')
        return sigma

    def impliedVol(self, S1, S2, K, r, T, price):
        def fPrice(sigma):
            return self.Price(S1, S2, K, r, T, sigma)

        def fVega(sigma):
            return self.Vega(S1, S2, K, r, T, sigma)

        impliedSigma = self.impliedVolatility(fPrice, fVega, price)

        return impliedSigma


"""*********************************************************************************************************
"""


class ThreeAssetsKirkMethod(object):

    def __init__(self, F1, F2, F3, K, r, T, sigma1, sigma2, sigma3, rho12, rho13, rho23, cp):
        self.F1 = F1
        self.F2 = F2
        self.F3 = F3
        self.K = K
        self.r = r
        self.T = T / 252
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.rho12 = rho12
        self.rho13 = rho13
        self.rho23 = rho23
        self.cp = cp

        self.a = F2 + F3 + K
        self.b2 = F2 / (F2 + F3 + K)
        self.b3 = F3 / (F2 + F3 + K)

        self.sigma = np.sqrt(sigma1 ** 2 + self.b2 ** 2 * sigma2 ** 2 + self.b3 ** 2 * sigma3 ** 2 - \
                             2 * self.b2 * sigma1 * sigma2 * rho12 - 2 * self.b3 * sigma1 * sigma3 * rho13 + \
                             2 * self.b2 * self.b3 * sigma2 * sigma3 * rho23)

        self.d1 = (np.log(F1 / self.a) + 0.5 * self.sigma ** 2 * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def OptionPrice(self):
        price = np.exp(self.r * self.T) * (self.F1 * norm.cdf(self.d1) - self.a * norm.cdf(self.d2))
        return price

    def OptionDelta(self):
        b2_pd_F2 = (self.F3 + self.K) / self.a ** 2
        b3_pd_F2 = -self.F3 / self.a ** 2

        b2_pd_F3 = -self.F2 / self.a ** 2
        b3_pd_F3 = (self.F2 + self.K) / self.a ** 2

        sigma_pd_F2 = 1 / (2 * self.sigma) * \
                      (b2_pd_F2 * \
                       (self.sigma2 ** 2 * self.b2 - self.sigma1 * self.sigma2 * self.rho12 + \
                        self.b3 * self.sigma2 * self.sigma3 * self.rho23) + \
                       b3_pd_F2 * \
                       (self.sigma3 ** 2 * self.b3 - self.sigma1 * self.sigma3 * self.rho13 + \
                        self.b2 * self.sigma2 * self.sigma3 * self.rho23))

        sigma_pd_F3 = 1 / (2 * self.sigma) * \
                      (b2_pd_F3 * \
                       (self.sigma2 ** 2 * self.b2 - self.sigma1 * self.sigma2 * self.rho12 + \
                        self.b3 * self.sigma2 * self.sigma3 * self.rho23) + \
                       b3_pd_F3 * \
                       (self.sigma3 ** 2 * self.b3 - self.sigma1 * self.sigma3 * self.rho13 + \
                        self.b2 * self.sigma2 * self.sigma3 * self.rho23))

        delta1 = np.exp(-self.r * self.T) * norm.cdf(self.d1)

        delta2 = np.exp(-self.r * self.T) * \
                 (-norm.cdf(self.d2) + self.a * norm.pdf(self.d2) * np.sqrt(self.T) * sigma_pd_F2)

        delta3 = np.exp(-self.r * self.T) * \
                 (-norm.cdf(self.d2) + self.a * norm.pdf(self.d2) * np.sqrt(self.T) * sigma_pd_F3)

        return delta1, delta2, delta3

    def OptionGamma(self):
        b2_pd_F2 = (self.F3 + self.K) / self.a ** 2
        b3_pd_F2 = -self.F3 / self.a ** 2

        b2_pd_F3 = -self.F2 / self.a ** 2
        b3_pd_F3 = (self.F2 + self.K) / self.a ** 2

        b2_pd_F2_pd_F2 = -2 * (self.F3 + self.K) / self.a ** 3
        b3_pd_F2_pd_F2 = 2 * self.F3 / self.a ** 3

        b2_pd_F3_pd_F3 = 2 * self.F2 / self.a ** 3
        b3_pd_F3_pd_F3 = -2 * (self.F2 + self.K) / self.a ** 3

        sigma_pd_F2 = 1 / (2 * self.sigma) * \
                      (b2_pd_F2 * \
                       (self.sigma2 ** 2 * self.b2 - self.sigma1 * self.sigma2 * self.rho12 + \
                        self.b3 * self.sigma2 * self.sigma3 * self.rho23) + \
                       b3_pd_F2 * \
                       (self.sigma3 ** 2 * self.b3 - self.sigma1 * self.sigma3 * self.rho13 + \
                        self.b2 * self.sigma2 * self.sigma3 * self.rho23))

        sigma_pd_F3 = 1 / (2 * self.sigma) * \
                      (b2_pd_F3 * \
                       (self.sigma2 ** 2 * self.b2 - self.sigma1 * self.sigma2 * self.rho12 + \
                        self.b3 * self.sigma2 * self.sigma3 * self.rho23) + \
                       b3_pd_F3 * \
                       (self.sigma3 ** 2 * self.b3 - self.sigma1 * self.sigma3 * self.rho13 + \
                        self.b2 * self.sigma2 * self.sigma3 * self.rho23))

        C1 = self.sigma2 ** 2 * self.b2 - self.sigma1 * self.sigma2 * self.rho12 + \
             self.b3 * self.sigma2 * self.sigma3 * self.rho23

        C2 = self.sigma2 ** 2 * b2_pd_F2 + self.sigma2 * self.sigma3 * self.rho23 * b3_pd_F2

        C3 = self.sigma3 ** 2 * self.b3 - self.sigma1 * self.sigma3 * self.rho13 + \
             self.b2 * self.sigma2 * self.sigma3 * self.rho23

        C4 = self.sigma3 ** 2 * b3_pd_F2 + self.sigma2 * self.sigma3 * self.rho23 * b2_pd_F2

        sigma_pd_F2_pd_F2 = -1 / self.sigma * sigma_pd_F2 ** 2 + 1 / self.sigma * \
                            (b2_pd_F2_pd_F2 * C1 + b2_pd_F2 * C2 + \
                             b3_pd_F2_pd_F2 * C3 + b3_pd_F2 * C4)

        C5 = self.sigma2 ** 2 * b2_pd_F3 + self.sigma2 * self.sigma3 * self.rho23 * b3_pd_F3

        C6 = self.sigma3 ** 2 * b3_pd_F3 + self.sigma2 * self.sigma3 * self.rho23 * b2_pd_F3

        sigma_pd_F3_pd_F3 = -1 / self.sigma * sigma_pd_F3 ** 2 + 1 / self.sigma * \
                            (b2_pd_F3_pd_F3 * C1 + b2_pd_F3 * C5 + \
                             b3_pd_F3_pd_F3 * C3 + b3_pd_F3 * C6)

        di_pd_F1 = 1 / (self.F1 * self.sigma * np.sqrt(self.T))

        #        d1_pd_F2 = -1 / self.sigma * sigma_pd_F2 * self.d2 - 1 / (self.sigma * np.sqrt(self.T) * self.a)

        d2_pd_F2 = -1 / self.sigma * sigma_pd_F2 * self.d1 - 1 / (self.sigma * np.sqrt(self.T) * self.a)

        #        d1_pd_F3 = -1 / self.sigma * sigma_pd_F3 * self.d2 - 1 / (self.sigma * np.sqrt(self.T) * self.a)

        d2_pd_F3 = -1 / self.sigma * sigma_pd_F3 * self.d1 - 1 / (self.sigma * np.sqrt(self.T) * self.a)

        gamma1 = np.exp(-self.r * self.T) * norm.pdf(self.d1) * di_pd_F1
        gamma2 = np.exp(-self.r * self.T) * norm.pdf(self.d2) * \
                 (-d2_pd_F2 + np.sqrt(self.T) * (sigma_pd_F2 + self.a * \
                                                 (sigma_pd_F2_pd_F2 - self.d2 * d2_pd_F2 * sigma_pd_F2)))

        gamma3 = np.exp(-self.r * self.T) * norm.pdf(self.d2) * \
                 (-d2_pd_F3 + np.sqrt(self.T) * (sigma_pd_F3 + self.a * \
                                                 (sigma_pd_F3_pd_F3 - self.d2 * d2_pd_F3 * sigma_pd_F3)))

        return gamma1, gamma2, gamma3


"""*********************************************************************************************************
"""


class BJSMethod(object):

    def __init__(self, S1, S2, K, r, T, sigma1, sigma2, rho):
        self.S1 = S1
        self.S2 = S2
        self.K = K
        self.r = r
        self.T = T / 252
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho

        self.a = S2 + K
        self.b = S2 / (S2 + K)
        self.sigma = np.sqrt(sigma1 ** 2 - 2 * self.b * rho * sigma1 * sigma2 + \
                             self.b ** 2 * sigma2 ** 2)

        self.d1 = (np.log(S1 / self.a) + 0.5 * self.sigma ** 2 * self.T) / \
                  (self.sigma * np.sqrt(self.T))

        self.d2 = (np.log(S1 / self.a) + (-0.5 * sigma1 ** 2 + rho * sigma1 * \
                                          sigma2 + 0.5 * self.b ** 2 * sigma2 ** 2 - self.b * sigma2 ** 2) * \
                   self.T) / (self.sigma * np.sqrt(self.T))

        self.d3 = (np.log(S1 / self.a) + (-0.5 * sigma1 ** 2 + 0.5 * self.b ** 2 * \
                                          sigma2 ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def OptionPrice(self):
        price = np.exp(-self.r * self.T) * (self.S1 * norm.cdf(self.d1) - \
                                            self.S2 * norm.cdf(self.d2) - \
                                            self.K * norm.cdf(self.d3))

        return price

    def OptionDelta(self):
        P = self.S1 * norm.pdf(self.d1) - self.S2 * norm.pdf(self.d2) - self.K * norm.pdf(self.d3)
        di_pd_S1 = 1 / (self.S1 * self.sigma * np.sqrt(self.T))

        delta1 = np.exp(-self.r * self.T) * (norm.cdf(self.d1) + P * di_pd_S1)

        d1_pd_S2 = 1 / (2 * np.sqrt(self.T) * self.sigma ** 3) * (self.sigma2 * \
                                                                  (self.rho * self.sigma1 - self.b * self.sigma2) * \
                                                                  self.K * self.b ** 2 / self.S2 ** 2 * (
                                                                              np.log(self.S1 / (self.S2 + self.K)) \
                                                                              - self.T * self.sigma ** 2) \
                                                                  - 2 * self.sigma ** 2 / \
                                                                  (self.S2 + self.K))

        d2_pd_S2 = 1 / (2 * np.sqrt(self.T) * self.sigma ** 3) * (self.sigma2 * \
                                                                  (self.rho * self.sigma1 - self.b * self.sigma2) * \
                                                                  self.K * self.b ** 2 / self.S2 ** 2 * (
                                                                              2 * np.log(self.S1 / (self.S2 + self.K)) \
                                                                              - self.T * self.sigma ** 2) \
                                                                  - 2 * self.sigma ** 2 / \
                                                                  (self.S2 + self.K))

        d3_pd_S2 = d2_pd_S2 - self.K * self.b ** 2 / self.S2 ** 2 * (1 - self.b) * \
                   (1 - self.rho) * self.sigma1 ** 2 * self.sigma2 ** 2 * \
                   np.sqrt(self.T) / self.sigma ** 3

        delta2 = np.exp(-self.r * self.T) * (-norm.cdf(self.d2) + ( \
                    self.S1 * norm.pdf(self.d1) * d1_pd_S2 - \
                    self.S2 * norm.pdf(self.d2) * d2_pd_S2 - \
                    self.K * norm.pdf(self.d3) * d3_pd_S2))

        return delta1, delta2

    def OptionGamma(self):
        di_pd_S1 = 1 / (self.S1 * self.sigma * np.sqrt(self.T))
        di_pd_S1_pd_S1 = -1 / (self.sigma * self.S1 ** 2 * np.sqrt(self.T))

        P1 = self.S1 * norm.pdf(self.d1) - self.S2 * norm.pdf(self.d2) - self.K * norm.pdf(self.d3)

        P2 = norm.pdf(self.d1) + (-self.S1 * self.d1 * norm.pdf(self.d1) + self.S2 * self.d2 * norm.pdf(self.d2) + \
                                  self.K * self.d3 * norm.pdf(self.d3)) * di_pd_S1

        gamma1 = np.exp(-self.r * self.T) * (norm.pdf(self.d1) * di_pd_S1 + di_pd_S1_pd_S1 * P1 + di_pd_S1 * P2)

        return gamma1


"""*********************************************************************************************************
"""


class ThreeAssetsBJSMethod(object):

    def __init__(self, F1, F2, F3, K, r, T, sigma1, sigma2, sigma3, rho12, rho13, rho23, cp):
        self.F1 = F1
        self.F2 = F2
        self.F3 = F3
        self.K = K
        self.r = r
        self.T = T / 252
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.rho12 = rho12
        self.rho13 = rho13
        self.rho23 = rho23
        self.cp = cp

        self.a = F2 + F3 + K
        self.b2 = F2 / (F2 + F3 + K)
        self.b3 = F3 / (F2 + F3 + K)

        self.sigma = np.sqrt(sigma1 ** 2 + self.b2 ** 2 * sigma2 ** 2 + self.b3 ** 2 * sigma3 ** 2 - \
                             2 * self.b2 * sigma1 * sigma2 * rho12 - 2 * self.b3 * sigma1 * sigma3 * rho13 + \
                             2 * self.b2 * self.b3 * sigma2 * sigma3 * rho23)

        self.d1 = (np.log(F1 / self.a) + (0.5 * sigma1 ** 2 + 0.5 * self.b2 ** 2 * sigma2 ** 2 + 0.5 * self.b3 ** 2 * \
                                          sigma3 ** 2 - self.b2 * sigma1 * sigma2 * rho12 - self.b3 * sigma1 * sigma3 * rho13 + \
                                          self.b2 * self.b3 * sigma2 * sigma3 * rho23) * self.T) / (
                              self.sigma * np.sqrt(self.T))

        self.d2 = (np.log(F1 / self.a) + (0.5 * self.b2 ** 2 * sigma2 ** 2 + 0.5 * self.b3 ** 2 * sigma3 ** 2 - \
                                          0.5 * sigma1 ** 2 - self.b2 * sigma2 ** 2 + self.b2 * self.b3 * sigma2 * sigma3 * rho23 - \
                                          self.b3 * sigma2 * sigma3 * rho23 + sigma1 * sigma2 * rho12) * self.T) / (
                              self.sigma * np.sqrt(self.T))

        self.d3 = (np.log(F1 / self.a) + (0.5 * self.b2 ** 2 * sigma2 ** 2 + 0.5 * self.b3 ** 2 * sigma3 ** 2 - \
                                          0.5 * sigma1 ** 2 - self.b3 * sigma3 ** 2 + self.b2 * self.b3 * sigma2 * sigma3 * rho23 - \
                                          self.b2 * sigma2 * sigma3 * rho23 + sigma1 * sigma3 * rho13) * self.T) / (
                              self.sigma * np.sqrt(self.T))

        self.d4 = (np.log(F1 / self.a) + (0.5 * self.b2 ** 2 * sigma2 ** 2 + 0.5 * self.b3 ** 2 * sigma3 ** 2 - \
                                          0.5 * sigma1 ** 2 + self.b2 * self.b3 * sigma2 * sigma3 * rho23) * self.T) / (
                              self.sigma * np.sqrt(self.T))

    def OptionPrice(self):
        price = np.exp(-self.r * self.T) * (self.F1 * norm.cdf(self.d1) - self.F2 * norm.cdf(self.d2) - \
                                            self.F3 * norm.cdf(self.d3) - self.K * norm.cdf(self.d4))

        return price


"""*********************************************************************************************************
"""


class DengLiMethod(object):

    def __init__(self, S1, S2, K, r, T, sigma1, sigma2, rho):
        self.S1 = S1
        self.S2 = S2
        self.K = K
        self.r = r
        self.T = T / 252
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho

        mu1 = np.log(S1) - 0.5 * sigma1 ** 2 * self.T
        mu2 = np.log(S2) - 0.5 * sigma2 ** 2 * self.T
        self.v1 = sigma1 * np.sqrt(self.T)
        self.v2 = sigma2 * np.sqrt(self.T)

        self.R = np.exp(mu2)
        self.epsilon = -1 / (2 * self.v1 * np.sqrt(1 - rho ** 2)) * self.v2 ** 2 * \
                       self.R * K / (self.R + K) ** 2

        self.C3 = 1 / (self.v1 * np.sqrt(1 - rho ** 2)) * (mu1 - np.log(self.R + K))

        self.D3 = 1 / (self.v1 * np.sqrt(1 - rho ** 2)) * (rho * self.v1 - self.v2 * self.R / (self.R + K))

        self.C2 = self.C3 + self.D3 * self.v2 + self.epsilon * self.v2 ** 2

        self.D2 = self.D3 + 2 * self.v2 * self.epsilon

        self.C1 = self.C3 + self.D3 * rho * self.v1 + self.epsilon * rho ** 2 * self.v1 ** 2 + np.sqrt(
            1 - rho ** 2) * self.v1

        self.D1 = self.D3 + 2 * rho * self.v1 * self.epsilon

        self.J0_1 = self.J0(self.C1, self.D1)
        self.J0_2 = self.J0(self.C2, self.D2)
        self.J0_3 = self.J0(self.C3, self.D3)

        self.J1_1 = self.J1(self.C1, self.D1)
        self.J1_2 = self.J1(self.C2, self.D2)
        self.J1_3 = self.J1(self.C3, self.D3)

        self.J2_1 = self.J2(self.C1, self.D1)
        self.J2_2 = self.J2(self.C2, self.D2)
        self.J2_3 = self.J2(self.C3, self.D3)

    def J0(self, u, v):
        result = norm.cdf(u / np.sqrt(1 + v ** 2))
        return result

    def J1(self, u, v):
        result = (1 + (1 + u ** 2) * v ** 2) / (1 + v ** 2) ** 2.5 * norm.pdf(u / np.sqrt(1 + v ** 2))
        return result

    def J2(self, u, v):
        result = ((6 - 6 * u ** 2) * v ** 2 + (21 - 2 * u ** 2 - u ** 4) * v ** 4 + 4 * (3 + u ** 2) * v ** 6 - 3) * \
                 u / (1 + v ** 2) ** 5.5 * norm.pdf(u / np.sqrt(1 + v ** 2))
        return result

    def OptionPrice(self):
        I1 = self.J0_1 + self.J1_1 * self.epsilon + 0.5 * self.J2_1 * self.epsilon ** 2
        I2 = self.J0_2 + self.J1_2 * self.epsilon + 0.5 * self.J2_2 * self.epsilon ** 2
        I3 = self.J0_3 + self.J1_3 * self.epsilon + 0.5 * self.J2_3 * self.epsilon ** 2

        price = np.exp(-self.r * self.T) * (self.S1 * I1 - self.S2 * I2 - self.K * I3)
        return price

    def OptionDelta(self):
        delta1 = np.exp(-self.r * self.T) * (self.J0_1 + self.J1_1 * self.epsilon + 0.5 * self.J2_1 * self.epsilon ** 2)
        delta2 = np.exp(-self.r * self.T) * (
                    -self.J0_2 - self.J1_2 * self.epsilon - 0.5 * self.J2_2 * self.epsilon ** 2)

        return delta1, delta2

    def Phi(self, x, y, u, v, w):
        result = norm.pdf(x / np.sqrt(1 + y ** 2)) * ((1 + y ** 2) ** 2 * u - x * (y + y ** 3) * v + \
                                                      (1 + (1 + x ** 2) * y ** 2) * w) / (1 + y ** 2) ** 2.5

        return result

    def OptionGamma(self):
        C_S1 = 1 / (self.v1 * self.S1 * np.sqrt(1 - self.rho ** 2))
        D_S1 = 0
        epsilon_S1 = 0

        C_S2 = self.epsilon / self.S2 * (2 * (self.R + self.K) / (self.v2 ** 2 * self.K) + 2 + \
                                         (1 - 2 * self.R / (self.R + self.K)) * self.v2 ** 2)

        D_S2 = self.epsilon / self.S2 * (2 / self.v2 + 2 * self.v2 * (1 - 2 * self.R / (self.R + self.K)))

        epsilon_S2 = self.epsilon / self.S2 * (1 - 2 * self.R / (self.R + self.K))

        gamma1 = self.Phi(self.C1, self.D1, C_S1, D_S1, epsilon_S1)
        gamma2 = -self.Phi(self.C2, self.D2, C_S2, D_S2, epsilon_S2)

        return gamma1, gamma2


"""*********************************************************************************************************
"""


class MCMethod(object):

    def __init__(self, S1, S2, K, r, T, sigma1, sigma2, rho, M):

        self.S1 = S1
        self.S2 = S2
        self.K = K
        self.r = r
        self.T = T / 252
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        self.M = M
        self.N = T

    def OptionPrice(self):

        S1_T, S2_T = MCsimulation(self.S1, self.S2, self.N, self.sigma1, self.sigma2, self.rho, self.M)
        spread = S1_T[:, -1] - S2_T[:, -1]
        payoff = np.where(spread < self.K, 0, spread - self.K)
        price = np.mean(payoff) * np.exp(-self.r * self.T)

        return price

    def NumericalDerivative(self, f, x0, h):

        if x0 == 0:
            derivative = (f(h) - f(-h)) / (2 * h)
        else:
            derivative = (f(x0 * (1 + h)) - f(x0 * (1 - h))) / (2 * x0 * h)
        return derivative

    def OptionDelta(self):
        h = 1e-5

        def fPrice1(S1):
            pricingMethod = MCMethod(S1, self.S2, self.K, self.r, self.N, \
                                     self.sigma1, self.sigma2, self.rho, self.M)

            return pricingMethod.OptionPrice()

        def fPrice2(S2):
            pricingMethod = MCMethod(self.S1, S2, self.K, self.r, self.N, \
                                     self.sigma1, self.sigma2, self.rho, self.M)

            return pricingMethod.OptionPrice()

        delta1 = self.NumericalDerivative(fPrice1, self.S1, h)
        delta2 = self.NumericalDerivative(fPrice2, self.S2, h)

        return delta1, delta2

    def OptionGamma(self):

        h = 1e-5

        def fDelta1(S1):
            pricingMethod = MCMethod(S1, self.S2, self.K, self.r, self.N, \
                                     self.sigma1, self.sigma2, self.rho, self.M)

            return pricingMethod.OptionDelta()[0]

        def fDelta2(S2):
            pricingMethod = MCMethod(self.S1, S2, self.K, self.r, self.N, \
                                     self.sigma1, self.sigma2, self.rho, self.M)

            return pricingMethod.OptionDelta()[1]

        gamma1 = self.NumericalDerivative(fDelta1, self.S1, h)
        gamma2 = self.NumericalDerivative(fDelta2, self.S2, h)

        return gamma1, gamma2


"""*******************************************************************************
"""


def MCsimulation(S1, S2, T, sigma1, sigma2, rho, M):
    # 股票实际价格模拟
    random.seed(seed=123)
    X = random.randn(M, T)
    Y = random.randn(M, T)

    deltaT = 1 / 252

    e1 = np.exp((-0.5 * sigma1 ** 2) * deltaT + sigma1 * np.sqrt(deltaT) * X)
    e2 = np.exp((-0.5 * sigma2 ** 2) * deltaT + sigma1 * rho * np.sqrt(deltaT) * X + \
                sigma2 * np.sqrt(1 - rho ** 2) * np.sqrt(deltaT) * Y)
    S1_T = np.cumprod(np.c_[S1 * np.ones((M, 1)), e1], axis=1)
    S2_T = np.cumprod(np.c_[S2 * np.ones((M, 1)), e2], axis=1)

    return S1_T, S2_T


def numRoundArr(num, tick):
    # 根据不同的tick调整价格 , num格式为narray
    row = num.shape[0]
    col = num.shape[1]
    roundNum = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if num[i, j] % tick > tick / 2:
                roundNum[i, j] = num[i, j] - num[i, j] % tick + tick
            else:
                roundNum[i, j] = num[i, j] - num[i, j] % tick

    return roundNum


def DeltaHedge(S1_0, S2_0, K, r, T, sigma1, sigma2, rho, M, tick1, tick2, m, n, commission, slippage, cp, bs):
    # m, n为构成价差各合约的数量
    S1, S2 = MCsimulation(S1_0, S2_0, T, sigma1, sigma2, rho, M)

    S1 = numRoundArr(S1, tick1) * m
    S2 = numRoundArr(S2, tick2) * n

    tick1 = m * tick1
    tick2 = n * tick2

    NominalPrincipal = 200000
    numOfOption = np.abs(NominalPrincipal / (S1_0 - S2_0))

    PricingMethod = KirkMethod(S1_0, S2_0, K, r, T, sigma1, sigma2, rho, cp)
    initialOptionPrice = PricingMethod.OptionPrice() * numOfOption

    numOfFuture1Arr = np.zeros((M, T + 1))
    numOfFuture2Arr = np.zeros((M, T + 1))

    priceIncre1Arr = np.zeros((M, T))
    priceIncre2Arr = np.zeros((M, T))

    PnLofFuture1Arr = np.zeros((M, T))
    PnLofFuture2Arr = np.zeros((M, T))
    lastOptionPrice = np.zeros((M, 1))

    for i in range(M):
        for j in range(T):
            PricingMethod = KirkMethod(S1[i, j], S2[i, j], K, r, T - j, sigma1, sigma2, rho, cp)
            delta1, delta2 = PricingMethod.OptionDelta()
            delta1 = delta1 * bs * cp * numOfOption
            delta2 = delta2 * bs * cp * numOfOption
            numOfFuture1Arr[i, j + 1] = -delta1
            numOfFuture2Arr[i, j + 1] = -delta2

    priceIncre1Arr = S1[:, 1:] - S1[:, :-1]
    priceIncre2Arr = S2[:, 1:] - S2[:, :-1]

    # **************************滑点费用及交易费用***********************************
    numOfFuture1Chg = numOfFuture1Arr[:, 1:] - numOfFuture1Arr[:, :-1]
    numOfFuture2Chg = numOfFuture2Arr[:, 1:] - numOfFuture2Arr[:, :-1]

    cost1Arr = S1[:, :-1] * np.abs(numOfFuture1Chg) * commission + \
               slippage * tick1 * np.abs(numOfFuture1Chg)

    cost2Arr = S2[:, :-1] * np.abs(numOfFuture2Chg) * commission + \
               slippage * tick2 * np.abs(numOfFuture2Chg)

    hedgeCost1 = np.mean(np.sum(cost1Arr, axis=1))
    hedgeCost2 = np.mean(np.sum(cost2Arr, axis=1))

    # ******************************************************************************

    PnLofFuture1Arr = priceIncre1Arr * numOfFuture1Arr[:, 1:]
    PnLofFuture2Arr = priceIncre2Arr * numOfFuture2Arr[:, 1:]
    PnLofFutureArr = PnLofFuture1Arr + PnLofFuture2Arr

    lastOptionPrice = np.maximum(cp * (S1[:, -1] - S2[:, -1] - K), 0) * bs * numOfOption

    PnLOfOption = np.mean(lastOptionPrice - initialOptionPrice)
    PnLOfFuture = np.mean(np.sum(PnLofFutureArr, axis=1)) - hedgeCost1 - hedgeCost2

    Spread = S1 - S2
    Smin = np.min(Spread[:, -1])
    Smax = np.max(Spread[:, -1])
    plt.plot(np.arange(Smin, Smax, 1), np.maximum(cp * (np.arange(Smin, Smax, 1) - K), 0) * bs)
    #    plt.plot(Spread[:,-1],np.sum(PnLofFutureArr, axis=1)/numOfOption,'or')
    plt.show()

    #    return (PnLOfOption+PnLOfFuture)/numOfOption, hedgeCost1/numOfOption, hedgeCost2/numOfOption
    return hedgeCost1 / numOfOption, hedgeCost2 / numOfOption



