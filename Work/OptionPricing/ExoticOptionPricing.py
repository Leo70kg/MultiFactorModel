# -*- coding: utf-8 -*-
# Leo70kg
from __future__ import division
from __future__ import division
import numpy as np
from numpy import random
from scipy.stats import norm
import BsModel


def AsianOptionPricing(F, K, r, T, tau, sigma, cp):
    T = T / 365
    M = (2 * np.exp(sigma ** 2 * T) - 2 * np.exp(sigma ** 2 * tau) * (1 + sigma ** 2 * (T - tau))) / (
                sigma ** 4 * (T - tau) ** 2)
    sigmaA = np.sqrt(np.log(M) / T)
    d1 = (np.log(F / K) + T * sigmaA ** 2 / 2) / (sigmaA * np.sqrt(T))
    d2 = d1 - sigmaA * np.sqrt(T)

    price = np.exp(-r * T) * (cp * F * norm.cdf(cp * d1) - cp * K * norm.cdf(cp * d2))

    return price


"""***********************************cash-or-nothing binary option**************************************"""


def DigitOptionPrice(F, K, fixedCash, r, q, T, sigma, cp):
    T = T / 252
    d = (np.log(F / K) + (r - q - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    price = fixedCash * np.exp(-r * T) * norm.cdf(cp * d)

    return price


def DigitOptionDelta(F, K, fixedCash, r, q, T, sigma, cp):
    T = T / 252

    d = (np.log(F / K) + (r - q - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    delta = fixedCash * np.exp(-r * T) * cp * norm.pdf(cp * d) / (sigma * np.sqrt(T) * F)

    return delta


def DigitOptionGamma(F, K, fixedCash, r, q, T, sigma, cp):
    T = T / 252
    d1 = (np.log(F / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    gamma = fixedCash * -np.exp(-r * T) * d1 * cp * norm.pdf(d2) / (sigma ** 2 * F ** 2 * T)

    return gamma


def DigitOptionVega(F, K, fixedCash, r, q, T, sigma, cp):
    T = T / 252
    d = (np.log(F / K) + (r - q - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

    vega = fixedCash * -np.exp(-r * T) * cp * norm.pdf(d) * (d / sigma + np.sqrt(T))

    return vega


"""******************************************************************************************************"""

"""*******************************barrier option*********************************************************"""


def BarrierOptionPrice(S, K, H, r, q, T, sigma, cp, oi, ud, cont=0, X=0):
    """down: ud=1, up: ud=-1
       out: oi=1, in: oi=-1
       cont: if the price is continuously moving, cont=0; if the price is discrete, e.g. every day closing price,
               cont=1
       X is the cash rebate, which is paid out at option expiration if the option has not been knocked in
       during its lifetime"""

    T = T / 252
    beta = 0.5826
    H = H * np.exp(-ud * beta * sigma * np.sqrt(1 / 252) * cont)

    mu = (r - q - sigma ** 2 / 2) / sigma ** 2
    lamda = np.sqrt(mu ** 2 + 2 * r / sigma ** 2)

    x1 = np.log(S / K) / (sigma * T ** 0.5) + (1 + mu) * sigma * T ** 0.5

    x2 = np.log(S / H) / (sigma * T ** 0.5) + (1 + mu) * sigma * T ** 0.5

    y1 = np.log(H ** 2 / (S * K)) / (sigma * T ** 0.5) + (1 + mu) * sigma * T ** 0.5

    y2 = np.log(H / S) / (sigma * T ** 0.5) + (1 + mu) * sigma * T ** 0.5

    z = np.log(H / S) / (sigma * T ** 0.5) + lamda * sigma * T ** 0.5

    A = cp * S * np.exp(-q * T) * norm.cdf(cp * x1) - cp * K * np.exp(-r * T) * \
        norm.cdf(cp * x1 - cp * sigma * T ** 0.5)

    B = cp * S * np.exp(-q * T) * norm.cdf(cp * x2) - cp * K * np.exp(-r * T) * \
        norm.cdf(cp * x2 - cp * sigma * T ** 0.5)

    C = cp * S * np.exp(-q * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(ud * y1) - \
        cp * K * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(ud * y1 - ud * sigma * T ** 0.5)

    D = cp * S * np.exp(-q * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(ud * y2) - \
        cp * K * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(ud * y2 - ud * sigma * T ** 0.5)

    E = X * np.exp(-r * T) * (norm.cdf(ud * x2 - ud * sigma * T ** 0.5) - \
                              (H / S) ** (2 * mu) * norm.cdf(ud * y2 - ud * sigma * T ** 0.5))

    F = X * ((H / S) ** (mu + lamda) * norm.cdf(ud * z) + \
             (H / S) ** (mu - lamda) * norm.cdf(ud * z - 2 * ud * lamda * sigma * T ** 0.5))

    if ud == -1:

        if oi == 1:

            if (K < H) and (cp == 1):
                return A - B + C - D + F

            elif (K > H) and (cp == 1):
                return F

            elif (K < H) and (cp == -1):
                return A - C + F

            else:
                return B - D + F

        else:

            if (K < H) and (cp == 1):
                return B - C + D + E

            elif (K > H) and (cp == 1):
                return A + E

            elif (K < H) and (cp == -1):
                return C + E

            else:
                return A - B + D + E

    else:

        if oi == 1:

            if (K < H) and (cp == 1):
                return B - D + F

            elif (K > H) and (cp == 1):
                return A - C + F

            elif (K < H) and (cp == -1):
                return F

            else:
                return A - B + C - D + F

        else:

            if (K < H) and (cp == 1):
                return A - B + D + E

            elif (K > H) and (cp == 1):
                return C + E

            elif (K < H) and (cp == -1):
                return A + E

            else:
                return B - C + D + E


def numericalDerivative(f, x0, h):
    if x0 == 0:
        derivative = (f(h) - f(-h)) / (2 * h)
    else:
        derivative = (f(x0 * (1 + h)) - f(x0 * (1 - h))) / (2 * x0 * h)
    return derivative


def numericalSecondDerivative(f, x0, h):
    if x0 == 0:
        derivative = (f(h) - f(0) + f(-h)) / (2 * h ** 2)
    else:
        derivative = (f(x0 * (1 + h)) - 2 * f(x0) + f(x0 * (1 - h))) / (x0 * h) ** 2
    return derivative


def BarrierOptionDelta(S0, K, H, r, q, T, sigma, cp, oi, ud, X=0):
    h = 1e-5

    def fPrice(S):
        return BarrierOptionPrice(S, K, H, r, q, T, sigma, cp, oi, ud, X=0)

    delta = numericalDerivative(fPrice, S0, h)

    return delta


def BarrierOptionGamma(S0, K, H, r, q, T, sigma, cp, oi, ud, X=0):
    h = 1e-5

    def fDelta(S):
        return BarrierOptionDelta(S, K, H, r, q, T, sigma, cp, oi, ud, X=0)

    gamma = numericalDerivative(fDelta, S0, h)

    return gamma


def BarrierOptionVega(S, K, H, r, q, T, sigma0, cp, oi, ud, X=0):
    h = 1e-5

    def fPrice(sigma):
        return BarrierOptionDelta(S, K, H, r, q, T, sigma, cp, oi, ud, X=0)

    vega = numericalDerivative(fPrice, sigma0, h)

    return vega


def SimulationForBarrier(S0, K, H, r, q, T, sigma, cp, oi, ud, M, N):
    # Generate M x N samples from N(0,1)
    random.seed(seed=123)
    X = np.random.randn(M, N)

    # Simulate M trajectories in N steps
    deltaT = T / 252 / N
    e = np.exp((r - q - 0.5 * sigma ** 2) * deltaT + sigma * np.sqrt(deltaT) * X)
    St = np.cumprod(np.c_[S0 * np.ones((M, 1)), e], axis=1)

    S = St[:, :-1]
    l1, l2 = S.shape

    Delta = np.zeros(S.shape)
    Gamma = np.zeros(S.shape)
    Vega = np.zeros(S.shape)

    if ud == 1:
        if oi == -1:

            for i in range(l1):
                for j in range(l2):

                    if S[i, j] <= H:

                        break
                    else:

                        delta = BarrierOptionDelta(S[i, j], K, H, r, q, T - j, sigma, cp, oi, ud)
                        gamma = BarrierOptionGamma(S[i, j], K, H, r, q, T - j, sigma, cp, oi, ud)
                        vega = BarrierOptionVega(S[i, j], K, H, r, q, T - j, sigma, cp, oi, ud)

                    Delta[i, j] = delta
                    Gamma[i, j] = gamma
                    Vega[i, j] = vega

                for k in range(j, l2):
                    delta = BsModel.bsDelta(S[i, k], K, r, q, T - k, sigma, cp)
                    gamma = BsModel.bsGamma(S[i, k], K, r, q, T - k, sigma, cp)
                    vega = BsModel.bsVega(S[i, k], K, r, q, T - k, sigma, cp)

                    Delta[i, k] = delta
                    Gamma[i, k] = gamma
                    Vega[i, k] = vega


        else:
            for i in range(l1):
                for j in range(l2):

                    if S[i, j] <= H:

                        delta = 0
                        gamma = 0
                        vega = 0
                        break
                    else:

                        delta = BarrierOptionDelta(S[i, j], K, H, r, q, T - j, sigma, cp, oi, ud)
                        gamma = BarrierOptionGamma(S[i, j], K, H, r, q, T - j, sigma, cp, oi, ud)
                        vega = BarrierOptionVega(S[i, j], K, H, r, q, T - j, sigma, cp, oi, ud)

                    Delta[i, j] = delta
                    Gamma[i, j] = gamma
                    Vega[i, j] = vega

    else:
        if oi == -1:

            for i in range(l1):
                for j in range(l2):

                    if S[i, j] >= H:
                        break
                    else:

                        delta = BarrierOptionDelta(S[i, j], K, H, r, q, T - j, sigma, cp, oi, ud)
                        gamma = BarrierOptionGamma(S[i, j], K, H, r, q, T - j, sigma, cp, oi, ud)
                        vega = BarrierOptionVega(S[i, j], K, H, r, q, T - j, sigma, cp, oi, ud)

                    Delta[i, j] = delta
                    Gamma[i, j] = gamma
                    Vega[i, j] = vega

                for k in range(j, l2):
                    delta = BsModel.bsDelta(S[i, k], K, r, q, T - k, sigma, cp)
                    gamma = BsModel.bsGamma(S[i, k], K, r, q, T - k, sigma, cp)
                    vega = BsModel.bsVega(S[i, k], K, r, q, T - k, sigma, cp)

                    Delta[i, k] = delta
                    Gamma[i, k] = gamma
                    Vega[i, k] = vega

        else:

            for i in range(l1):
                for j in range(l2):

                    if S[i, j] >= H:

                        delta = 0
                        gamma = 0
                        vega = 0
                        break
                    else:

                        delta = BarrierOptionDelta(S[i, j], K, H, r, q, T - j, sigma, cp, oi, ud)
                        gamma = BarrierOptionGamma(S[i, j], K, H, r, q, T - j, sigma, cp, oi, ud)
                        vega = BarrierOptionVega(S[i, j], K, H, r, q, T - j, sigma, cp, oi, ud)

                    Delta[i, j] = delta
                    Gamma[i, j] = gamma
                    Vega[i, j] = vega

    return S, Delta, Gamma, Vega


"""******************************************************************************************************"""

"""************************************Lookback Option***************************************************"""


def FloatingStrikeLookbackOptionPrice(S, S_extreme, r, q, T, sigma, cp):
    """if option is call, cp=1, strike price is S_min, S_extreme=S_min
       if option is put, cp=-1, strike price is S_max, S_extreme=S_max"""

    b = r - q
    T = T / 252
    d1 = (np.log(S / S_extreme) + (b + sigma ** 2 / 2) * T) / (sigma * T ** 0.5)
    d2 = d1 - sigma * T ** 0.5

    if b == 0:
        price = cp * S * np.exp(-r * T) * norm.cdf(cp * d1) - cp * S_extreme * np.exp(-r * T) * norm.cdf(cp * d2) + \
                S * np.exp(-r * T) * sigma * T ** 0.5 * (norm.pdf(d1) + d1 * (-cp * norm.cdf(-cp * d1)))

    else:
        price = cp * S * np.exp((b - r) * T) * norm.cdf(cp * d1) - cp * S_extreme * np.exp(-r * T) * \
                norm.cdf(cp * d2) + S * np.exp(-r * T) * sigma ** 2 / (2 * b) * (cp * (S / S_extreme) ** \
                                                                                 (-2 * b / (sigma ** 2)) * norm.cdf(
                    -cp * (d1 - 2 * b / sigma * T ** 0.5)) - cp * np.exp(b * T) * \
                                                                                 norm.cdf(-cp * d1))

    return price


def FixedStrikeLookbackOptionPrice(S, S_extreme, X, r, q, T, sigma, cp):
    """if option is call, S_extreme = S_max
       if option is put, S_extreme = S_min """

    t = T / 252
    price = FloatingStrikeLookbackOptionPrice(S, S_extreme, r, q, T, sigma, -cp) + cp * S * np.exp(-q * t) - \
            cp * X * np.exp(-r * t)

    return price


"""******************************************************************************************************"""

"""**********************************************Asian Option********************************************"""


def TWDiscreteArithmeticAsianOptionPrice(S, S_A, X, r, q, T, sigma, cp, n, m, t1=0):
    """Turnbull-Wakeman formula
       t1为开始计算平均价格的起始点， 一般为0
       n为期权存续期内计算平均价格的天数
       当前时间不在计算平均时间期限内
       S_A为已实现的平均价格, n为总的平均次数，m为当前处在第几次的平均"""

    b = r - q
    t = T / 252
    t1 = t1 / 252
    h = (t - t1) / (n - 1)

    if b == 0:

        E_At = S
        E_At2 = S ** 2 * np.exp(sigma ** 2 * t1) / n ** 2 * (
                    (1 - np.exp(sigma ** 2 * h * n)) / (1 - np.exp(sigma ** 2 * h)) \
                    + 2 / (1 - np.exp(sigma ** 2 * h)) * (
                                n - (1 - np.exp(sigma ** 2 * h * n)) / (1 - np.exp(sigma ** 2 * h))))

    else:

        E_At = S / n * np.exp(b * t1) * (1 - np.exp(b * h * n)) / (1 - np.exp(b * h))
        E_At2 = S ** 2 * np.exp((2 * b + sigma ** 2) * t1) / n ** 2 * ((1 - np.exp((2 * b + sigma ** 2) * h * n)) / \
                                                                       (1 - np.exp((2 * b + sigma ** 2) * h)) + 2 / (
                                                                                   1 - np.exp((b + sigma ** 2) * h)) * \
                                                                       ((1 - np.exp(b * h * n)) / (
                                                                                   1 - np.exp(b * h)) - (1 - np.exp(
                                                                           (2 * b + sigma ** 2) * h * n)) / \
                                                                        (1 - np.exp((2 * b + sigma ** 2) * h))))

    sigmaA = np.sqrt((np.log(E_At2) - 2 * np.log(E_At)) / t)

    d1 = (np.log(E_At / X) + t * sigmaA ** 2 / 2) / (sigmaA * t ** 0.5)
    d2 = d1 - sigmaA * t ** 0.5

    if m > 0 and m < n - 1:

        if S_A > n / m * X:
            S_A_revised = S_A * m / n + E_At * (n - m) / n

            if cp == 1:
                price = np.exp(-r * t) * (S_A_revised - X)

            else:
                price = 0

        else:

            X_revised = n / (n - m) * X - m / (n - m) * S_A
            price = np.exp(-r * t) * (cp * E_At * norm.cdf(cp * d1) - cp * X_revised * norm.cdf(cp * d2)) * (n - m) / n

    elif m == n - 1:

        X_revised = n * X - (n - 1) * S_A
        price = BsModel.bsPrice(S, X_revised, r, q, t, sigma, cp, 'TD') / n

    else:

        price = np.exp(-r * t) * (cp * E_At * norm.cdf(cp * d1) - cp * X * norm.cdf(cp * d2)) * (n - m) / n

    return price


"""******************************************************************************************************"""


def CurranDiscreteArithmeticAsianOptionPrice(S, S_A, X, r, q, T, sigma, cp, n, m, t1=0):
    """Curran formula
       t1为开始计算平均价格的起始点， 一般为0
       n为期权存续期内计算平均价格的天数
       当前时间不在计算平均时间期限内
       S_A为已实现的平均价格, n为总的平均次数，m为当前处在第几次的平均"""

    b = r - q
    t = T / 252
    t1 = t1 / 252
    deltaT = (t - t1) / (n - 1)

    sum1 = 0
    sum2 = 0

    mu = np.log(S) + (b - sigma ** 2 / 2) * (t1 + (n - 1) * deltaT / 2)
    sigma_x = np.sqrt(sigma ** 2 * (t1 + deltaT * (n - 1) * (2 * n - 1) / (6 * n)))

    if b == 0:
        EA = S
    else:
        EA = S / n * np.exp(b * t1) * (1 - np.exp(b * deltaT * n)) / (1 - np.exp(b * deltaT))

    for i in range(1, n + 1, 1):
        ti = t1 + (i - 1) * deltaT
        mu_i = np.log(S) + (b - sigma ** 2 / 2) * ti
        sigma_i = np.sqrt(sigma ** 2 * (t1 + (i - 1) * deltaT))
        sigma_xi = sigma ** 2 * (t1 + deltaT * ((i - 1) - i * (i - 1) / (2 * n)))

        sum1 = sum1 + np.exp(mu_i + sigma_xi * (np.log(X) - mu) / sigma_x ** 2 + (sigma_i ** 2 - sigma_xi ** 2 / \
                                                                                  sigma_x ** 2) / 2)

    X_hat = 2 * X - 1 / n * sum1

    for j in range(1, n + 1, 1):
        ti = t1 + (j - 1) * deltaT
        mu_i = np.log(S) + (b - sigma ** 2 / 2) * ti
        sigma_i = np.sqrt(sigma ** 2 * (t1 + (j - 1) * deltaT))
        sigma_xi = sigma ** 2 * (t1 + deltaT * ((j - 1) - j * (j - 1) / (2 * n)))

        sum2 = sum2 + np.exp(mu_i + sigma_i ** 2 / 2) * norm.cdf((mu - np.log(X_hat)) / sigma_x + sigma_xi / sigma_x)

    price = np.exp(-r * t) * cp * (1 / n * sum2 - X * norm.cdf(cp * (mu - np.log(X_hat)) / sigma_x)) * (n - m) / n

    if m > 0 and m < n - 1:

        if S_A > n / m * X:

            if cp == 1:
                S_A_revised = S_A * m / n + EA * (n - m) / n
                price = np.exp(-r * t) * (S_A_revised - X)
            else:
                price = 0
        else:
            X_revised = n / (n - m) * X - m / (n - m) * S_A
            price = np.exp(-r * t) * cp * (1 / n * sum2 - X_revised * norm.cdf(cp * (mu - np.log(X_hat)) / \
                                                                               sigma_x)) * (n - m) / n

    elif m == n - 1:

        X_revised = n * X - (n - 1) * S_A
        price = BsModel.bsPrice(S, X_revised, r, q, t, sigma, cp, 'TD') / n

    else:

        price = np.exp(-r * t) * cp * (1 / n * sum2 - X * norm.cdf(cp * (mu - np.log(X_hat)) / sigma_x)) * (n - m) / n

    return price


"""******************************************************************************************************"""


def MCsimulation(S0, T, sigma, M, N, mu=0):
    # 股票实际价格模拟
    random.seed(seed=123)
    X = random.randn(M, N)
    deltaT = T / 252 / N
    e = np.exp((mu - 0.5 * sigma ** 2) * deltaT + sigma * np.sqrt(deltaT) * X)
    ST = np.cumprod(np.c_[S0 * np.ones((M, 1)), e], axis=1)
    return ST


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


def numRound(num, tick):
    if num % tick > tick / 2:
        roundNum = num - num % tick + tick
    else:
        roundNum = num - num % tick

    return roundNum


def priceOptionMC(S0, K, r, q, T, sigma, M, N, payoff_function):
    """
    priceOptionMC: Black-Scholes price of a generic option providing a payoff.
    INPUT:
        S0 : Initial value of the underlying asset
        r : Risk-free interest rate
        T : Time to expiry
    sigma : Volatility
        M : Number of simulations
        N : Number of observations
    payoff_function : payoff function of the option

    OUTPUT:
        price_MC : MC estimate of the price of the option in the Black-Scholes model

    """
    T = T / 252
    ## Generate M x N samples from N(0,1)
    random.seed(seed=123)
    X = np.random.randn(M, N)

    ## Simulate M trajectories in N steps
    deltaT = T / N
    e = np.exp((r - q - 0.5 * sigma ** 2) * deltaT + sigma * np.sqrt(deltaT) * X)
    S = np.cumprod(np.c_[S0 * np.ones((M, 1)), e], axis=1)

    ## Compute the payoff for each trajectory
    payoff = payoff_function(S)

    ## MC estimate of the price and the error of the option
    discountFactor = np.exp(-r * T)

    price_MC = discountFactor * np.mean(payoff)

    return price_MC


"""***************************************************************************"""


def MCPriceOfBarrier(S0, K, H, r, q, T, sigma, cp, oi, ud, M, N):
    """call: cp=1, put: cp=-1, in: oi=-1, out: oi=1, up: ud=-1, down: ud=1"""

    def CompareIfExceedThresh(S, H, oi, ud):
        if oi == 1 and ud == -1:
            return np.all(S < H, axis=1)

        elif oi == 1 and ud == 1:
            return np.all(-ud * S < -ud * H, axis=1)

        elif oi == -1 and ud == -1:
            return ~np.all(S < H, axis=1)

        else:
            return ~np.all(-ud * S < -ud * H, axis=1)

    def payoff(S):
        return np.where((S[:, -1] - K) * cp < 0, 0, cp * (S[:, -1] - K)) * CompareIfExceedThresh(S, H, oi, ud)

    return priceOptionMC(S0, K, r, q, T, sigma, M, N, payoff)


"""*******************************Greeks*************************************"""


def MCDeltaOfBarrier(S0, K, B, r, q, T, sigma, cp, oi, ud, M, N):
    h = 1e-3

    def fPrice(S):
        return MCPriceOfBarrier(S, K, B, r, q, T, sigma, cp, oi, ud, M, N)

    delta = numericalDerivative(fPrice, S0, h)

    return delta


"""**************************************************************************"""


def priceAsianArithmeticMeanCallMC(S0, K, r, T, sigma, M, N):
    def payoff(S):
        S_ar_mean = np.mean(S[:, 1:], 1)
        return np.where(S_ar_mean < K, 0, S_ar_mean - K)

    return priceOptionMC(S0, K, r, T, sigma, M, N, payoff)


def priceAsianArithmeticMeanPutMC(S0, K, r, T, sigma, M, N):
    def payoff(S):
        S_ar_mean = np.mean(S[:, 1:], 1)
        return np.where(S_ar_mean > K, 0, K - S_ar_mean)

    return priceOptionMC(S0, K, r, T, sigma, M, N, payoff)


def MCPriceOfEuropean(S0, K, r, q, T, sigma, cp, M, N):
    def payoff(S):
        return np.where(cp * (S[:, -1] - K) < 0, 0, cp * (S[:, -1] - K))

    return priceOptionMC(S0, K, r, q, T, sigma, M, N, payoff)


def DeltaHedgeFixedInterval(S0, K, r, T, sigma, M, N, Q, tick, commission, slippage, freq=1):
    # 固定时间对冲，频率单位为天, T单位为年，N为天数，即每天模拟出一个数据, N=T*365, Q为单位期权数量
    pricingEngine = MCpricing()
    greeks = Greeks()
    S = numRoundArr(pricingEngine.MCsimulation(S0, T, sigma, M, N), tick)

    M1 = 10000
    N1 = 500
    optionPrice0 = pricingEngine.priceEuropeanCallMC(S0, K, r, T, sigma, M1, N1) * Q
    delta0 = greeks.delta(S0, K, r, T, sigma, M1, N1) * Q
    cashAccount0 = optionPrice0 - np.round(delta0) * S0 * (1 + slippage) * (1 + commission)

    deltaArr = np.zeros((M, N + 1))
    optionPriceArr = np.zeros((M, N + 1))
    cashAccountArr = np.zeros((M, N + 1))

    deltaArr[:, 0] = delta0
    optionPriceArr[:, 0] = optionPrice0
    cashAccountArr[:, 0] = cashAccount0

    for i in range(M):
        for j in range(N):
            delta = greeks.delta(S[i, j + 1], K, r, T - (j + 1), sigma, M1, N1) * Q
            optionPrice = pricingEngine.priceEuropeanCallMC(S[i, j + 1], K, r, T - (j + 1), sigma, M1, N1) * Q
            deltaSign = np.sign(delta - deltaArr[i, j])
            if j == N - 1:
                cashAccount = cashAccountArr[i, j] + (optionPriceArr[i, j] - optionPrice) \
                              + np.round(deltaArr[i, j]) * S[i, j + 1] * (1 - slippage) * (1 - commission)
            else:
                cashAccount = cashAccountArr[i, j] + (optionPriceArr[i, j] - optionPrice) \
                              + np.round(deltaArr[i, j]) * (S[i, j + 1] - S[i, j]) \
                              + deltaSign * S[i, j + 1] * np.round(delta - deltaArr[i, j]) * (
                                          1 + deltaSign * slippage) * (1 + deltaSign * commission)

            deltaArr[i, j + 1] = delta
            optionPriceArr[i, j + 1] = optionPrice
            cashAccountArr[i, j + 1] = cashAccount

    return cashAccountArr


if __name__ == '__main__':

    sigma_1M_CF = np.arange(0.135, 0.2, 0.01)
    sigma_1M_AP = np.arange(0.29, 0.4, 0.01)
    sigma_1M_SR = np.arange(0.14, 0.24, 0.01)

    priceLis_1M_CF = []
    priceLis_1M_AP = []
    priceLis_1M_SR = []

    for sigma in sigma_1M_CF:
        price = TWDiscreteArithmeticAsianOptionPrice(16780, 16780, 16780, 0.06, 0.06, 22, sigma, 1, 22, 0)
        priceLis_1M_CF.append(price)

    for sigma in sigma_1M_AP:
        price = TWDiscreteArithmeticAsianOptionPrice(10343, 10343, 10343, 0.06, 0.06, 22, sigma, 1, 22, 0)
        priceLis_1M_AP.append(price)

    for sigma in sigma_1M_SR:
        price = TWDiscreteArithmeticAsianOptionPrice(4882, 4882, 4882, 0.06, 0.06, 22, sigma, 1, 22, 0)
        priceLis_1M_SR.append(price)

    sigma_6W_CF = np.arange(0.21, 0.3, 0.01)
    sigma_6W_AP = np.arange(0.39, 0.49, 0.01)
    sigma_6W_SR = np.arange(0.14, 0.24, 0.01)

    priceLis_6W_CF = []
    priceLis_6W_AP = []
    priceLis_6W_SR = []

    for sigma in sigma_6W_CF:
        price = TWDiscreteArithmeticAsianOptionPrice(16780, 16780, 16780, 0.06, 0.06, 33, sigma, 1, 33, 0)
        priceLis_6W_CF.append(price)

    for sigma in sigma_6W_AP:
        price = TWDiscreteArithmeticAsianOptionPrice(10343, 10343, 10343, 0.06, 0.06, 33, sigma, 1, 33, 0)
        priceLis_6W_AP.append(price)

    for sigma in sigma_6W_SR:
        price = TWDiscreteArithmeticAsianOptionPrice(4882, 4882, 4882, 0.06, 0.06, 33, sigma, 1, 33, 0)
        priceLis_6W_SR.append(price)

