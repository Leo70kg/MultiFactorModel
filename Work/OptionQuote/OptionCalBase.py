# -*- coding: utf-8 -*-
# Leo70kg
from __future__ import division
from abc import ABCMeta, abstractmethod
import six


class OptionCalBase(six.with_metaclass(ABCMeta, object)):
    """报价抽象基类"""

    @abstractmethod
    def connect2db(self, *args, **kwargs):
        """连接数据库"""
        pass

    @abstractmethod
    def base_vol(self, *args, **kwargs):
        """计算基准波动率"""
        pass

    @abstractmethod
    def hedge_vol(self, *args, **kwargs):
        """计算对冲成本波动率"""
        pass

    @abstractmethod
    def vol_batch(self, *args, **kwargs):
        """写入excel中的所需波动率"""
        pass

