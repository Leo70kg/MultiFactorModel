# _*_ coding:utf-8 _*_
# Leo70kg
from __future__ import division
from abc import ABCMeta, abstractmethod
import six


class QuoteBase(six.with_metaclass(ABCMeta, object)):
    """报价抽象基类"""

    @abstractmethod
    def quote(self, *args, **kwargs):
        """输出报价dataframe"""
        pass

    def df2excel(self, *args, **kwargs):
        """将报价dataframe写入excel"""
        pass

