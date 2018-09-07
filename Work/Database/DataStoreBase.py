# -*- coding: utf-8 -*-
# Leo70kg
from __future__ import division
from abc import ABCMeta, abstractmethod
import six


class DataStoreBase(six.with_metaclass(ABCMeta, object)):
    """存储数据抽象基类"""

    @abstractmethod
    def connect2db(self, *args, **kwargs):
        """连接数据库"""
        pass

    @abstractmethod
    def data_handle(self, *args, **kwargs):
        """
        将数据写入数据库
        """
        pass

    @abstractmethod
    def create_new_table(self, *args, **kwargs):
        """写入数据前决定是否删除已有的数据库"""

        pass

    @abstractmethod
    def drop_existed_table(self, *args, **kwargs):
        """写入数据前决定是否删除已有的数据库"""

        pass

    @abstractmethod
    def find_start_date(self, *args, **kwargs):
        """写入数据前决定是否创建新的数据库"""

        pass

