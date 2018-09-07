# -*- coding: utf-8 -*-
# Leo70kg
from __future__ import division
import datetime
import json
import time
import math
import pandas as pd


def datetime2date(d):
    return datetime.date(d.year, d.month, d.day)


def load_future_trade_time():
    with open(r"C:\Users\BHRS-ZY-PC\PycharmProjects\Work\Database\future_trade_time.json", "r") as load_f:
        load_dict = json.loads(load_f.read())

    return load_dict


def load_future_code():
    """获取期货合约的代码和中文名称"""
    with open(r"C:\Users\BHRS-ZY-PC\PycharmProjects\Work\Database\future_code_name.json", "r") as load_f:
        load_dict = json.loads(load_f.read())

    return load_dict


def if_insert_date(db, table_name, data):
    """因为有时候收盘取数据的时候，当天数据还没有写入，取下来的当天收盘价其实是前一天的收盘价，所以这里添加一个判断函数，
       如果发现当天收盘价等于前一天收盘价，且其他字段例如最高价、成交量等为NAN，则停止写入数据库
    """
    cursor = db.cursor()

    sql = """SELECT CLOSE from {:s}_DAILY
            where id=(select max(id) from {:s}_DAILY)""".format(table_name, table_name)

    cursor.execute(sql)
    last_close = cursor.fetchone()[0]
    today_close = data.Data[4][0]

    def judge(logic_lis):
        for k in range(len(logic_lis)):
            if logic_lis[k]:
                return True
                break

        return False

    if last_close == today_close and judge([math.isnan(data.Data[i][0]) for i in xrange(len(data.Data) - 1)]):
        raise Exception("Data downloaded error")
    else:
        pass


def find_last_bd(cursor, date):
    """查找上一个交易日日期"""

    # cursor = db.cursor()
    sql = """select TRADE_DATE from trade_date where id= 
            (select id from trade_date where TRADE_DATE = '{:%Y-%m-%d}')-1""".format(date)
    cursor.execute(sql)

    return cursor.fetchone()[0]


def get_daily_ret(cursor, code, start_date, end_date):

    table_name = get_code_split(code)[0]

    sql = """select TRADE_DATE, LOG_RETURN from {:s}_daily where TRADE_DATE between 
            '{:%Y-%m-%d}' and '{:%Y-%m-%d}'""".format(table_name, start_date, end_date)

    cursor.execute(sql)
    ret = cursor.fetchall()

    date_time_lis = [ret[i][0] for i in range(len(ret))]
    ret_lis = [ret[i][1] for i in range(len(ret))]

    return pd.Series(ret_lis, index=date_time_lis).dropna()


def get_close_price(cursor, code, date):
    """获取收盘价"""
    table_name = get_code_split(code)[0]

    sql = """select CLOSE from {:s}_daily where TRADE_DATE = '{:%Y-%m-%d}'""".format(table_name, date)

    cursor.execute(sql)
    result = cursor.fetchone()[0]

    return result


def get_min_ret(cursor, code, freq, start_date, end_date):
    """获取期货合约分钟收益率，目前支持1分钟、5分钟和15分钟, 若合约有夜盘，从start_date前一天晚上20:59:00开始获取数据
    """

    name_lis = get_code_split(code)
    table_name = name_lis[0]
    exchange_code = name_lis[1]

    trade_time_dict = load_future_trade_time()

    if trade_time_dict[exchange_code][table_name][2] == 3:

        start_time = 32340
        end_time = 54000
        sql = """select TRADE_TIME, LOG_RETURN from {:s}_{:d}minute 
                where TIME_STAMP >= {:d} and TIME_STAMP <= {:d} and 
                TRADE_TIME between '{:%Y-%m-%d} 08:59:00' and 
                '{:%Y-%m-%d} 15:00:00'""".format(table_name, freq, start_time, end_time, start_date, end_date)

    elif trade_time_dict[exchange_code][table_name][2] == 4:

        start_time = 32340
        end_time = trade_time_dict[exchange_code][table_name][1][0]
        sql = """select TRADE_TIME, LOG_RETURN from {:s}_{:d}minute 
                where TIME_STAMP >= {:d} and TIME_STAMP <= {:d} and 
                TRADE_TIME between '{:%Y-%m-%d} 20:59:00' and 
                '{:%Y-%m-%d} 15:00:00'""".format(table_name, freq, start_time, end_time, find_last_bd(cursor,
                                                                                                      start_date),
                                                 end_date)

    else:
        start_time = trade_time_dict[exchange_code][table_name][1][1]
        end_time = 32340    # 对应早晨8:59:00，因为1分钟数据早盘从8:59:00开始有数据

        sql = """select TRADE_TIME, LOG_RETURN from {:s}_{:d}minute where id not in (select id from 
                {:s}_{:d}minute where TIME_STAMP > {:d} and TIME_STAMP < {:d}) and 
                TRADE_TIME between '{:%Y-%m-%d} 20:59:00' and '{:%Y-%m-%d} 15:00:00' """.format(table_name, freq,
                                                                                                table_name, freq,
                                                                                                start_time, end_time,
                                                                                                find_last_bd(cursor,
                                                                                                             start_date)
                                                                                                , end_date)

    cursor.execute(sql)
    ret = cursor.fetchall()

    date_time_lis = [ret[i][0] for i in range(len(ret))]
    ret_lis = [ret[i][1] for i in range(len(ret))]

    return pd.Series(ret_lis, index=date_time_lis).dropna()


def get_tick_size(cursor, code):

    sql = """select TICK_SIZE from contract_info where CODE = '{:s}'""".format(code)

    cursor.execute(sql)
    result = cursor.fetchone()[0]

    return result


def get_code_split(code):
    """将wind代码分开，e.g. ‘cu.shf’ => ['cu', 'shf']"""

    split = code.split('.')
    return split[0], split[1]


def deco1(func):
    """用于计算函数func运行时间"""
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        func(*args, **kwargs)
        end_time = datetime.datetime.now()
        secs = (end_time - start_time).seconds
        print('use %d secs' % secs)
    return wrapper


def deco2(func):
    """用于程序运行之前输出指示信息"""
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        display = time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time()))

        print(display, ": Downloading [", args[1], "] to " + args[2].strftime('%Y-%m-%d'))

    return wrapper

