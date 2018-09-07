# -*- coding: utf-8 -*-
# Leo70kg
from __future__ import division
import math
from WindPy import w
import pymysql
import datetime
import numpy as np
import pandas as pd
from DataStoreBase import DataStoreBase
import Util


class StockData(DataStoreBase):

    def connect2db(self):

        db = pymysql.connect(host='192.168.16.23', user='zuoyou', password='bhrsysp', db='stock_information',
                             charset='utf8')
        return db

    def create_new_table(self, db, table_name, condition=False):

        if condition:
            cursor = db.cursor()
            sql = """CREATE TABLE IF NOT EXISTS daily_info_{:s}(
                id int primary key auto_increment,
                trade_date date not null,
                close_price float,
                adj_factor FLOAT,
                trade_status varchar(10),
                max_up_down int,
                log_ret FLOAT)
                ENGINE=myisam
                """.format(table_name)

            cursor.execute(sql)

        else:
            pass

    def create_new_minute_table(self, db, table_name, condition=False):
        pass

    def drop_existed_daily_table(self, db, table_name, condition=False):

        if condition:
            cursor = db.cursor()

            sql = """truncate table daily_content_{:s}""".format(table_name)
            cursor.execute(sql)

        else:
            pass

    def drop_existed_minute_table(self, db, table_name, condition=False):
        pass

    def daily_data_handle(self, symbol, end_date, create_condition=False, drop_condition=False):

        db = self.connect2db()
        cursor = db.cursor()
        table_name = Util.code_split(symbol)[0]

        sql1 = """SELECT trade_date from daily_info_{:s}
                    where id=(select max(id) from daily_info_{:s})""".format(table_name, table_name)

        rowNum = cursor.execute(sql1)
        if rowNum == 0:

            ipoDate = w.wss(symbol, "ipo_date").Data[0][0]
            start_date = np.maximum(datetime.datetime(2016, 1, 1), ipoDate)
        else:
            lastDate = datetime2date(cursor.fetchone()[0])
            sql2 = """select trading_date from trading_date where id= 
                        (select id from trading_date where trading_date = '{:%Y-%m-%d}')+1""".format(lastDate)

            cursor.execute(sql2)
            start_date = cursor.fetchone()[0]
        #                start_date = w.tdaysoffset(1, lastDate, "").Data[0][0]

        stock = w.wsd(symbol, "close,adjfactor,trade_status,maxupordown", start_date, end_date)

        sql = "INSERT INTO daily_info_{:s} (trade_date, close_price, adj_factor, trade_status, max_up_down) VALUES (%s, %s, %s, %s, %s)".format(
            table_name)
        sql6 = "INSERT INTO daily_content_{:s} (trade_date, close_price, adj_factor, trade_status, max_up_down) VALUES (%s, %s, %s, %s, %s)".format(
            table_name)
        param = [(stock.Times[i].strftime('%Y-%m-%d'), stock.Data[0][i], stock.Data[1][i], \
                  stock.Data[2][i], stock.Data[3][i]) for i in xrange(len(stock.Data[0]))]

        cursor.executemany(sql, param)
        cursor.executemany(sql6, param)
        endtime = datetime.datetime.now()
        print (endtime - starttime).seconds

        print(self.getCurrentTime(), ": Downloading [", symbol,
              "] From " + start_date.strftime('%Y-%m-%d') + " to " + end_date.strftime('%Y-%m-%d'))
        print(self.getCurrentTime(), ": Download A Stock Has Finished .")
        db.commit()
        cursor.close()

    def minute_data_handle(self, symbol, end_date, create_condition=False, drop_condition=False):
        pass

