# -*- coding: utf-8 -*-
# Leo70kg
from __future__ import division
import FutureDailyDataStore
import FutureMinuteDataStore
from WindPy import w
import datetime


if __name__ == '__main__':
    w.start()
    end_date = datetime.date.today()
    symbols1 = ['AG.SHF', 'AL.SHF', "AP.CZC", "AU.SHF", 'BU.SHF', 'C.DCE', 'CF.CZC',
                'CS.DCE', 'CU.SHF', 'FG.CZC', 'HC.SHF', 'I.DCE', 'J.DCE', 'SM.CZC',
                'JD.DCE', 'JM.DCE', 'L.DCE', 'M.DCE', 'MA.CZC', 'NI.SHF', 'OI.CZC',
                'P.DCE', 'PB.SHF', 'PP.DCE', 'RB.SHF', 'RM.CZC', 'RU.SHF', 'SF.CZC',
                'SN.SHF', 'SR.CZC', 'TA.CZC', 'V.DCE', 'Y.DCE', 'ZC.CZC',
                'ZN.SHF', 'A.DCE', 'WH.CZC', 'SC.INE']

    a = FutureDailyDataStore.FutureDailyData()
    a.batch_data_handle(symbols1, end_date)

    symbols2 = ['AG.SHF', 'AL.SHF', "AP.CZC", "AU.SHF", 'BU.SHF', 'C.DCE', 'CF.CZC',
                'CS.DCE', 'CU.SHF', 'FG.CZC', 'HC.SHF', 'I.DCE', 'J.DCE', 'SM.CZC',
                'JD.DCE', 'JM.DCE', 'L.DCE', 'M.DCE', 'MA.CZC', 'NI.SHF', 'OI.CZC',
                'P.DCE', 'PB.SHF', 'PP.DCE', 'RB.SHF', 'RM.CZC', 'RU.SHF', 'SF.CZC',
                'SN.SHF', 'SR.CZC', 'TA.CZC', 'V.DCE', 'Y.DCE', 'ZC.CZC',
                'ZN.SHF', 'A.DCE', 'WH.CZC']

    for i in [1, 5, 15]:
        a = FutureMinuteDataStore.FutureMinuteData(i)
        a.batch_data_handle(symbols2, end_date)

