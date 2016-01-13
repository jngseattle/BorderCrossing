import unittest2 as unittest
import logging
from predict import get_prediction, get_baseline
import datetime as dt
import pandas as pd


class TestData(unittest.TestCase):
    def test_get_prediction(self):
        '''
        Verify that minimum number of points have been predicted on each day
        '''
        minimum_points = 8
        begin = dt.date(2014, 1, 1)
        end = dt.date(2017, 1, 1)

        lane = 'Car'
        errorcount = 0

        for location in ['Peace Arch', 'Pacific Highway']:
            for direction in ['Southbound', 'Northbound']:
                start = begin
                while start <= end:
                    df = get_prediction(start, location, direction, lane)
                    if pd.isnull(df.waittime).sum() > minimum_points:
                        print '{0},{1},{2}'.format(start, location, direction)

                        errorcount += 1

                    start += dt.timedelta(1)

        self.assertEqual(errorcount, 0)

    def test_get_baseline(self):
        '''
        Verify that minimum number of points have been predicted on each day
        '''
        minimum_points = 8
        begin = dt.date(2014, 1, 1)
        end = dt.date(2017, 1, 1)

        lane = 'Car'
        errorcount = 0

        for location in ['Peace Arch', 'Pacific Highway']:
            for direction in ['Southbound', 'Northbound']:
                start = begin
                while start <= end:
                    df = get_baseline(start, location, direction, lane)
                    if pd.isnull(df.waittime).sum() > minimum_points:
                        print '{0},{1},{2}'.format(start, location, direction)

                        errorcount += 1

                    start += dt.timedelta(1)

        self.assertEqual(errorcount, 0)

if __name__ == '__main__':
    unittest.main()
