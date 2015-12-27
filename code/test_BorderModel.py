import unittest2 as unittest
from BorderModel import BorderData, clean_df_subset
from BorderModel import BorderImpute, xy_laglead
from BorderModel import create_neighbor_features, create_leadlag
from dbhelper import pd_query
import copy
from random import randint
import numpy as np
import pandas as pd
import pdb


class TestBorderImpute(unittest.TestCase):
    def setUp(self):
        pass

    def test_create_neighbor_features(self):
        df0 = pd.DataFrame(np.array([3, 6, 9, 6, 2, 3, 5, 7, 9, 0]),
                           columns=('data',))
        df1 = create_leadlag(df0, feature='data')
        df2 = create_neighbor_features(df1, feature='data')

        leadres = np.array([6.5, 6.1, 4.1, 3.4, 5, 5.9, 6.11, 5.14, 0, np.nan])
        lagres = np.array([np.nan, 3., 4.71, 6.67, 6.6, 5., 3.9, 3.9, 5.1, 7.])

        self.assertTrue(np.allclose(df2.lead, leadres, rtol=0.01,
                        equal_nan=True))
        self.assertTrue(np.allclose(df2.lag, lagres, rtol=0.01,
                        equal_nan=True))

    def test_create_lead_lag(self):
        # Prepare test data
        X0 = np.random.randint(low=0, high=10, size=(10, 2))
        y0 = np.random.randint(low=0, high=100, size=(10))
        df0 = pd.DataFrame(np.hstack((y0.reshape(len(y0), 1), X0)),
                           columns=('y', 'waittime', 'volume'))

        # Test
        df = create_leadlag(df0)
        # pdb.set_trace()
        self.assertTrue((pd.isnull(df0.waittime.shift(-1)) |
                        (df.waittime_lead_1 == df0.waittime.shift(-1))).all())
        self.assertTrue((pd.isnull(df0.waittime.shift(-2)) |
                        (df.waittime_lead_2 == df0.waittime.shift(-2))).all())
        self.assertTrue((pd.isnull(df0.waittime.shift(1)) |
                        (df.waittime_lag_1 == df0.waittime.shift(1))).all())
        self.assertTrue((pd.isnull(df0.waittime.shift(2)) |
                        (df.waittime_lag_2 == df0.waittime.shift(2))).all())

    def test_xy_laglead(self):
        # Prepare test data
        X0 = np.random.randint(low=0, high=10, size=(10, 5))
        y0 = np.random.randint(low=0, high=100, size=(10))
        df0 = pd.DataFrame(np.hstack((y0.reshape(len(y0), 1), X0)),
                           columns=('y', 'data', 'lead2', 'lead1',
                                    'lag1', 'lag2'))
        # Simulate NaN values
        df0.loc[0, 'lead2'] = None
        df0.loc[1, 'lag2'] = None

        # Test lead only case
        X, y = xy_laglead(df0, ['lead1', 'lead2'], ['lag1', 'lag2'],
                          label='y', lead=True)
        self.assertTrue(np.array_equal(y.values, np.delete(y0, 0)))
        self.assertTrue(np.array_equal(X.values,
                                       np.delete(np.delete(X0, 0, 0),
                                                 [3, 4], 1)))

        # Test lag only case
        X, y = xy_laglead(df0, ['lead1', 'lead2'], ['lag1', 'lag2'],
                          label='y', lag=True)
        self.assertTrue(np.array_equal(y.values, np.delete(y0, 1)))
        self.assertTrue(np.array_equal(X.values,
                                       np.delete(np.delete(X0, 1, 0),
                                                 [1, 2], 1)))

        # Test lead/lag case
        X, y = xy_laglead(df0, ['lead1', 'lead2'], ['lag1', 'lag2'],
                          label='y', lead=True, lag=True)
        self.assertTrue(np.array_equal(y.values, np.delete(y0, [0, 1])))
        self.assertTrue(np.array_equal(X.values, np.delete(X0, [0, 1], 0)))


class TestBorderData(unittest.TestCase):
    def setUp(self):
        query = '''
                select
                    c.date,
                    waittime,
                    year,
                    month,
                    dayofmonth,
                    week,
                    dayofweek,
                    minofday
                from crossingdata c
                join datefeatures d on c.date = d.date
                where
                    valid=1
                    and waittime is not null
                    and crossing_id = 1
                    and dayofmonth = 5
                    and time = '12:00:00'
                    order by c.date
                '''

        self.df = pd_query(query)
        self.feature = ['year', 'month', 'dayofmonth', 'week', 'dayofweek',
                        'minofday']
        self.label = 'waittime'

    def test_normal(self):
        '''
        Normal use case - use raw data from query
        '''
        data = BorderData(self.df)

        # Test self.feature
        self.assertEqual(len(self.feature), len(data.X.columns.values))
        for f in self.feature:
            self.assertIn(f, data.X.columns.values)
        # Test y self.label
        for tries in range(20):
            i = randint(0, len(self.df) - 1)
            self.assertEqual(self.df[self.label].values[i], data.y.values[i])

    def test_subset(self):
        '''
        Subset use case - identify self.feature to keep
        '''
        subset = ['dayofweek', 'minofday']
        dfnew = clean_df_subset(self.df, subset)
        data = BorderData(dfnew)

        # Test self.feature
        self.assertEqual(len(subset), len(data.X.columns.values))
        for f in subset:
            self.assertIn(f, data.X.columns.values)
        # Test y self.label
        for tries in range(20):
            i = randint(0, len(self.df) - 1)
            self.assertEqual(self.df[self.label].values[i], data.y.values[i])

    def test_new_feature(self):
        '''
        New self.feature use case - keep it simple
                                just add to dataframe beforehand
        '''
        dfnew = self.df.copy()
        dfnew['ym'] = self.df.year * self.df.month
        data = BorderData(dfnew)

        # Test self.feature
        superset = copy.copy(self.feature)
        superset.append('ym')
        self.assertEqual(len(superset), len(data.X.columns.values))
        for f in superset:
            self.assertIn(f, data.X.columns.values)
        # Test y self.label
        for tries in range(20):
            i = randint(0, len(self.df) - 1)
            self.assertEqual(self.df[self.label].values[i], data.y.values[i])

    def test_new_label(self):
        '''
        New self.label use case - create target data, e.g. via smoothing,
                             or using volume self.label instead of waittime
        '''
        dfnew = self.df.copy()
        dfnew['wt2'] = self.df.waittime * 2
        data = BorderData(dfnew, label='wt2')

        # Test self.feature
        self.assertEqual(len(self.feature), len(data.X.columns.values))
        for f in self.feature:
            self.assertIn(f, data.X.columns.values)
        # Test y self.label
        for tries in range(20):
            i = randint(0, len(self.df) - 1)
            self.assertEqual(dfnew['wt2'].values[i], data.y.values[i])

    def test_subset_new_label(self):
        '''
        Subset use case - identify self.feature to keep
        '''
        dfnew = self.df.copy()
        dfnew['wt2'] = self.df.waittime * 2
        subset = ['dayofweek', 'minofday']
        dfnew = clean_df_subset(dfnew, subset, label='wt2')
        data = BorderData(dfnew, label='wt2')

        # Test self.feature
        self.assertEqual(len(subset), len(data.X.columns.values))
        for f in subset:
            self.assertIn(f, data.X.columns.values)
        # Test y self.label
        for tries in range(20):
            i = randint(0, len(self.df) - 1)
            self.assertEqual(dfnew['wt2'].values[i], data.y.values[i])

    def test_cvfolds(self):
        data = BorderData(self.df, 2)

        self.assertEqual(max(data.cv[-1, -2]),
                         data.df[data.df.year == 2014].index.max())
        self.assertEqual(min(data.cv[-1, -1]),
                         data.df[data.df.year == 2015].index.min())
        self.assertEqual(max(data.cv[-1, -1]),
                         data.df[data.df.year == 2015].index.max())
        self.assertEqual(len(data.cv[-1, -1]),
                         data.df[data.df.year == 2015].waittime.count())

        data = BorderData(self.df, 3)

        self.assertEqual(max(data.cv[-1, -2]),
                         data.df[data.df.year == 2014].index.max())
        self.assertEqual(min(data.cv[-1, -1]),
                         data.df[data.df.year == 2015].index.min())
        self.assertEqual(max(data.cv[-1, -1]),
                         data.df[data.df.year == 2015].index.max())
        self.assertEqual(len(data.cv[-1, -1]),
                         data.df[data.df.year == 2015].waittime.count())

if __name__ == '__main__':
    unittest.main()
