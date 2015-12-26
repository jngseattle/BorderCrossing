import unittest2 as unittest
from BorderModel import BorderData, clean_df_subset
from dbhelper import pd_query
import copy
from random import randint


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

        self.feature = ['year', 'month', 'dayofmonth', 'week', 'dayofweek', 'minofday']
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
