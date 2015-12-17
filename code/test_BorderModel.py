import unittest
from BorderModel import BorderData, clean_df_subset
from modelhelper import cvfolds
from dbhelper import pd_query
import copy
from random import randint

class TestBorderData(unittest.TestCase):
    def test_normal(self):
        '''
        Normal use case - use raw data from query
        '''
        data = BorderData(df, cv)

        # Test features
        self.assertEqual(len(features), len(data.X.columns.values))
        for f in features:
            self.assertIn(f, data.X.columns.values)
        # Test y label
        for tries in range(20):
            i = randint(0, len(df) - 1)
            self.assertEqual(df[label].values[i], data.y.values[i])

    def test_subset(self):
        '''
        Subset use case - identify features to keep
        '''
        subset = ['dayofweek', 'minofday']
        dfnew = clean_df_subset(df, subset)
        data = BorderData(dfnew, cv)

        # Test features
        self.assertEqual(len(subset), len(data.X.columns.values))
        for f in subset:
            self.assertIn(f, data.X.columns.values)
        # Test y label
        for tries in range(20):
            i = randint(0, len(df) - 1)
            self.assertEqual(df[label].values[i], data.y.values[i])

    def test_new_features(self):
        '''
        New features use case - keep it simple
                                just add to dataframe beforehand
        '''
        dfnew = df.copy()
        dfnew['ym'] = df.year * df.month
        data = BorderData(dfnew, cv)

        # Test features
        superset = copy.copy(features)
        superset.append('ym')
        self.assertEqual(len(superset), len(data.X.columns.values))
        for f in superset:
            self.assertIn(f, data.X.columns.values)
        # Test y label
        for tries in range(20):
            i = randint(0, len(df) - 1)
            self.assertEqual(df[label].values[i], data.y.values[i])

    def test_new_label(self):
        '''
        New label use case - create target data, e.g. via smoothing,
                             or using volume label instead of waittime
        '''
        dfnew = df.copy()
        dfnew['wt2'] = df.waittime * 2
        data = BorderData(dfnew, cv, label='wt2')

        # Test features
        self.assertEqual(len(features), len(data.X.columns.values))
        for f in features:
            self.assertIn(f, data.X.columns.values)
        # Test y label
        for tries in range(20):
            i = randint(0, len(df) - 1)
            self.assertEqual(dfnew['wt2'].values[i], data.y.values[i])

    def test_subset_new_label(self):
        '''
        Subset use case - identify features to keep
        '''
        dfnew = df.copy()
        dfnew['wt2'] = df.waittime * 2
        subset = ['dayofweek', 'minofday']
        dfnew = clean_df_subset(dfnew, subset, label='wt2')
        data = BorderData(dfnew, cv, label='wt2')

        # Test features
        self.assertEqual(len(subset), len(data.X.columns.values))
        for f in subset:
            self.assertIn(f, data.X.columns.values)
        # Test y label
        for tries in range(20):
            i = randint(0, len(df) - 1)
            self.assertEqual(dfnew['wt2'].values[i], data.y.values[i])

    def test_cvfolds(self):
        data = BorderData(df, cv)
        cvx = data.cvfolds(3)

        self.assertEqual(max(cvx[-1, -2]),
                         data.df[data.df.year == 2014].index.max())
        self.assertEqual(min(cvx[-1, -1]),
                         data.df[data.df.year == 2015].index.min())
        self.assertEqual(max(cvx[-1, -1]),
                         data.df[data.df.year == 2015].index.max())
        self.assertEqual(len(cvx[-1, -1]),
                         data.df[data.df.year == 2015].waittime.count())


if __name__ == '__main__':
    query = '''
            select
                c.date,
                waittime,
                year,
                month,
                dayofmonth,
                week,
                dayofweek,
                EXTRACT(MINUTE FROM time - '00:00:00'::time) + 60 * EXTRACT(HOUR FROM time - '00:00:00'::time) minofday
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

    df = pd_query(query)
    cv = cvfolds(df)

    features = ['year', 'month', 'dayofmonth', 'week', 'dayofweek', 'minofday']
    label = 'waittime'

    unittest.main()
