import numpy as np
import pandas as pd
import datetime
import pdb
from sklearn.metrics import r2_score, mean_squared_error, \
    explained_variance_score
from scipy.optimize import minimize
from dbhelper import pd_query
import statsmodels.api as sm


class BorderData(object):
    '''
        ATTRIBUTES
            date: date series
            X: feature dataframe
            y: label series
            label: name of label
            cv: all cross validation folds
            cv_train: cross validation folds for training set
            test_indices: indices of test set
            X_train: feature dataframe for training set
            X_test: feature dataframe for test set
            y_train: label series for training set
            y_test: label series for test set
            baseline: series for last years data averaged by day of week
            yhat: series of predicted values
    '''
    def __init__(self, df, years=3, label='waittime'):
        '''
        IN
            df: dataframe with date, features and label
            years: number of years in a train/test group
            label: label for y
        '''
        self.df = df.copy()
        self.date = df['date']

        # Handle event categoricals
        events = self._event_dummies(df.copy())

        self.X = df.drop('date', axis=1)

        # Remove labels from X
        for l in ['waittime', 'volume', label]:
            if l in self.X.columns.values:
                self.X = self.X.drop(l, axis=1)

        self.label = label
        self.y = df[label]

        # cv can be used for training with all data
        self.cv = self._cvfolds(years)

        # When optimizing model, most current year (e.g., 2015) is test set
        # all other years are training set
        self.cv_train = self.cv[:-1]
        self.test_indices = self.cv[-1][1]
        self._prepare_train_test()
        self.baseline = self.baseline_model(label=self.label)['baseline']

    def _event_dummies(self, df):
        cols = [c for c in df.columns.values if 'event' in c]
        return create_dummies(df, cols)

    def _prepare_train_test(self):
        '''
        Create train test set
        '''
        # Set last fold as test set
        self.X_test = np.array(self.X)[self.test_indices]
        self.y_test = np.array(self.y)[self.test_indices]
        # Remove last fold from training set
        self.X_train = np.delete(np.array(self.X), self.test_indices, 0)
        self.y_train = np.delete(np.array(self.y), self.test_indices, 0)

    def baseline_model(self, label='waittime'):
        '''
        Return dataframe of last year of training set averaged over day of week
        '''
        low = self.cv_train[-1][1][0]
        high = self.cv_train[-1][1][-1]
        df = pd.DataFrame(self.df.iloc[low:high + 1]
                          .groupby(['dayofweek', 'minofday'])[label]
                          .mean())

        df.columns = ['baseline']

        df = self._df_last().reset_index() \
                 .merge(df.reset_index(), how='left',
                        on=['dayofweek', 'minofday']).set_index('date') \
                 .sort_index()

        return df

    def predict(self, model):
        '''
        IN
            model: trained sklearn model which used data from this object
        OUT
            Dataframe of test set with predictions for plotting
        '''
        df = self._df_last()

        df['prediction'] = model.predict(self.X_test)
        df.prediction = df.prediction.clip(lower=0)

        self.yhat = df['prediction']
        return df

    def predict_ensemble(self, model=None):
        '''
        Calculates harmonic mean
        IN
            model (optional)
        OUT
            TBD
        '''
        if self.yhat is None and model is None:
            raise RuntimeError('model parameter missing')

        if model is not None and self.yhat is None:
            self.predict(model)

        if self.yhat is not None:
            wy, wb = calculate_weights(self.y_test, (self.yhat, self.baseline))
            self.ensemble = harmonic_mean((self.yhat, self.baseline), (wy, wb))
            self.weights = (wy, wb)

    def _df_last(self):
        '''
        Returns dataframe for last fold in data
        '''
        df = self.df.copy()
        df = df.set_index('date')
        return df.iloc[self.test_indices[0]:self.test_indices[-1] + 1]

    # TODO: support for skipped years - LOW priority
    def _cvfolds(self, years):
        '''
        IN
            X: dataframe with date column, assumed ordered by date
            years: number of years for group; 3 --> 2 train + 1 test
        OUT
            Train/test indices to split data in train test sets.
        '''
        df = self.df
        min_year = df.date.min().year
        max_year = df.date.max().year

        cv = []
        test_year = min_year + years - 1

        while test_year <= max_year:
            train = df[(df.date >= datetime.date(test_year - years + 1, 1, 1))
                       & (df.date < datetime.date(test_year, 1, 1))]
            test = df[(df.date >= datetime.date(test_year, 1, 1))
                      & (df.date < datetime.date(test_year + 1, 1, 1))]

            cv.append((list(train.index), list(test.index)))

            test_year += 1

        return np.array(cv)

    def plot(self, start, end):
        '''
        Plot waittime, baseline and prediction
        Assumes model has been fit
        '''
        df = self._df_last()
        df['baseline'] = self.baseline

        if self.yhat is None:
            raise RuntimeError('yhat is null.  Run predict method first.')

        df['prediction'] = self.yhat
        df = df.loc[(df.index > start) & (df.index <= end)]
        df[['waittime', 'baseline', 'prediction']].plot(figsize=(14, 6),
                                                        alpha=.7)

    def print_metrics(self, model):
        '''
        Print comparison metrics between baseline and model
        IN
            model: trained sklearn model object
        '''
        # Built-in model metrics
        if hasattr(model, 'best_estimator_'):
            if hasattr(model.best_estimator_, 'oob_score_'):
                print "OOB: ", model.best_estimator_.oob_score_
        if hasattr(model, 'best_score_'):
            print "Best score: ", model.best_score_

        # MSE
        df = self._df_last()
        print "** MSE for last cv fold **"
        print "Baseline : ", mean_squared_error(self.y_test, self.baseline)
        print "Model    : ", mean_squared_error(self.y_test, self.yhat)
        if hasattr(self, 'ensemble'):
            print "Ensemble : ", mean_squared_error(self.y_test,
                                                    self.ensemble)
            print "Weights  : ", str(self.weights)

        # R^2
        print "** R^2 for last cv fold **"
        print "Baseline : ", r2_score(self.y_test, self.baseline)
        print 'Model    : ', r2_score(self.y_test, self.yhat)
        if hasattr(self, 'ensemble'):
            print "Ensemble : ", r2_score(self.y_test, self.ensemble)

        # Explained Variance
        print "** Explained variance for last cv fold **"
        print "Baseline : ", explained_variance_score(self.y_test,
                                                      self.baseline)
        print 'Model    : ', explained_variance_score(self.y_test,
                                                      self.yhat)
        if hasattr(self, 'ensemble'):
            print "Ensemble : ", explained_variance_score(self.y_test,
                                                          self.ensemble)


class BorderImpute(object):
    '''
    Class for imputing data using neighbor values as features
    Assumes 3 part modeling of neighbors
        * lead + lag effects
        * lead effects
        * lag effects
    Assumes all models will include volume and date features
    Date features can be overridden

    ATTRIBUTES
        model_ll
        model_lead
        model_lag
        dfsource: dataframe of source data with neighbor effects
    '''
    def __init__(self, start, end, neighborfunc):
        self.start = start
        self.end = end
        pass

    def prepare_source(self, query):
        '''
        Select data and prepare for modeling
        * Raw wait time
        * Smoothed volume
        * Date features
        Add neighbor data

        IN
            munger_id
            crossing_id
        OUT
            dataframe of prepared source data
        '''
        # Get data from DB
        # Get lead feature as series
        # Get lag feature as series
        # Combine in dataframe, excluding NaN
        pass

    def build_model(self, estimator):
        '''
        Run fit for each neighbor model
        '''
        self.model_ll = copy.copy(estimator)
        self.model_ll.fit(xy_laglead(self.sourcedf, lag=True, lead=True))

        self.model_lead = copy.copy(estimator)
        self.model_lead.fit(xy_laglead(self.sourcedf, lead=True))

        self.model_lag = copy.copy(estimator)
        self.model_lag.fit(xy_laglead(self.sourcedf, lag=True))


def xy_laglead(df, leadcols, lagcols, label='waittime', lead=False, lag=False):
    '''
    Prepare data with lag lead columns for modeling
    Note that lag and lead at data boundaries will have NaN values
    that must be removed from training data
    IN
        df: dataframe
            if lead = True, must have lead column
            if lag = True, must have lag column
        label: column name of label
        lead: boolean indicating that lead column is a feature
        lag: boolean indicating that lag column is a feature
        leadcols: column names of lead features (required)
        lagcols: column names of lag features (required)
    OUT
        X dataframe with lead/lag columns if lead/lag are true respectively
        y label series
    '''
    nanfilter = np.ones(len(df)).astype(bool)
    if lead:
        for col in leadcols:
            nanfilter &= ~pd.isnull(df[col])
    if lag:
        for col in lagcols:
            nanfilter &= ~pd.isnull(df[col])

    y = df.copy()[nanfilter][label]

    X = df.copy()[nanfilter].drop(label, 1)
    if not lead:
        X = X.drop(leadcols, 1)
    if not lag:
        X = X.drop(lagcols, 1)

    return X, y


def add_neighbors(dfin, colin, colout, size=4):
    '''
    IN
        dfin: input dataframe
        colin: column name for input data
        colout: column name of output data
        size: number of neighbors to consider
    OUT
        dataframe with new column added
    '''
    pass


def clean_df_subset(df, subset, label='waittime'):
    '''
    IN
        df: dataframe with date, features and label
        subset: list of columns to keep
        label: label to keep
    OUT
        clean dataframe with date, label and select features
    '''
    dfnew = df[['date', label]]
    dfnew = dfnew.join(df[subset])
    return dfnew


def create_dummies(df, cols, drop=False):
    '''
    IN
        df - original dataframe
        cols - list of categorical columns
        drop - boolean for dropping columns for linear models
    OUT
        dataframe of only categorical dummy columns
    '''
    newdf = df.copy()
    newdf['i'] = newdf.index.values   # adding a column just for join purposes
    newdf = newdf[['i']]

    for col in cols:
        newdf = newdf.join(pd.get_dummies(df[col], prefix=col))
        # Drop a dummy variable from each column to remove colinearity
        if drop:
            newdf = newdf.drop(newdf.columns[len(newdf.columns) - 1], axis=1)

    newdf = newdf.drop('i', axis=1)

    return newdf


def harmonic_mean(data, weights):
    '''
    IN
        data: list of lists of predictions
        weights: list of weights
    OUT
        harmonic mean
    '''
    denom = sum([w / d for w, d in zip(weights, data)])
    return sum(weights) / denom


def calculate_weights(target, predict):
    '''
    IN
        target: actual values
        predict: list of predictions
    OUT
        list of weights
    '''
    res = minimize(_mse, [2, 1], args=(target, predict))
    return res.x


def _mse(weights, target, predict):
    '''
    IN
        target: actual values
        predict: list of predictions
    OUT
        MSE
    '''
    return mean_squared_error(target, harmonic_mean(predict, weights))


def smooth(munger_id, crossing_id, field, limit=None, path='../data'):
    '''
    Smooth data and write output to CSV

    IN
        munger_id
        crossing_id
        dataframe with date and data field ordered by date
    OUT
        None
    '''
    query = '''
            select
                c.date,
                {0}
            from crossingdata c
            join datefeatures d on c.date = d.date
            where
                valid=1
                and {0} is not null
                and crossing_id = {1}
            order by c.date {2};
            '''

    if limit is not None:
        limitstring = "limit %s" % (limit)
    else:
        limitstring = ""
    df = pd_query(query.format(field, crossing_id, limitstring))

    lowess = sm.nonparametric.lowess
    z = lowess(df[field], df.index, frac=12. / len(df), it=1)

    df['smooth'] = z[:, 1]
    df.smooth = df.smooth.clip_lower(0)

    dfcsv = df.reset_index()[['date', 'smooth']]
    dfcsv['munger_id'] = munger_id
    dfcsv['crossing_id'] = crossing_id
    dfcsv['is_waittime'] = field == 'waittime'

    filepath = '{0}/munge{1}_{2}_{3}.csv'.format(path, munger_id,
                                                 crossing_id, field)
    dfcsv.to_csv(filepath,
                 columns=['munger_id', 'crossing_id', 'date',
                          'smooth', 'is_waittime'],
                 index=False, header=False)

    return df
