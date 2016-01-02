import numpy as np
import pandas as pd
import datetime as dt
import pdb
from sklearn.metrics import r2_score, mean_squared_error, \
    explained_variance_score
from sklearn.grid_search import GridSearchCV
from scipy.optimize import minimize
from dbhelper import pd_query
import statsmodels.api as sm
import copy
from ipywidgets import FloatProgress
from IPython.display import display
from BorderQuery import select_mungedata, select_predictions


def daily_average_features(series, sampling='30min', percent_nonnull=0.9,
                           rolling=True, lag=True, delta=True, numdelta=12):
    '''
    Builds rolling averages, lag averages and delta averages from a series
    Grain of 1 day

    IN
        series: series of label data with time as index
                assumes no gaps in times series
        sampling: sampling for output
        percent_nonnull: minimum percent of observations that each averaging
                         window must have to return a non-null average value
        rolling: boolean, enable rolling_means
        lag: boolean, enable lag averages
        delta: boolean, enable delta averages
    OUT
        dataframe with average features
            index shifted forward 1 day from input series
    '''
    # Create a series of averages by day
    daily = series.resample('D', how='mean')

    df = pd.DataFrame()

    # Calculate rolling averages
    if rolling:
        for days in [7, 14, 21, 28, 366]:
            df['avg_roll_{0}'.format(days)] = \
                pd.rolling_mean(daily, days, min_periods=days *
                                percent_nonnull)

    # Add lag averages
    # Note that index will be shifted at end of functions,
    # so lag=1 does not need to be shifted
    if lag:
        for days in range(1, 8):
            df['avg_lag_{0}'.format(days)] = daily.shift(days - 1)

    # Calculate deltas of averages
    if delta:
        for weeks in range(1, numdelta + 1):
            df['avg_delta_{0}'.format(weeks)] = daily - daily.shift(weeks * 7)
            df['avg_delta_{0}'.format(weeks)] = \
                df['avg_delta_{0}'.format(weeks)].fillna(method='pad')

    # Upsample to match original data
    # Resample ends at last index value, so intraday values are not filled
    # To reconcile, an additional day is added to dataframe
    ix = pd.DatetimeIndex(start=df.index[0].date(),
                          end=df.index[-1].date() + dt.timedelta(1), freq='D')
    df = df.reindex(ix).resample(sampling, fill_method='pad')

    # Shift forward to next day since averages are used as lag features
    df.index = df.index + dt.timedelta(1)

    # Remove the row corresponding to last day that was added above
    return df.ix[:-1]


class IncrementalModel(object):
    '''
    ATTRIBUTES
        df: training dataframe with features
            assume - waittime is label
        X_test: test feature matrix
            assume - same grain as training data
        categoricals: list of categorical prefixes
        sampling: sample period as offset alias, (default: 30 mins)
        model: initialized sklearn regressor
    '''
    def __init__(self, df, model, categoricals=None, sampling='30min',
                 averager=daily_average_features, rolling=True, lag=True,
                 delta=True, percent_nonnull=0.9, numdelta=12):
        self.model = model
        self.averager = averager
        self.sampling = sampling
        self.percent_nonnull = percent_nonnull
        self.rolling = rolling
        self.lag = lag
        self.delta = delta
        self.numdelta = numdelta

        # Prepare training data
        # Resample to remove gaps, inserting NA's
        # Averager needs time series without gaps
        self.df = df.copy()
        if 'date' in self.df:
            self.df = self.df.set_index('date')

        # Add categoricals
        self.categoricals = categoricals
        if self.categoricals is not None:
            self.df = handle_categoricals(self.df, self.categoricals)

        # Add rolling averages and lag averages to training data
        # Remove nulls introduced by resampling and averager
        self.df = self.df.resample(self.sampling)
        self.df = self.df.join(self.averager(self.df.waittime,
                               percent_nonnull=self.percent_nonnull,
                               rolling=self.rolling, lag=self.lag,
                               delta=self.delta, numdelta=self.numdelta))
        self.df = self.df.dropna()

        # Prepare X, y training data
        self.X = self.df.drop('waittime', 1)
        self.y = self.df.waittime

        # Fit the model
        self.model.fit(self.X, self.y)

    def predict(self, X_test):
        # Prepare test data
        # X_test should be properly sampled, but resample to be safe
        self.X_test = X_test.copy()
        if 'date' in self.X_test:
            self.X_test = self.X_test.set_index('date')
        self.X_test = self.X_test.resample(self.sampling)

        # Verify that training data and test data are contiguous
        if self.X.index[-1].date() != \
                self.X_test.index[0].date() - dt.timedelta(days=1):
            raise ValueError('Last day of training data must be day before \
first day of test data.')

        # Handle categoricals
        if self.categoricals is not None:
            self.X_test = handle_categoricals(self.X_test, self.categoricals)

        # Initialize for predictions
        predict = pd.Series()
        date = self.X_test.index[0].date()

        while sum(self.X_test.index.date == date) > 0:
            # Add rolling averages and lag averages to training data
            # Remove nulls introduced by averager
            # TODO: more performant approach that doesn't require recomputing
            #       averages for previous training data
            Xt_1 = self.X_test.join(self.averager(self.y.append(predict),
                                    percent_nonnull=self.percent_nonnull,
                                    rolling=self.rolling, lag=self.lag,
                                    delta=self.delta, numdelta=self.numdelta))
            Xt_1 = Xt_1[Xt_1.index.date == date]
            Xt_1 = Xt_1.dropna()
            if len(Xt_1) == 0:
                raise ValueError('Test data is all nulls after averager.  \
Consider setting percent_nonnull to lower value.')

            # Predict for 1 day
            predict_1 = pd.Series(self.model.predict(Xt_1), Xt_1.index)
            predict = predict.append(predict_1)

            date += dt.timedelta(days=1)

        self.y_predict = predict

        return predict

    def baseline(self):
        '''
        Compute baseline prediction based on day of week for last year of
        training data
        '''
        # Find last time period; correct for leap day
        last = self.y.index[-1]
        if last.month == 2 and last.day == 29:
            last = last - dt.timedelta(1)

        # Calculate day of week averages for last year
        y_last = self.y[(self.y.index > last - dt.timedelta(365)) &
                        (self.y.index <= last)]
        dow_avg = y_last.groupby([y_last.index.dayofweek,
                                  y_last.index.hour,
                                  y_last.index.minute]).mean().reset_index()
        dow_avg.columns = ['dow', 'hour', 'minute', 'waittime']

        # Build dataframe with same shape and date index as y_predict
        baseline = pd.DataFrame(self.y_predict.index,
                                self.y_predict.index)
        baseline['dow'] = baseline.index.dayofweek
        baseline['hour'] = baseline.index.hour
        baseline['minute'] = baseline.index.minute

        # Combine
        baseline = baseline.merge(dow_avg).set_index('date')

        return baseline.waittime

    def ensemble(self, actual, baseline):
        '''
        Ensemble predictions with baseline optimized for test values

        IN
            actual: test data
            baseline: baseline data
        '''
        # Compute weights.  Since actual may have missing data, predictions
        # and baseline are filtered
        # Harmonic mean is used for ensembling
        wy, wb = optimize_scalar_weights(actual,
                                         (self.y_predict.loc[actual.index],
                                          baseline.loc[actual.index]))

        print "Weights: ", wy, wb
        return harmonic_mean((baseline.loc[actual.index],
                             self.y_predict.loc[actual.index]), (wy, wb))

    def score(self, actual):
        if hasattr(self, 'y_predict'):
            actual = actual.resample(self.sampling, how='mean').dropna()

            model_r2 = r2_score(actual, self.y_predict.loc[actual.index])

            baseline = self.baseline()
            baseline_r2 = r2_score(actual, baseline.loc[actual.index])

            ensemble = self.ensemble(actual, baseline)
            ensemble_r2 = r2_score(actual, ensemble)

            return {'model': model_r2, 'ensemble': ensemble_r2,
                    'baseline': baseline_r2}
        else:
            raise RuntimeError('predict must be called before score')


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
    def __init__(self, df, years=3, label='waittime', categoricals=None):
        '''
        IN
            df: dataframe with date, features and label
            years: number of years in a train/test group
            label: label for y
        '''
        self.df = df.copy()
        self.date = df['date']

        # One hot encoding of categoricals
        if categoricals is not None:
            self.df = handle_categoricals(self.df, categoricals)

        self.X = self.df.drop('date', axis=1)

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
            # predict_weight = self.calculate_weights(model)
            predict_weight = None
            wy, wb = optimize_scalar_weights(self.y_test,
                                             (self.yhat, self.baseline),
                                             predict_weight=predict_weight)

            self.ensemble = harmonic_mean((self.yhat, self.baseline), (wy, wb))
            self.weights = (wy, wb)

    def calculate_weights(self, model):
        '''
        Calculate weights for predictions based on feature_importances_
        Includes spreading function

        [DO NOT USE: Does not perform as well as scalar weightings]

        IN
            model: trained sklearn model with feature_importances_
        OUT
            array of observation weights
        '''
        # Set feature weights based on log feature importances
        # Below threshold value, set weight to 1
        weight = np.log(model.feature_importances_)
        weight = [x + 9 if x > -6 else 1 for x in weight]
        # weight = 2

        # Non-event columns to exclude from weighting
        # But do not delete columns until after multiply to keep arrays aligned
        exclude = [i for i, x in enumerate(self.X.columns.values)
                   if 'event' not in x]

        events = np.array(np.delete((self.X_test * weight), exclude, 1).sum(1))
        return (np.convolve(events, [1, 2, 3, 3, 3, 2, 1]) + 1)[:len(events)]

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
            train = df[(df.date >= dt.date(test_year - years + 1, 1, 1))
                       & (df.date < dt.date(test_year, 1, 1))]
            test = df[(df.date >= dt.date(test_year, 1, 1))
                      & (df.date < dt.date(test_year + 1, 1, 1))]

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


# Helper functions for BorderImpute
def create_neighbor_features(dfin, feature='waittime', lead='lead', lag='lag'):
    '''
    Create lag/lead average values from a target observation
    Average weighted by distance from target observation

    IN
        dfin: input dataframe with individual lead/lag features
        feature: name of base feature for creating lag/lead features
        lead: string for lead feature
        lag: string for lag feature
    OUT
        dataframe with aggregate lead/lag features
    '''
    # Sort column names
    lead_cols = sorted([val for val in dfin.columns.values
                        if '{0}_{1}'.format(feature, lead) in val])
    lag_cols = sorted([val for val in dfin.columns.values
                       if '{0}_{1}'.format(feature, lag) in val])

    # Simple linear decay
    lead_weight = [len(lead_cols) - i for i in range(len(lead_cols))]
    lag_weight = [len(lag_cols) - i for i in range(len(lag_cols))]

    # Calculate weighted average, ignoring nulls
    df = pd.DataFrame()
    df[lead] = (dfin[lead_cols] * lead_weight).sum(1) /\
               (~pd.isnull(dfin[lead_cols]) * lead_weight).sum(1)
    df[lag] = (dfin[lag_cols] * lag_weight).sum(1) /\
              (~pd.isnull(dfin[lag_cols]) * lag_weight).sum(1)

    return df


def create_leadlag(dfin, feature='waittime', lead='lead', lag='lag', size=4):
    '''
    Create lag/lead features from a target observation

    IN
        dfin: input dataframe
        feature: name of base feature for creating lag/lead features
        lead: string for lead feature
        lag: string for lag feature
        size: number of neighbors to consider
    OUT
        dataframe with only lead/lag columns aligned to dfin
    '''
    df = pd.DataFrame()

    for i in range(1, size + 1):
        df['{0}_{1}_{2}'.format(feature, lead, i)] = dfin[feature].shift(-i)
        df['{0}_{1}_{2}'.format(feature, lag, i)] = dfin[feature].shift(i)

    return df


def xy_laglead(df, leadcols, lagcols, label='waittime', lead=False, lag=False):
    '''
    Prepare data with lag lead columns for modeling
    Note that lag and lead at data boundaries will have NaN values
    that must be removed from training data

    IN
        df: dataframe
            if lead = True, must have lead column
            if lag = True, must have lag column
        leadcols: column names of lead features (required)
        lagcols: column names of lag features (required)
        label: column name of label
        lead: boolean indicating that lead column is a feature
        lag: boolean indicating that lag column is a feature
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


def emulate_testdata(df, label='waittime', threshold=10, proportion=.85):
    '''
    Set label value to zero for values below threshold
    Algorithm for setting to zero includes some randomness to prevent a
    hard boundary at threshold

    IN
        dataframe with matching columns as source data
        label: name of lable column
        threshold
        proportion: percentage of values under threshold to set to zero
    OUT
        dataframe of modified label values aligned with df
        with columns date & *label*_new
    '''
    # Build non-linear weighting for values below threshold
    weight = threshold - df[label].clip(upper=threshold)
    weight = weight ** 2

    # Select values to set to zero
    dfzero = df.sample(n=int(proportion * sum(weight > 0)), weights=weight)

    # Set sample values to zero, aligned with original dataframe
    modvalues = df.copy()[label]
    modvalues[df.isin(dfzero).date.values] = 0

    # Construct dataframe with date and new label name
    dfout = df.copy()
    dfout[label] = modvalues

    return dfout


class BorderImpute(object):
    '''
    Class for imputing data using neighbor values as features
    Assumes 3 part modeling of neighbors
        * lead + lag effects
        * lead effects
        * lag effects

    ATTRIBUTES
        model_ll: model with lag/lead effects
        model_lead: model with lead effects
        model_lag: model with lag effects
        label: name of label
        neighborfunc: function that creates features based on neighbor data
        sourcedf: dataframe of source data with neighbor effects
        targetdf: dataframe of target data with neighbor effects
        predictdf: dataframe with latest predictions
    '''
    def __init__(self, label='waittime', threshold=10,
                 neighborfunc=create_neighbor_features):
        self.label = label
        self.threshold = threshold
        self.neighborfunc = neighborfunc
        pass

    def prepare_source(self, df):
        '''
        Prepare source data for modeling

        IN
            df: dataframe with features and label
        OUT
            dataframe of prepared source data
        '''
        # Add neighbor features
        df_neighbors = self.neighborfunc(create_leadlag(df),
                                         feature=self.label)
        self.sourcedf = df.join(df_neighbors)

        # Filter out values above threshold
        self.sourcedf = self.sourcedf[self.sourcedf.waittime < self.threshold]

        if 'date' in df.columns:
            self.sourcedf = self.sourcedf.set_index('date')

        return self.sourcedf

    def build_model(self, estimator):
        '''
        Trains each neighbor model

        IN
            estimator: initialized sklearn model
        OUT
            None
        '''
        X, y = xy_laglead(self.sourcedf, ['lead'], ['lag'], lag=True,
                          lead=True, label=self.label)
        self.model_ll = copy.copy(estimator)
        self.model_ll.fit(X, y)

        X, y = xy_laglead(self.sourcedf, ['lead'], ['lag'], lead=True,
                          label=self.label)
        self.model_lead = copy.copy(estimator)
        self.model_lead.fit(X, y)

        X, y = xy_laglead(self.sourcedf, ['lead'], ['lag'], lag=True,
                          label=self.label)
        self.model_lag = copy.copy(estimator)
        self.model_lag.fit(X, y)

    def load_models(self, model_ll, model_lead, model_lag):
        '''
        Can be used for to rerun predictions without having to retrain models

        IN
            model_ll: model with lag/lead effects
            model_lead: model with lead effects
            model_lag: model with lag effects
        '''
        self.model_ll = model_ll
        self.model_lead = model_lead
        self.model_lag = model_lag

    def prepare_target(self, df):
        '''
        Prepare target data for modeling.  Assumes that rows with zeros in
        label are rows that will be predicted.

        IN
            df: dataframe with features and label
        OUT
            dataframe of prepared source data
        '''
        df1 = df.copy()
        # Set all zero values to null
        df1[self.label] = df1[self.label].apply(lambda x:
                                                None if x == 0 else x)

        # Add neighbor features
        df_neighbors = self.neighborfunc(create_leadlag(df1),
                                         feature=self.label)
        self.targetdf = df1.join(df_neighbors)

        if 'date' in df.columns:
            self.targetdf = self.targetdf.set_index('date')

        return self.targetdf

    def predict(self):
        '''
        Iteratively predict values based on available neighbor data
        On each iteration, predictdf attribute is revised
        '''
        self.predictdf = self.targetdf.copy()
        dftest = self.predictdf.copy()
        dftest = dftest[pd.isnull(dftest[self.label])]

        # Set up progressbar
        nullcount = pd.isnull(self.predictdf[self.label]).sum()
        maxnullcount = nullcount
        f = FloatProgress(min=0, max=maxnullcount)
        display(f)

        while nullcount > 0:
            if ((~pd.isnull(dftest.lead)) &
               (~pd.isnull(dftest.lag))).sum() > 0:
                dftest = self.predict_once(dftest, lead=True, lag=True)

            if (~pd.isnull(dftest.lead)).sum() > 0:
                dftest = self.predict_once(dftest, lead=True)

            if (~pd.isnull(dftest.lag)).sum() > 0:
                dftest = self.predict_once(dftest, lag=True)

            nullcount = pd.isnull(self.predictdf[self.label]).sum()
            print nullcount
            f.value = maxnullcount - nullcount

    def predict_once(self, df, lead=False, lag=False):
        # Omit null values
        if lead:
            df = df[~pd.isnull(df.lead)]
        if lag:
            df = df[~pd.isnull(df.lag)]

        # Prepare feature matrix
        X_test = df.drop(self.label, 1)
        if not lag:
            X_test = X_test.drop('lag', 1)
        if not lead:
            X_test = X_test.drop('lead', 1)

        # Predict, with decay scaling for lead or lag only
        if lead and lag:
            yhat = self.model_ll.predict(X_test)
        elif lead:
            yhat = self.model_lead.predict(X_test) / 2.
        elif lag:
            yhat = self.model_lag.predict(X_test) / 2.
        else:
            raise RuntimeError('lead or lag must be True')

        # Add predictions to predictdf
        self.predictdf.loc[X_test.index, self.label] = yhat

        # Recalculate lead/lag
        df_neighbors = self.neighborfunc(create_leadlag(self.predictdf),
                                         feature=self.label)
        self.predictdf = self.predictdf.drop(['lag', 'lead'], 1)\
            .join(df_neighbors)

        return self.predictdf[pd.isnull(self.predictdf[self.label])]


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


def handle_categoricals(df, prefix):
    '''
    One hot encoding of features with matching suffix

    IN
        df: dataframe
        prefix: string prefix of feature name
    OUT
        dataframe with original feature removed and encoded features added
    '''
    df1 = df.copy()

    # TODO: match beginning of label name
    for cat in prefix:
        # Remove categoricals from df
        for col in df1.columns:
            if cat in col:
                df1 = df1.drop(col, 1)

        cols = [c for c in df.columns.values if cat in c]
        df1 = df1.join(create_dummies(df, cols))

    return df1


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


def optimize_scalar_weights(target, predict, predict_weight=None):
    '''
    IN
        target: actual values
        predict: list of predictions
    OUT
        list of weights
    '''
    try:
        res = minimize(_mse, [1, 1], args=(target, predict, predict_weight))
        return res.x
    except ValueError:
        print 'minimize unexplained ValueError.  Returning default weights'
        return [1, 1]


def _mse(weights, target, predict, predict_weight):
    '''
    IN
        target: actual values
        predict: list of predictions
    OUT
        MSE
    '''
    # pdb.set_trace()
    wy, wb = weights
    if predict_weight is not None:
        wb = wb * predict_weight

    return mean_squared_error(target, harmonic_mean(predict, (wy, wb)))


def model_years(df, model, start, end, categoricals=None):
    '''
    Run model over years from start to end

    IN
        df: dataframe with features and label
        model: initialized sklearn model
        start: int, start year
        end: int, end year
    '''
    trained = {}
    for year in range(start, end + 1):
        dfin = df.copy()[df.date < dt.date(year + 1, 1, 1)]
        print "Training... ", year
        data = BorderData(dfin, categoricals=categoricals)

        params = {}
        grid = GridSearchCV(model, params, cv=data.cv_train)
        grid.fit(data.X_train, data.y_train)

        data.predict(grid)
        data.predict_ensemble()
        print "Baseline : ", r2_score(data.y_test, data.baseline)
        print 'Model    : ', r2_score(data.y_test, data.yhat)
        print "Ensemble : ", r2_score(data.y_test, data.ensemble)

        trained[year] = (data, grid)

    return trained


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
