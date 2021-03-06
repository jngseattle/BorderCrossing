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
from BorderQuery import select_mungedata, select_predictions, \
    select_features, select_mungedata_simple
import pprint


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
                 numdelta=12, optimize_weights=False):
        self.model = model
        self.sampling = sampling
        self.numdelta = numdelta
        self.actual = None
        self.optimize_weights = optimize_weights

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

        # Add delta averages to training data
        # Remove nulls introduced by resampling and averager
        self.df = self.df.resample(self.sampling)
        self.df = self.df.join(self.deltas(self.df.waittime))
        # Need to hold onto last day of y_test for predict
        self.y_test = self.df[self.df.index.date == max(self.df.index.date)].waittime
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

        # Handle categoricals
        # Align columns to X, excluding delta columns
        if self.categoricals is not None:
            columns = [col for col in self.X.columns.values
                       if 'avg_delta' not in col]
            self.X_test = handle_categoricals(self.X_test, self.categoricals,
                                              columns=columns)

        # Initialize for predictions
        predict = pd.Series()
        date = self.X_test.index[0].date()
        y_test = self.y_test.resample(self.sampling)

        while sum(self.X_test.index.date == date) > 0:
            # Add delta averages to training data
            # TODO: more performant approach that doesn't require recomputing
            #       averages for previous training data
            Xt_1 = self.X_test.join(self.deltas(y_test.append(predict)))
            Xt_1 = Xt_1[Xt_1.index.date == date]
            Xt_1 = Xt_1.dropna()
            if len(Xt_1) == 0:
                raise ValueError('Test data is all nulls after averager.')

            # Predict for 1 day
            predict_1 = pd.Series(self.model.predict(Xt_1), Xt_1.index)
            predict = predict.append(predict_1)

            date += dt.timedelta(days=1)

        self.y_predict = predict

        return predict

    def set_actual(self, series):
        '''
        IN
            series: series with datetime index
        '''
        self.actual = series

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

        return baseline.waittime.sort_index()

    def ensemble(self, actual=None, baseline=None):
        '''
        Ensemble predictions with baseline optimized for test values

        IN
            actual: test data
                    used for filtering predictions to match actual dataframe
                    so scoring can be properly computed
            baseline: baseline data
        '''
        if baseline is None:
            baseline = self.baseline()

        # Compute weights.  Since actual may have missing data, predictions
        # and baseline are filtered
        # Harmonic mean is used for ensembling
        if self.optimize_weights:
            wy, wb = optimize_scalar_weights(actual,
                                             (self.y_predict.loc[actual.index],
                                              baseline.loc[actual.index]))
            print "Weights: ", wy, wb
        else:
            wy, wb = 1, 1

        if actual is not None:
            return harmonic_mean((baseline.loc[actual.index],
                                  self.y_predict.loc[actual.index]), (wy, wb))
        else:
            return harmonic_mean((baseline, self.y_predict), (wy, wb))

    def score(self, actual=None):
        if hasattr(self, 'y_predict'):
            if actual is None:
                actual = self.actual
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

    def deltas(self, series):
        '''
        Builds delta averages from a series; grain of 1 day

        IN
            series: series of label data with time as index
                    assumes no gaps in times series
        OUT
            dataframe with average features
                index shifted forward 1 day from input series
        '''
        # Create a series of averages by day
        daily = series.resample('D', how='mean')

        # Construct deltas using time shift; fill forward for any gaps
        df = pd.DataFrame()
        for weeks in range(1, self.numdelta + 1):
            df['avg_delta_{0}'.format(weeks)] = daily - daily.shift(weeks * 7)
            df['avg_delta_{0}'.format(weeks)] = \
                df['avg_delta_{0}'.format(weeks)].fillna(method='pad')

        # Upsample to match original data
        # Resample function ends at last index value, so intraday values are not filled
        # To reconcile, an additional day is added to dataframe
        ix = pd.DatetimeIndex(start=df.index[0].date(),
                              end=df.index[-1].date() + dt.timedelta(1),
                              freq='D')
        df = df.reindex(ix).resample(self.sampling, fill_method='pad')

        # There are pathological cases where training data is missing
        # Backfill first, then zero fill
        df = df.fillna(method='bfill')
        df = df.fillna(0)

        # Shift forward to next day since deltas are used as lag features
        df.index = df.index + dt.timedelta(1)

        # Remove the row corresponding to last day that was added above
        return df.ix[:-1]


def run_Incremental(model, munger_id, xing, train_start, train_end,
                    test_start, test_end):
    df_train = select_mungedata(munger_id, xing, train_start, train_end)
    X_test = select_features(test_start, test_end)
    actual = select_mungedata_simple(munger_id, xing, test_start, test_end)

    grid = GridSearchCV(model, {})
    im = IncrementalModel(df_train, grid, categoricals=['event'])
    im.set_actual(actual.waittime)
    im.predict(X_test)

    return im


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
    def __init__(self, label='waittime', threshold=10, window=4,
                 neighborfunc=create_neighbor_features, progressbar=True):
        self.label = label
        self.threshold = threshold
        self.neighborfunc = neighborfunc
        self.window = window
        self.progressbar = progressbar

    def prepare_source(self, df):
        '''
        Prepare source data for modeling
        Values above threshold are removed to improve model for predicting
        values below threshold

        IN
            df: dataframe with features and label
        OUT
            dataframe of prepared source data
        '''
        # Add neighbor features
        df_neighbors = self.neighborfunc(create_leadlag(df, size=self.window),
                                         feature=self.label)
        self.sourcedf = df.join(df_neighbors)

        # Filter out values above threshold
        self.sourcedf = self.sourcedf[self.sourcedf.waittime < self.threshold]

        if 'date' in df.columns:
            self.sourcedf = self.sourcedf.set_index('date')

        return self.sourcedf

    def build_model(self, estimator):
        '''
        Trains 3 models which use lead and lag values for estimates
        * Lead + lag
        * Lead only
        * Lag only

        IN
            estimator: initialized sklearn model
        OUT
            None
        '''
        X, y = xy_laglead(self.sourcedf, ['lead'], ['lag'], lag=True,
                          lead=True, label=self.label)
        self.model_ll = copy.deepcopy(estimator)
        self.model_ll.fit(X, y)
        self.columns_ll = X.columns.values

        X, y = xy_laglead(self.sourcedf, ['lead'], ['lag'], lead=True,
                          label=self.label)
        self.model_lead = copy.deepcopy(estimator)
        self.model_lead.fit(X, y)
        self.columns_lead = X.columns.values

        X, y = xy_laglead(self.sourcedf, ['lead'], ['lag'], lag=True,
                          label=self.label)
        self.model_lag = copy.deepcopy(estimator)
        self.model_lag.fit(X, y)
        self.columns_lag = X.columns.values

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
        df_neighbors = self.neighborfunc(create_leadlag(df1, size=self.window),
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
        if self.progressbar:
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
            if self.progressbar:
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

        # Predict
        if lead and lag:
            yhat = self.model_ll.predict(X_test)
        elif lead:
            yhat = self.model_lead.predict(X_test)
        elif lag:
            yhat = self.model_lag.predict(X_test)
        else:
            raise RuntimeError('lead or lag must be True')

        # Add predictions to predictdf
        self.predictdf.loc[X_test.index, self.label] = yhat

        # Recalculate lead/lag
        df_neighbors = self.neighborfunc(create_leadlag(self.predictdf,
                                                        size=self.window),
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


def handle_categoricals(df, prefix, columns=None):
    '''
    One hot encoding of features with matching suffix

    IN
        df: dataframe
        prefix: string prefix of feature name
        columns: list of columns to match on output dataframe
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

    # Build a new dataframe aligned to 'columns'
    if columns is not None:
        dftemp = pd.DataFrame()
        for col in columns:
            if col in df1.columns:
                dftemp[col] = df1[col].values
            else:
                dftemp[col] = 0
        dftemp.index = df1.index.values
        dftemp.index.name = df1.index.name

        df1 = dftemp

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
        print 'scipy.optimize.minimize returned unexplained ValueError.  \
Returning default weights.'
        return [1, 1]


def _mse(weights, target, predict, predict_weight):
    '''
    IN
        target: actual values
        predict: list of predictions
    OUT
        MSE
    '''
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


def smooth(munger_id, crossing_id, field, limit=None, path='../data', df=None):
    '''
    Smooth data and write output to CSV

    IN
        munger_id
        crossing_id
        field: name of target field
        limit: string for limiting query; used for testing
        path: path to data directory
        df: dataframe to override default functionality of query raw data
            dataframe must have datetime index and data ordered by date
    OUT
        None
    '''
    if df is None:
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


def print_importances(model, columns):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(sort_importances)


def sort_importances(model, columns):
    return sorted(zip(columns, model.feature_importances_),
                  key=lambda x: x[1])[::-1]


def rolling_volume_aggregate(days, percent=.5):
    '''
    Returns a dataframe with date and multiple rolling_means of the average
    aggregrate volume inbalance between north and south crossings

    IN
        days: list of days for rolling_means
        percent: percentage of values that can be missing for rolling_mean to
                 not return NA
    OUT
        dataframe of form: date, vol_mean_1, vol_mean_2, etc.
                           for each day in days; nulls excluded from output
    '''
    series = pd_query('select date, volume from dailyvolume order by date')\
        .set_index('date').volume
    df = pd.DataFrame()

    for day in days:
        df['vol_mean_{0}'.format(day)] = \
            pd.rolling_mean(series, day, min_periods=day * percent).shift(1)

    df.index = pd.to_datetime(df.index)

    return df.dropna()
