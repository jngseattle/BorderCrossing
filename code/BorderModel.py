import numpy as np
import pandas as pd
import datetime
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

        # Handle categoricals
        if categoricals is not None:
            for cat in categoricals:
                # Remove categoricals from self.df
                for col in self.df.columns:
                    if cat in col:
                        self.df = self.df.drop(col, 1)
                # Add dummified categoricals
                self.df = self.df.join(self._dummies(df.copy(), cat))

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

    def _dummies(self, df, categorical):
        cols = [c for c in df.columns.values if categorical in c]
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

    def calculate_weights():
        '''
        Apply higher weight to dates for events with high feature importance
        '''
        pass

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
        dfin = df.copy()[df.date < datetime.date(year + 1, 1, 1)]
        print "Training... ", year
        data = BorderData(dfin, categoricals=categoricals)

        params = {}
        grid = GridSearchCV(model, params, cv=data.cv_train)
        grid.fit(data.X_train, data.y_train)

        data.predict(grid)
        print "Baseline : ", r2_score(data.y_test, data.baseline)
        print 'Model    : ', r2_score(data.y_test, data.yhat)

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
