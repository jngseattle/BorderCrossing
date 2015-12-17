import numpy as np
import pandas as pd
import datetime

class BorderData(object):
    '''
        ATTRIBUTES
            date: date series
            X: feature dataframe
            y: label series
            label: name of label
            cv_train: cross validation folds for training set
            test_indices: indices of test set
            X_train: feature dataframe for training set
            X_test: feature dataframe for test set
            y_train: label series for training set
            y_test: label series for test set
    '''
    def __init__(self, df, cv, label='waittime'):
        '''
        IN
            df: dataframe with date, features and label
            cv: cross validation folds
            label: label for y
        '''
        self.df = df.copy()
        self.date = df['date']
        self.X = df.drop('date', axis=1)

        # Remove labels from X
        for l in ['waittime', 'volume', label]:
            if l in self.X.columns.values:
                self.X = self.X.drop(l, axis=1)

        self.label = label
        self.y = df[label]
        self.cv_train = cv[:-1]
        self.test_indices = cv[-1][1]
        self.prepare_train_test()

    def prepare_train_test(self):
        '''
        Create train test set
        '''
        # Set last fold as test set
        self.X_test = np.array(self.X)[self.test_indices]
        self.y_test = np.array(self.y)[self.test_indices]
        # Remove last fold from training set
        self.X_train = np.delete(np.array(self.X), self.test_indices, 0)
        self.y_train = np.delete(np.array(self.y), self.test_indices, 0)

    def baseline_model(self):
        '''
        Return dataframe of last year of training set averaged over day of week
        '''
        low = self.cv_train[-1][1][0]
        high = self.cv_train[-1][1][-1]
        return pd.DataFrame(self.df.iloc[low:high + 1]
                            .groupby(['dayofweek', 'minofday'])
                            .waittime.mean())

    def prediction(self, model):
        '''
        IN
            model: trained model which used data from this object
        OUT
            Dataframe of test set with predictions for plotting
        '''
        df = self.df.copy()
        df = df.set_index('date')
        df = df.iloc[self.test_indices[0]:self.test_indices[-1] + 1]

        df['prediction'] = model.predict(self.X_test)
        return df

    def cvfolds(self, years=3):
        '''
        IN
            X: dataframe with date column, assumed ordered by date
            years: number of years for group; 2 train + 1 test
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


def cvfolds(X, years=3):
    '''
    IN
        X: dataframe with date column, assumed ordered by date
        years: number of years for group; 2 train + 1 test
    OUT
        Train/test indices to split data in train test sets.
    '''
    min_year = X.date.min().year
    max_year = X.date.max().year

    cv = []
    test_year = min_year + years - 1

    while test_year <= max_year:
        train = X[(X.date >= datetime.date(test_year - years + 1, 1, 1))
                  & (X.date < datetime.date(test_year, 1, 1))]
        test = X[(X.date >= datetime.date(test_year, 1, 1))
                 & (X.date < datetime.date(test_year + 1, 1, 1))]
        cv.append((list(train.index), list(test.index)))

        test_year += 1

    return cv
