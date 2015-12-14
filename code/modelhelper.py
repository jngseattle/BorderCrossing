import numpy as np
import datetime
import pandas as pd


def cvfolds(X, years=3):
    '''
    IN: X - dataframe with date column, assumed ordered by date
        years - number of years for group; 2 train + 1 test
    OUT: Train/test indices to split data in train test sets.
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


if __name__ == '__main__':
    pass
