import cPickle as pickle
import datetime
import random
from dbhelper import pd_query


with open('../models/rf_v0.1.small.pkl') as f:
    model = pickle.load(f)


def get_features(start, end):
    query = '''
            select year, month, dayofweek, minofday
            from datefeatures
            where date >= '%s' and date <'%s'
            order by date
            '''
    return pd_query(query, (start, end))


def predict(start, end):
    X = get_features(start, end)
    yhat = model.predict(X)
    return yhat


def daily_prediction(date, location, direction='Southbound', lane='Cars'):
    start = datetime.datetime.strptime(date, '%Y-%m-%d')
    end = start + datetime.timedelta(hours=24)
    values = predict(start, end)

    time = start

    labels = []
    # values = []
    while time < end:
        # values.append(random.randint(0, 90))

        if time.minute == 0:
            labels.append(str(time.time())[:-3])
        else:
            labels.append("")
        time += datetime.timedelta(minutes=5)

    return labels, values


def daily_prediction_test(date, location, direction='Southbound', lane='Cars'):
    start = datetime.datetime(2016, 2, 19)      # Actual date is immaterial
    end = start + datetime.timedelta(hours=24)
    time = start

    labels = []
    values = []
    while time < end:
        values.append(random.randint(0, 90))

        if time.minute == 0:
            labels.append(str(time.time())[:-3])
        else:
            labels.append("")
        time += datetime.timedelta(minutes=5)

    return labels, values


if __name__ == '__main__':
    delays = daily_prediction(datetime.date('2/9/2016'), 'Peace Arch')
    print delays[:50]
