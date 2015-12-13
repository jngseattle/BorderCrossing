import cPickle as pickle
import datetime
import random


def daily_prediction(date, location, direction='Southbound', lane='Cars'):
    start = datetime.datetime(2016, 2, 19)
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
