import cPickle as pickle
import datetime
import random
from dbhelper import pd_query, get_crossings
import pdb


# Set globals
# TODO: read from config file - LOW PRI
CROSSINGS = get_crossings()
MUNGER = 2
PMODEL = 'v0.5'      # Prediction model
BMODEL = 'b2014'     # Baseline model


def predict(location, direction, lane, start, end):
    '''
    IN
        location: name of crossing
        direction: name of direction
        lane: name of lane
        start: datetime
        end: datetime
    OUT
        dataframe with date, prediction and baseline
    '''
    xid = CROSSINGS[(CROSSINGS.location_name == location)
                    & (CROSSINGS.direction_name == direction)
                    & (CROSSINGS.lane_name == lane)].index[0]

    if id is None:
        raise RuntimeError('Crossing not matched for: ', location,
                           direction, lane)

    query = '''
            select
                p.date,
                p.waittime as predict,
                b.waittime as baseline
            from predictions p
            inner join predictions b
                on p.date = b.date
                and b.crossing_id = {3}
                and b.munger_id = '{2}'
                and b.model_version = '{1}'
            where
                p.date >= '{4}' and p.date < '{5}'
                and p.crossing_id = {3}
                and p.munger_id = '{2}'
                and p.model_version = '{0}'
            order by p.date
            '''

    df = pd_query(query.format(PMODEL, BMODEL, MUNGER, xid, start, end))
    return df


def daily_prediction(start, location, direction='Southbound', lane='Car'):
    '''
    IN
        start: datetime
        location: name of crossing
        direction: name of direction
        lane: name of lane
    '''
    end = start + datetime.timedelta(hours=24)
    df = predict(location, direction, lane, start, end)

    return df


if __name__ == '__main__':
    delays = daily_prediction(datetime.datetime(2016, 1, 10), 'Pacific Highway')
    print delays[:50]
