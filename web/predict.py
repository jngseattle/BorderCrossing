import cPickle as pickle
import datetime
import random
from dbhelper import pd_query, get_crossings
import pdb


# Set globals
# TODO: read from config file - LOW PRI
CROSSINGS = get_crossings()
PMODEL = 'v2.1'      # Prediction model
BMODEL_POST2016 = 'b2015'     # Baseline model
BMODEL_PRE2016 = 'b.1year'     # Baseline model


def get_prediction(start, location, direction, lane):
    xid, start, end = get_chart_params(start, location, direction, lane)

    query = '''
            select
                d.date,
                waittime
            from datefeatures d
            left join predictions p
                on d.date = p.date
                and crossing_id = {1}
                and model_version = '{0}'
            where
                d.date >= '{2}' and d.date < '{3}'
                and (minute = 0 or minute=30)
            order by date
            '''

    df = pd_query(query.format(PMODEL, xid, start, end))

    return df


def get_baseline(start, location, direction, lane):
    xid, start, end = get_chart_params(start, location, direction, lane)

    query = '''
            select
                d.date,
                waittime
            from datefeatures d
            left join predictions p
                on d.date = p.date
                and crossing_id = {1}
                and model_version = '{0}'
            where
                d.date >= '{2}' and d.date < '{3}'
                and (minute = 0 or minute=30)
            order by date
            '''

    # Before 2016, baseline is computed from a rolling average
    # After 2016, baseline is same as last week of 2015
    if start.year >= 2016:
        df = pd_query(query.format(BMODEL_POST2016, xid, start, end))
    else:
        df = pd_query(query.format(BMODEL_PRE2016, xid, start, end))

    return df


def get_actual(start, location, direction, lane):
    xid, start, end = get_chart_params(start, location, direction, lane)

    query = '''
            select
                d.date,
                waittime
            from datefeatures d
            left join crossingdata c
                on d.date = c.date
                and crossing_id = {0}
            where
                d.date >= '{1}' and d.date < '{2}'
                and (minute = 0 or minute=30)
            order by date
            '''

    df = pd_query(query.format(xid, start, end))
    return df


def get_chart_params(start, location, direction, lane):
    end = start + datetime.timedelta(hours=24)

    xid = CROSSINGS[(CROSSINGS.location_name == location)
                    & (CROSSINGS.direction_name == direction)
                    & (CROSSINGS.lane_name == lane)].index[0]

    if id is None:
        raise RuntimeError('Crossing not matched for: ', location,
                           direction, lane)

    return xid, start, end


if __name__ == '__main__':
    pass
