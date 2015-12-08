import mysql.connector
import psycopg2
import os
import datetime
import pandas as pd


def pd_query(query, vals, result=None, i=None):
    '''
    Simple routine to query with dataframe result (supports threads)
    For threading, call with:
        threading.Thread(target=pd_query, args=(query, vals, result, i))
    IN: query (str) : query string
        vals (list) : query variables
        results (list) : initialized to number of threads
        i (int) : thread index
    OUT: Pandas dataframe (also written to results[i])
    '''
    with PgDB() as db:
        res = pd.read_sql(query % vals, con=db.conn)
        if result is not None:
            result[i] = res

    return res

def get_directions():
    dirs = {}

    with PgDB() as db:
        db.cur.execute("select id, name from direction")
        for id, name in db.cur:
            dirs[name] = id

    return dirs


def get_lanes():
    lanes = {}

    with PgDB() as db:
        db.cur.execute("select id, name from lane")

        for id, name in db.cur:
            lanes[name] = id

    return lanes


def get_crossings():
    '''
    Return list of all valid crossings, e.g. location/lane/direction combos
    '''
    query = '''
            select
                crossing.id,
                location_id,
                lane_id,
                direction_id,
                location.name as location_name,
                lane.name as lane_name,
                direction.name as direction_name
            from crossing
            join location on location.id = location_id
            join lane on lane.id = lane_id
            join direction on direction.id = direction_id
            order by location.name, lane.name, direction.name;
            '''

    data = []
    with PgDB() as db:
        return pd.read_sql(query, con=db.conn)
    #     db.cur.execute(query)
    #     for row in db.cur:
    #         data.append({'id': row[0],
    #                      'location_id': row[1],
    #                      'lane_id': row[2],
    #                      'direction_id': row[3],
    #                      'location_name': row[4],
    #                      'lane_name': row[5],
    #                      'direction_name': row[6]})
    # return data


def get_crossing_id(location_id, lane_name, dir_name):
    '''
    IN: location_id (int)
        lane_name (str)
        dir_name (str) [Northbound/Southbound]
    OUT: id (int) for matching record in crossing table
    '''
    with PgDB() as db:
        query = '''
                select crossing.id from crossing
                join lane on lane.id = lane_id
                join direction on direction.id = direction_id
                where location_id=%s
                and lane.name=%s
                and direction.name=%s
                '''
        db.cur.execute(query, (location_id, lane_name, dir_name))

        count = 0
        for (id, ) in db.cur:
            count += 1
            if count > 1:
                raise ValueError('More than one crossing id found')

    return int(id)


def get_oneday(start, location, direction, lane):
    end = start + datetime.timedelta(days=1)
    return get_crossingdata(start, end, location, direction, lane)


def get_oneweek(start, location, direction, lane):
    end = start + datetime.timedelta(weeks=1)
    return get_crossingdata(start, end, location, direction, lane)


def get_crossingdata(start, end, location, direction, lane):
    with PgDB() as db:
        query = '''
                select * from crossingdata_denorm
                where date >= '%s' and date < '%s'
                and location_name = '%s'
                and direction_name = '%s'
                and lane_name = '%s'
                '''
        df = pd.read_sql(query % (str(start),
                                  str(end),
                                  location,
                                  direction,
                                  lane), con=db.conn)

    return df


def get_dow_average(start, end, location, direction, lane):
    '''
    Day of week average
    '''
    with PgDB() as db:
        query = '''
                select
                    EXTRACT(DOW FROM date) as dow,
                    date::time as time,
                    avg(waittime) as avg_wt,
                    avg(volume) as avg_vol,
                    stddev_samp(waittime) as std_wt,
                    stddev_samp(volume) as std_vol,
                    sum(valid) as sum_valid
                from crossingdata_denorm
                where date >= '%s' and date < '%s'
                    and location_name = '%s'
                    and direction_name = '%s'
                    and lane_name = '%s'
                    and valid = 1
                group by EXTRACT(DOW FROM date),
                    date::time
                '''
        df = pd.read_sql(query % (str(start),
                                  str(end),
                                  location,
                                  direction,
                                  lane), con=db.conn)
        return df.sort_values(['dow', 'time'])


class MyDB:
    def __enter__(self):
        self.conn = mysql.connector.connect(
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASSWORD'),
            host='localhost',
            database='border')
        self.cur = self.conn.cursor()
        return self

    def __exit__(self, *args):
        self.conn.close()
        self.cur.close()


class PgDB:
    def __enter__(self):
        self.conn = psycopg2.connect(
            "dbname=border user={0}".format(os.getenv('POSTGRESQL_USER')))
        self.cur = self.conn.cursor()
        return self

    def __exit__(self, *args):
        self.conn.close()
        self.cur.close()


if __name__ == '__main__':
    print get_directions()
    print get_lanes()
    print get_crossing_id(134, 'Nexus', 'Northbound')
    print get_crossings()[:3]

    start = datetime.datetime(2015, 11, 1, 12, 0, 0)
    end = start + datetime.timedelta(minutes=30)
    print get_crossingdata(start, end, 'Peace Arch', 'Northbound', 'Car')

    start = datetime.datetime(2015, 10, 1, 1)
    end = datetime.datetime(2015, 11, 1, 1)
    print get_dow_average(start, end, 'Peace Arch', 'Northbound', 'Car')
