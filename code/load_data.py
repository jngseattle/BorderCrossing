from dbhelper import MyDB, get_crossing_id
import multiprocessing as mp
import threading
import pymongo
import datetime


def spawn_processes():
    client = pymongo.MongoClient()
    mgdb = client.border

    pool = mp.Pool(processes=4)

    # split data into 1 year chunks
    chunks = []
    for year in range(2007, 2017):
        result = []

        # read from mongo and turn into a list of dicts so MP can pickle
        for res in mgdb.data.find(
            {'start': {'$gte': datetime.datetime(year, 1, 1),
                       '$lt': datetime.datetime(year + 1, 1, 1)}}):
            result.append(res)
        chunks.append(result)

    pool.map(insert_parallel, chunks)

    pool.close()


def insert_parallel(chunk):
    jobs = []

    # Tried multithreading but ran into issues likely due to DB connections
    # When run just as parallel processes, runs at full throttle, so threads
    # would not have helped anyway
    with MyDB() as db:
        for document in chunk:
            insert_one(db, document)


def insert_serial():
    client = pymongo.MongoClient()
    mgdb = client.border

    with MyDB() as db:
        for data in mgdb.data.find():
            insert_one(db, data)


def insert_one(db, data):
    query = 'insert into crossingdata (date, crossing_id, waittime, volume, valid) values(%s, %s, %s, %s, %s)'

    x_id = get_crossing_id(data['crossing_id'], data['lane'], data['dir'])
    dt = [datetime.datetime.utcfromtimestamp(epoch) for epoch in data['GroupStarts']]
    waittime = data['Values'][0]
    volume = data['Values'][4]
    valid = data['Values'][5]

    for date, wait, vol, vld in zip(dt, waittime, volume, valid):
        if vld == 0:
            db.cur.execute(query, (date, x_id, None, None, vld))
        else:
            db.cur.execute(query, (date, x_id, wait, vol, vld))

    db.conn.commit()


if __name__ == '__main__':
    spawn_processes()
    # insert_serial()
