from dbhelper import pd_query, PgDB


def select_mungedata(munger_id, crossing_id, start_date):
    query = '''
            select
                m.date,
                metric as waittime,
                year,
                month,
                week,
                dayofweek,
                minofday,
                w.temp_max,
                w.temp_mean,
                w.temp_min,
                w.viz_max,
                w.wind_max,
                w.precip,
                w.rain,
                w.snow,
                w.fog,
                w.thunderstorm,
                wp1.temp_max as temp_max_p1,
                wp1.temp_mean as temp_mean_p1,
                wp1.temp_min as temp_min_p1,
                wp1.precip as precip_p1,
                wp1.rain as rain_p1,
                wp1.snow as snow_p1,
                wp1.thunderstorm as thunderstorm_p1,
                wp2.temp_max as temp_max_p2,
                wp2.temp_mean as temp_mean_p2,
                wp2.temp_min as temp_min_p2,
                wp2.precip as precip_p2,
                wp2.rain as rain_p2,
                wp2.snow as snow_p2,
                wp2.thunderstorm as thunderstorm_p2,
                wp3.temp_max as temp_max_p3,
                wp3.temp_mean as temp_mean_p3,
                wp3.temp_min as temp_min_p3,
                wp3.precip as precip_p3,
                wp3.rain as rain_p3,
                wp3.snow as snow_p3,
                wp3.thunderstorm as thunderstorm_p3,
                wm1.temp_max as temp_max_m1,
                wm1.temp_mean as temp_mean_m1,
                wm1.temp_min as temp_min_m1,
                wm1.precip as precip_m1,
                wm1.rain as rain_m1,
                wm1.snow as snow_m1,
                wm1.thunderstorm as thunderstorm_m1,
                wm2.temp_max as temp_max_m2,
                wm2.temp_mean as temp_mean_m2,
                wm2.temp_min as temp_min_m2,
                wm2.precip as precip_m2,
                wm2.rain as rain_m2,
                wm2.snow as snow_m2,
                wm2.thunderstorm as thunderstorm_m2,
                s.event,
                s_lead1.event as event_lead1,
                s_lag1.event as event_lag1,
                s_lead2.event as event_lead2,
                s_lag2.event as event_lag2,
                s_lead3.event as event_lead3,
                s_lag3.event as event_lag3,
                s_lead4.event as event_lead4,
                s_lag4.event as event_lag4,
                1 as sea,
                1 as sea_lag1,
                1 as sea_lead1,
                1 as sea_lag2,
                1 as sea_lead2,
                1 as sea_lag3,
                1 as sea_lead3,
                1 as van,
                1 as van_lag1,
                1 as van_lead1,
                1 as van_lag2,
                1 as van_lead2,
                1 as van_lag3,
                1 as van_lead3
            from mungedata m
            join datefeatures d on m.date = d.date
            left join publicholiday h on m.date::timestamp::date = h.date
            left join weather w on m.date::timestamp::date = w.date
            left join weather wp1 on m.date::timestamp::date =
                wp1.date - interval '1 day'
            left join weather wp2 on m.date::timestamp::date =
                wp2.date - interval '2 day'
            left join weather wp3 on m.date::timestamp::date =
                wp3.date - interval '3 day'
            left join weather wm1 on m.date::timestamp::date =
                wm1.date + interval '1 day'
            left join weather wm2 on m.date::timestamp::date =
                wm2.date + interval '2 day'
            left join specialdates s on m.date::timestamp::date = s.date
            left join specialdates s_lead1 on m.date::timestamp::date =
                s_lead1.date - interval '1 day'
            left join specialdates s_lag1 on m.date::timestamp::date =
                s_lag1.date + interval '1 day'
            left join specialdates s_lead2 on m.date::timestamp::date =
                s_lead2.date - interval '2 day'
            left join specialdates s_lag2 on m.date::timestamp::date =
                s_lag2.date + interval '2 day'
            left join specialdates s_lead3 on m.date::timestamp::date =
                s_lead3.date - interval '3 day'
            left join specialdates s_lag3 on m.date::timestamp::date =
                s_lag3.date + interval '3 day'
            left join specialdates s_lead4 on m.date::timestamp::date =
                s_lead4.date - interval '4 day'
            left join specialdates s_lag4 on m.date::timestamp::date =
                s_lag4.date + interval '4 day'
            left join schoolcalendar sea on m.date::timestamp::date =
                sea.date_out and sea.district='seattle'
            left join schoolcalendar sea_lag1 on m.date::timestamp::date =
                sea_lag1.date_out + interval '1 day'
                and sea_lag1.district='seattle'
            left join schoolcalendar sea_lead1 on m.date::timestamp::date =
                sea_lead1.date_out - interval '1 day'
                and sea_lead1.district='seattle'
            left join schoolcalendar sea_lag2 on m.date::timestamp::date =
                sea_lag2.date_out + interval '2 day'
                and sea_lag2.district='seattle'
            left join schoolcalendar sea_lead2 on m.date::timestamp::date =
                sea_lead2.date_out - interval '2 day'
                and sea_lead2.district='seattle'
            left join schoolcalendar sea_lag3 on m.date::timestamp::date =
                sea_lag3.date_out + interval '3 day'
                and sea_lag3.district='seattle'
            left join schoolcalendar sea_lead3 on m.date::timestamp::date =
                sea_lead3.date_out - interval '3 day'
                and sea_lead3.district='seattle'
            left join schoolcalendar van on m.date::timestamp::date =
                van.date_out and van.district='vancouver'
            left join schoolcalendar van_lag1 on m.date::timestamp::date =
                van_lag1.date_out + interval '1 day'
                and van_lag1.district='vancouver'
            left join schoolcalendar van_lead1 on m.date::timestamp::date =
                van_lead1.date_out - interval '1 day'
                and van_lead1.district='vancouver'
            left join schoolcalendar van_lag2 on m.date::timestamp::date =
                van_lag2.date_out + interval '2 day'
                and van_lag2.district='vancouver'
            left join schoolcalendar van_lead2 on m.date::timestamp::date =
                van_lead2.date_out - interval '2 day'
                and van_lead2.district='vancouver'
            left join schoolcalendar van_lag3 on m.date::timestamp::date =
                van_lag3.date_out + interval '3 day'
                and van_lag3.district='vancouver'
            left join schoolcalendar van_lead3 on m.date::timestamp::date =
                van_lead3.date_out - interval '3 day'
                and van_lead3.district='vancouver'
        where
            crossing_id = {0}
            and munger_id = {1}
            and (minute = 0 or minute = 30)
            and is_waittime = true
            and m.date >= '{2}'
        order by m.date;
        '''

    return pd_query(query.format(crossing_id, munger_id, start_date))


def select_mungedata_simple(munger_id, crossing_id, start_date, end_date):
    '''
    Select data with date features only
    '''
    query = '''
            select
                m.date,
                metric as waittime,
                year,
                month,
                week,
                dayofweek,
                minofday
            from mungedata m
            join datefeatures d on m.date = d.date
            where
                crossing_id = {0}
                and munger_id = {1}
                and (minute = 0 or minute = 30)
                and is_waittime = true
                and m.date >= '{2}'
                and m.date < '{3}'
            order by m.date;
            '''

    return pd_query(query.format(crossing_id, munger_id, start_date, end_date))


def select_crossingdata(crossing_id, start_date):
    query = '''
            select
                c.date,
                waittime,
                volume,
                year,
                month,
                week,
                dayofweek,
                minofday
            from crossingdata c
            join datefeatures d on c.date = d.date
        where
            crossing_id = {0}
            and (minute = 0 or minute = 30)
            and c.date >= '{1}'
        order by c.date;
        '''

    return pd_query(query.format(crossing_id, start_date))


def select_features(start_date, end_date):
    query = '''
            select
                d.date,
                year,
                month,
                week,
                dayofweek,
                minofday,
                w.temp_max,
                w.temp_mean,
                w.temp_min,
                w.viz_max,
                w.wind_max,
                w.precip,
                w.rain,
                w.snow,
                w.fog,
                w.thunderstorm,
                wp1.temp_max as temp_max_p1,
                wp1.temp_mean as temp_mean_p1,
                wp1.temp_min as temp_min_p1,
                wp1.precip as precip_p1,
                wp1.rain as rain_p1,
                wp1.snow as snow_p1,
                wp1.thunderstorm as thunderstorm_p1,
                wp2.temp_max as temp_max_p2,
                wp2.temp_mean as temp_mean_p2,
                wp2.temp_min as temp_min_p2,
                wp2.precip as precip_p2,
                wp2.rain as rain_p2,
                wp2.snow as snow_p2,
                wp2.thunderstorm as thunderstorm_p2,
                wp3.temp_max as temp_max_p3,
                wp3.temp_mean as temp_mean_p3,
                wp3.temp_min as temp_min_p3,
                wp3.precip as precip_p3,
                wp3.rain as rain_p3,
                wp3.snow as snow_p3,
                wp3.thunderstorm as thunderstorm_p3,
                wm1.temp_max as temp_max_m1,
                wm1.temp_mean as temp_mean_m1,
                wm1.temp_min as temp_min_m1,
                wm1.precip as precip_m1,
                wm1.rain as rain_m1,
                wm1.snow as snow_m1,
                wm1.thunderstorm as thunderstorm_m1,
                wm2.temp_max as temp_max_m2,
                wm2.temp_mean as temp_mean_m2,
                wm2.temp_min as temp_min_m2,
                wm2.precip as precip_m2,
                wm2.rain as rain_m2,
                wm2.snow as snow_m2,
                wm2.thunderstorm as thunderstorm_m2,
                s.event,
                s_lead1.event as event_lead1,
                s_lag1.event as event_lag1,
                s_lead2.event as event_lead2,
                s_lag2.event as event_lag2,
                s_lead3.event as event_lead3,
                s_lag3.event as event_lag3,
                s_lead4.event as event_lead4,
                s_lag4.event as event_lag4,
                1 as sea,
                1 as sea_lag1,
                1 as sea_lead1,
                1 as sea_lag2,
                1 as sea_lead2,
                1 as sea_lag3,
                1 as sea_lead3,
                1 as van,
                1 as van_lag1,
                1 as van_lead1,
                1 as van_lag2,
                1 as van_lead2,
                1 as van_lag3,
                1 as van_lead3
            from datefeatures d
            left join publicholiday h on d.date::timestamp::date = h.date
            left join weather w on d.date::timestamp::date = w.date
            left join weather wp1 on d.date::timestamp::date =
                wp1.date - interval '1 day'
            left join weather wp2 on d.date::timestamp::date =
                wp2.date - interval '2 day'
            left join weather wp3 on d.date::timestamp::date =
                wp3.date - interval '3 day'
            left join weather wm1 on d.date::timestamp::date =
                wm1.date + interval '1 day'
            left join weather wm2 on d.date::timestamp::date =
                wm2.date + interval '2 day'
            left join specialdates s on d.date::timestamp::date = s.date
            left join specialdates s_lead1 on d.date::timestamp::date =
                s_lead1.date - interval '1 day'
            left join specialdates s_lag1 on d.date::timestamp::date =
                s_lag1.date + interval '1 day'
            left join specialdates s_lead2 on d.date::timestamp::date =
                s_lead2.date - interval '2 day'
            left join specialdates s_lag2 on d.date::timestamp::date =
                s_lag2.date + interval '2 day'
            left join specialdates s_lead3 on d.date::timestamp::date =
                s_lead3.date - interval '3 day'
            left join specialdates s_lag3 on d.date::timestamp::date =
                s_lag3.date + interval '3 day'
            left join specialdates s_lead4 on d.date::timestamp::date =
                s_lead4.date - interval '4 day'
            left join specialdates s_lag4 on d.date::timestamp::date =
                s_lag4.date + interval '4 day'
            left join schoolcalendar sea on d.date::timestamp::date =
                sea.date_out and sea.district='seattle'
            left join schoolcalendar sea_lag1 on d.date::timestamp::date =
                sea_lag1.date_out + interval '1 day'
                and sea_lag1.district='seattle'
            left join schoolcalendar sea_lead1 on d.date::timestamp::date =
                sea_lead1.date_out - interval '1 day'
                and sea_lead1.district='seattle'
            left join schoolcalendar sea_lag2 on d.date::timestamp::date =
                sea_lag2.date_out + interval '2 day'
                and sea_lag2.district='seattle'
            left join schoolcalendar sea_lead2 on d.date::timestamp::date =
                sea_lead2.date_out - interval '2 day'
                and sea_lead2.district='seattle'
            left join schoolcalendar sea_lag3 on d.date::timestamp::date =
                sea_lag3.date_out + interval '3 day'
                and sea_lag3.district='seattle'
            left join schoolcalendar sea_lead3 on d.date::timestamp::date =
                sea_lead3.date_out - interval '3 day'
                and sea_lead3.district='seattle'
            left join schoolcalendar van on d.date::timestamp::date =
                van.date_out and van.district='vancouver'
            left join schoolcalendar van_lag1 on d.date::timestamp::date =
                van_lag1.date_out + interval '1 day'
                and van_lag1.district='vancouver'
            left join schoolcalendar van_lead1 on d.date::timestamp::date =
                van_lead1.date_out - interval '1 day'
                and van_lead1.district='vancouver'
            left join schoolcalendar van_lag2 on d.date::timestamp::date =
                van_lag2.date_out + interval '2 day'
                and van_lag2.district='vancouver'
            left join schoolcalendar van_lead2 on d.date::timestamp::date =
                van_lead2.date_out - interval '2 day'
                and van_lead2.district='vancouver'
            left join schoolcalendar van_lag3 on d.date::timestamp::date =
                van_lag3.date_out + interval '3 day'
                and van_lag3.district='vancouver'
            left join schoolcalendar van_lead3 on d.date::timestamp::date =
                van_lead3.date_out - interval '3 day'
                and van_lead3.district='vancouver'
        where
            d.date >= '{0}' and d.date < '{1}'
            and (minute = 0 or minute = 30)
        order by d.date;
        '''

    return pd_query(query.format(start_date, end_date)).set_index('date')


def select_features_simple(start_date, end_date):
    '''
    Select date features only
    '''
    query = '''
            select
                d.date,
                year,
                month,
                week,
                dayofweek,
                minofday
            from datefeatures d
            where
                d.date >= '{0}' and d.date < '{1}'
                and (minute = 0 or minute = 30)
            order by d.date;
            '''

    return pd_query(query.format(start_date, end_date)).set_index('date')


def select_predictions(munger_id, model_version, crossing_id,
                       start_date, end_date):
    '''
    Select date features only
    '''
    query = '''
            select
                date,
                waittime
            from predictions
            where
                munger_id = {0}
                and model_version = '{1}'
                and crossing_id = {2}
                and date >= '{3}' and date < '{4}'
            order by date;
            '''

    return pd_query(query.format(munger_id, model_version, crossing_id,
                                 start_date, end_date)).set_index('date')


def insert_predictions(model_id, munger_id, crossing_id, dates, waittime):
    '''
    Insert data into predictions table
    '''
    query = '''
            insert into predictions
            (model_version, munger_id, crossing_id, date, waittime)
            values('{0}', {1}, {2}, '{3}', {4})
            '''

    with PgDB() as db:
        for date, val in zip(dates, waittime):
            db.cur.execute(query.format(model_id, munger_id, crossing_id,
                                        date, val))
        db.conn.commit()
