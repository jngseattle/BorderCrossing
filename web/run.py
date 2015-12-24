from flask import Flask, render_template, request, g, flash, redirect
from forms import BorderForm
from predict import daily_prediction
from bokeh import embed
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import HoverTool
import datetime
app = Flask(__name__)
app.config.from_object('config')


# home page
@app.route('/', methods=['POST', 'GET'])
def index():
    form = BorderForm()
    if form.validate_on_submit():
        # flash('Date="%s", location=%s' %
        #       (form.date.data, str(form.location.data)))

        url = '/predict/{0}/{1}'.format(form.date.data, form.location.data)
        return redirect(url)

    return render_template('index.html',
                           form=form)


# chart page
@app.route('/predict/<date>/<location>')
@app.route('/predict/<date>/<location>/<direction>')
@app.route('/predict/<date>/<location>/<direction>/<lane>')
def prediction_page(date, location, direction='Southbound', lane='Car'):
    # Get predictions
    start = datetime.datetime.strptime(date, '%Y-%m-%d')
    results = daily_prediction(start, location, direction, lane)

    # Create labels by hour
    labels = []
    time = datetime.datetime(2008, 4, 11)      # Actual date is immaterial
    end = time + datetime.timedelta(hours=24)
    while time < end:
        if time.minute == 0 and time.hour % 2 == 0:
            labels.append(str(time.time())[:-3])
        else:
            labels.append("")
        time += datetime.timedelta(minutes=30)

    # Date formatting for calendar control
    date_formatted = datetime.datetime \
        .strptime(date, '%Y-%m-%d').strftime('%m/%d/%Y')

    # Dates for arrow buttons
    next_week = start + datetime.timedelta(days=7)
    next_week = next_week.strftime('%Y-%m-%d')
    last_week = start - datetime.timedelta(days=7)
    last_week = last_week.strftime('%Y-%m-%d')
    tomorrow = start + datetime.timedelta(days=1)
    tomorrow = tomorrow.strftime('%Y-%m-%d')
    yesterday = start - datetime.timedelta(days=1)
    yesterday = yesterday.strftime('%Y-%m-%d')

    return render_template('chart.html',
                           date=date,
                           default_date=date_formatted,
                           dow=start.strftime("%A"),
                           location=location,
                           direction=direction,
                           lane=lane,
                           predict=list(results.predict),
                           baseline=list(results.baseline),
                           labels=labels,
                           next_week=next_week,
                           last_week=last_week,
                           tomorrow=tomorrow,
                           yesterday=yesterday
                           )


@app.route('/chart')
def chart():
    labels = ["January","February","March","April","May","June","July","August"]
    values = [10,9,8,7,6,4,7,8]
    return render_template('chart.html', values=values, labels=labels)


if __name__ == '__main__':
    app.run(debug=True)
