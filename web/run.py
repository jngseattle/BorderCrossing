from flask import Flask, render_template, request, g, flash, redirect
from forms import BorderForm
from predict import daily_prediction
from bokeh import embed
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import HoverTool
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
        # return redirect('/predict')
        return redirect(url)

    return render_template('index.html',
                           form=form)


# chart page
@app.route('/predict/<date>/<location>')
@app.route('/predict/<date>/<location>/<direction>')
@app.route('/predict/<date>/<location>/<direction>/<lane>')
def prediction_page(date, location, direction='Southbound', lane='Cars'):
    results = daily_prediction(date, location, direction, lane)

    # labels/values will be pulled from results
    labels = ["January","February","March","April","May","June","July","August"]
    values = [10,9,8,7,6,4,7,8]

    return render_template('chart.html',
                            date=date,
                            location=location,
                            direction=direction,
                            lane=lane,
                            values=values,
                            labels=labels)


@app.route('/chart')
def chart():
    labels = ["January","February","March","April","May","June","July","August"]
    values = [10,9,8,7,6,4,7,8]
    return render_template('chart.html', values=values, labels=labels)


if __name__ == '__main__':
    app.run(debug=True)
