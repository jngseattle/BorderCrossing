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
        flash('Date="%s", location=%s' %
              (form.date.data, str(form.location.data)))
        g.date = form.date.data
        g.location = form.location.data
        return redirect('/predict')

    return render_template('index.html',
                           form=form)


# chart page
@app.route('/predict')
def prediction_page():

    date = g.date
    location = g.location

    # v1 will have a single direction and lane
    direction = 'Southbound'
    lane = 'Cars'

    delays = daily_prediction(date, location, direction, lane)

    return render_template('chart.html',
                            date=date,
                            location=location,
                            direction=direction,
                            lane=lane,
                            delays=delays)


# chart test
def create_linechart():
    x = range(1, 10)
    y = range(1, 10)

    hover = HoverTool(tooltips=[("(x, y)", "($x, $y)")])

    p = figure(plot_width=800, plot_height=400, tools=[hover])

    p.line(x, y)

    return p


@app.route('/chart')
def test_chart():
    chart = create_linechart()
    script, div = embed.components(chart)
    return render_template('chart.html', script=script, div=div)


if __name__ == '__main__':
    app.run(debug=True)
