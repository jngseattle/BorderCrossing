from flask.ext.wtf import Form
from wtforms.fields import StringField, DateField
from wtforms.validators import DataRequired

class BorderForm(Form):
    date = DateField('date', validators=[DataRequired()], format='%m/%d/%Y')
    location = StringField('location')
