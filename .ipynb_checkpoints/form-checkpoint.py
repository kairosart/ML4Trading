from flask_wtf import FlaskForm
from wtforms import TextField, IntegerField, DecimalField, SubmitField
from wtforms import validators, ValidationError


# Form class
class StartValuesForm(FlaskForm):
    start_val = IntegerField('Initial Capital', [validators.Required("Please enter a value.")])
    symbol = TextField('Stock Symbol', [validators.Required("Please enter stock symbol as AMZN.")])
    commission = DecimalField('Commision')
    impact = DecimalField('Impact')
    num_shares = IntegerField('Shares number', [validators.Required("Please enter number of shares.")])
    submit = SubmitField("Send") 