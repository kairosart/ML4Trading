from flask import Flask, session
app = Flask(__name__)
# Add these extra two lines
app.secret_key = 's3cr3t'
app.config['SESSION_TYPE'] = 'filesystem'

@app.route('/<int:x>')
def count(x):
    # See if we already instantiated the list
    s = session.get('sum', None)
    if not s:
        # If it's not there, add our first item.
        session['sum'] = x
    else:
        # If it's there, add the current number
        session['sum']+=x
        # Display current count
    return str(session['sum'])


@app.route('/')
def home():
    return 'Open this page and go to /5 or some other number'

if __name__ == '__main__':
    app.run(host='0.0.0.0')
