
# Debugging
from werkzeug.debug import DebuggedApplication
from main import app
app = DebuggedApplication(app, evalex=True)
