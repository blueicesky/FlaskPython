from gevent.wsgi import WSGIServer
from app import app
app.debug = True
WSGIServer(('', 5000), app).serve_forever()