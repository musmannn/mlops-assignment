"""Entry points.

Production (Docker):  gunicorn -c gunicorn.conf.py visionguard.web.wsgi:app
Local, debug OFF:     python -m visionguard.web.wsgi
"""

import os

from visionguard.web import create_app

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

app = create_app()

if __name__ == "__main__":
    # debug=False: the Werkzeug debugger would let anyone who can trigger an error run
    # Python code on the server, and the reloader would load the model twice.
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), debug=False)
