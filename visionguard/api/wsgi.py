"""gunicorn -b 0.0.0.0:7860 visionguard.api.wsgi:app"""

from visionguard.api import create_api

app = create_api()
