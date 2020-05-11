from flask import Flask
from flask_cors import CORS

def create_app(app_name='cprofiler'):
    app = Flask(app_name)
    app.config.from_object('config.BaseConfig')

    cors = CORS(app, resources={r'/api/*': {'origins': '*'}})

    from main import api
    app.register_blueprint(api, url_prefix="/api")

    return app