from flask import Flask
from flask_cors import CORS

def create_app(app_name='cprofiler'):
    app = Flask(app_name, static_folder=None)
    app.config.from_object('config.BaseConfig')

    cors = CORS(app, resources={r'/cprofiler/*': {'origins': '*'}})

    from main import cprofiler
    app.register_blueprint(cprofiler, url_prefix="/cprofiler")

    return app