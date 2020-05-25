from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager

def create_app(app_name='cprofiler'):
    app = Flask(app_name, static_folder=None)
    app.config.from_object('config.BaseConfig')
    # Setup the Flask-JWT-Extended extension
    app.config['JWT_SECRET_KEY'] = 'dtc'
    jwt = JWTManager(app)

    cors = CORS(app, resources={r'/cprofiler/*': {'origins': '*'}})

    from main import cprofiler
    app.register_blueprint(cprofiler, url_prefix="/cprofiler")

    return app