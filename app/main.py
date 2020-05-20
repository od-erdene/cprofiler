# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, send_from_directory, jsonify, Blueprint
from flask_jwt_extended import jwt_required, create_access_token, get_jwt_identity

from models.model import LGBMModel

import os
import sys
import datetime
import logging


# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['csv'])
model = LGBMModel()
cprofiler = Blueprint('cprofiler', __name__, template_folder='templates', static_folder='static', static_url_path='static')

logger = None
def init_log():
    global logger
    json_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(os.path.dirname(os.path.abspath(__file__)),
        "logs/log_{}".format(datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))))
    logger = logging.getLogger(__name__)

@cprofiler.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(cprofiler.root_path, 'static'), 'favicon.ico', mimetype='image/x-icon')

# Provide a method to create access tokens. The create_access_token()
# function is used to actually generate the token, and you can return
# it to the caller however you choose.
@cprofiler.route('/login', methods=['POST'])
def login():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    username = request.json.get('username', None)
    password = request.json.get('password', None)
    if not username:
        return jsonify({"msg": "Missing username parameter"}), 400
    if not password:
        return jsonify({"msg": "Missing password parameter"}), 400

    if username != 'dtc' or password != 'P@ssw0rd!':
        return jsonify({"msg": "Bad username or password"}), 401

    # Identity can be any data that is json serializable
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token), 200

@cprofiler.route('/')
def index():
    if get_jwt_identity() is None:
        return render_template("login.html")
    return render_template("index.html")

@cprofiler.route('/upload', methods=['POST'])
@jwt_required
def upload():
    if get_jwt_identity() is None:
        return render_template("login.html")
    # ファイルがなかった場合の処理
    if 'file' in request.files:
        return jsonify(status="ファイルがありません")

    # データの取り出し
    file = request.files['csv_file']
    
    # ファイルのチェック
    if file and allwed_file(file.filename):
        response = model.predict(file)

        return jsonify(status="ok", val=response)
    
    return jsonify(status="unknown error!")

def allwed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS