# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, send_from_directory, jsonify, Blueprint

from models.model import LGBMModel

import os
import sys
import datetime
import logging

from appserver import app

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['csv'])
model = LGBMModel()
cprofiler = Blueprint('cprofiler', __name__)

logger = None
def init_log():
    global logger
    json_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(os.path.dirname(os.path.abspath(__file__)),
        "logs/log_{}".format(datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))))
    logger = logging.getLogger(__name__)

@cprofiler.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/x-icon')

@cprofiler.route('/')
def index():
    return render_template("index.html")

@cprofiler.route('/upload', methods=['POST'])
def upload():
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