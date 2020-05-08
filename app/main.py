# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_cors import CORS
from models.model import Model2

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
CORS(app)
model = Model2()

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/x-icon')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    print("upload")
    # ファイルがなかった場合の処理
    if 'file' in request.files:
        return "ファイルがありません"

    # データの取り出し
    file = request.files['csv_file']
    print("file")

    # ファイルのチェック
    if file and allwed_file(file.filename):
        return model.predict(file)

    print("unknown error!")
    return "unknown error!"

def allwed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    app.run(debug=True)