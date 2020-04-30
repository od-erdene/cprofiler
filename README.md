# cprofiler

## Env
- Python
```
cprofiler %  python -V
Python 3.6.7
```
- pip (virtualenv)
```
cprofiler % pip install -r app/requirements.txt
```

- run
```
cprofiler % python app/main.py 
```



















- 参考に
favicon表示
```
1. indexで表示(Jinja2を用いて表示しているよう)
<link rel="shortcut icon" href="{{ url_for('static', filename='favicon.ico') }}" type="image/x-icon">
<link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}" type="image/x-icon">
2. indexを以下の固定（flask側で表示させる）
app.add_url_rule('/favicon.ico',　redirect_to=url_for('static', filename='favicon.ico'))
または、
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/x-icon')


```