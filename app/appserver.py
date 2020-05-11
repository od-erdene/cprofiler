if __name__ == '__main__':
    from application import create_app
    app = create_app()
    app.run(host='0.0.0.0') 
    # app.run(debug=True)    