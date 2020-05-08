from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import joblib
import os

abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
date = datetime.today().strftime('%Y_%m_%d')

LOG_FILENAME = os.path.join(dir_name, "logs/log_{}.txt".format(date))
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

def preprocess(df):
    logging.debug("2. Preprocessing....")
    for i in ["D_q1","PINCOME","AREA","MARRIED","HINCOME","CHILD","SEX","PREFECTURE","JOB","STUDENT","AGEID"]:
        if i in df.columns:
            df = df.drop(i, axis=1)
    # loading encoders
    logging.debug("3. Encoding....")
    os_encoder = joblib.load(os.path.join(dir_name, "model_data/os_encoder.pkl"))
    browser_encoder = joblib.load(os.path.join(dir_name, "model_data/browser_encoder.pkl"))
    device_encoder = joblib.load(os.path.join(dir_name, "model_data/device_encoder.pkl"))
    # updating encoders for new values
    for i in df["os"].values:
        if i not in os_encoder:
            os_encoder[i] = max(os_encoder.values())+1
    for i in df["browser"].values:
        if i not in browser_encoder:
            browser_encoder[i] = max(browser_encoder.values())+1
    for i in df["device"].values:
        if i not in device_encoder:
            device_encoder[i] = max(device_encoder.values())+1     
    # encode input data
    df["device"] = [device_encoder[i] for i in df["device"].values]
    df["os"] = [os_encoder[i] for i in df["os"].values]
    df["browser"] = [browser_encoder[i] for i in df["browser"].values]
    logging.debug("4. Done preprocessing!")
    return df 

def predict_gender(df):
    logging.debug("5. Predicting gender....")
    model = joblib.load(os.path.join(dir_name, "model_data/gender_lightgbm.pkl"))
    preds = model.predict(df)
    logging.debug("6. Done predicting gender!")
    return preds

def predict(df):
    logging.debug("1. Starting prediction....")
    df_clean = preprocess(df)
    gender_preds = predict_gender(df_clean)
    gender_dist = Counter(gender_preds)

    result = {
        "sex":{
            "male": str(int((gender_dist[0]/(gender_dist[0]+gender_dist[1]))*100))+"%",
            "female": str(int((gender_dist[1]/(gender_dist[0]+gender_dist[1]))*100))+"%"
        },
        "pincome":{},
        "hincome":{},
        "area":{},
        "married":{},
        "child":{},
        "prefecture":{},
        "job":{},
        "student":{},
        "age":{}
    }
    logging.debug("8. Done predicting everything!")
    print(result)
    return result
