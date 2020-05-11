import os
import joblib
import logging
import pandas as pd

from collections import Counter

logger = logging.getLogger(__name__)

class LGBMModel(object):
    def __init__(self):
        logger.debug("init start...")
        dir_name = os.path.dirname(os.path.abspath(__file__))
        # loading encoders
        self._os_encoder = joblib.load(os.path.join(dir_name, "params/os_encoder.pkl"))
        self._browser_encoder = joblib.load(os.path.join(dir_name, "params/browser_encoder.pkl"))
        self._device_encoder = joblib.load(os.path.join(dir_name, "params/device_encoder.pkl"))
        self._gender_model = joblib.load(os.path.join(dir_name, "params/gender_lightgbm.pkl"))

        logger.debug("init end.")

    def _preprocess(self, df):
        logger.debug("2. Preprocessing....")
        for i in ["D_q1","PINCOME","AREA","MARRIED","HINCOME","CHILD","SEX","PREFECTURE","JOB","STUDENT","AGEID"]:
            if i in df.columns:
                df = df.drop(i, axis=1)

        # updating encoders for new values
        for i in df["os"].values:
            if i not in self._os_encoder:
                self._os_encoder[i] = max(self._os_encoder.values())+1
        for i in df["browser"].values:
            if i not in self._browser_encoder:
                self._browser_encoder[i] = max(self._browser_encoder.values())+1
        for i in df["device"].values:
            if i not in self._device_encoder:
                self._device_encoder[i] = max(self._device_encoder.values())+1     
        # encode input data
        df["device"] = [self._device_encoder[i] for i in df["device"].values]
        df["os"] = [self._os_encoder[i] for i in df["os"].values]
        df["browser"] = [self._browser_encoder[i] for i in df["browser"].values]
        logging.debug("3. Done preprocessing!")
        return df

    def _predict_gender(self, df):
        logging.debug("4. Predicting gender....")
        preds = self._gender_model.predict(df)
        logging.debug("5. Done predicting gender!")
        return preds

    def predict(self, csv_file):
        logging.debug("1. Starting prediction....")
        df_clean = self._preprocess(pd.read_csv(csv_file))
        gender_preds = self._predict_gender(df_clean)
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
        logging.debug("6. Done predicting everything!")
        # print(result)
        return result