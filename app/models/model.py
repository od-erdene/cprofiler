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
            "pincome":{
                "200万未満" : "5%",
                "わからない" : "9%",
                "200〜400万未満" : "16%",
                "400〜600万未満" : "15%",
                "600〜800万未満" : "11%",
                "800〜1000万未満" : "14%",
                "1000〜1200万未満" : "10%",
                "1200〜1500万未満" : "8%",
                "1500〜2000万未満" : "7%",
                "2000万円以上" : "5%"
            },
            "hincome":{
                "200万未満" : "5%",
                "わからない" : "9%",
                "200〜400万未満" : "16%",
                "400〜600万未満" : "15%",
                "600〜800万未満" : "11%",
                "800〜1000万未満" : "14%",
                "1000〜1200万未満" : "10%",
                "1200〜1500万未満" : "8%",
                "1500〜2000万未満" : "7%",
                "2000万円以上" : "5%"
            },
            "area":{
                "北海道" : "11%",
                "東北地方" : "14%",
                "関東地方" : "12%",
                "中部地方" : "13%",
                "近畿地方" : "13%",
                "中国地方" : "10%",
                "四国地方" : "13%",
                "九州地方" : "14%"
            },
            "married":{
                "未婚" : "75%",
                "既婚" : "25%"
            },
            "child":{
                "子供なし" : "37%",
                "子供あり" : "63%"
            },
            "prefecture":{
                "北海道" : "3%", 
                "群馬県" : "3%", 
                "埼玉県" : "3%", 
                "千葉県" : "3%", 
                "東京都" : "3%", 
                "神奈川県" : "3%", 
                "新潟県" : "3%", 
                "富山県" : "3%", 
                "石川県" : "3%", 
                "福井県" : "3%", 
                "山梨県" : "3%", 
                "青森県" : "3%", 
                "長野県" : "3%", 
                "岐阜県" : "3%", 
                "静岡県" : "3%", 
                "愛知県" : "3%", 
                "三重県" : "3%", 
                "滋賀県" : "3%", 
                "京都府" : "3%", 
                "大阪府" : "3%", 
                "兵庫県" : "3%", 
                "奈良県" : "3%", 
                "岩手県" : "3%", 
                "和歌山県" : "3%", 
                "鳥取県" : "3%", 
                "島根県" : "3%", 
                "岡山県" : "3%", 
                "広島県" : "3%", 
                "山口県" : "3%", 
                "徳島県" : "3%", 
                "香川県" : "3%", 
                "愛媛県" : "3%", 
                "高知県" : "3%", 
                "宮城県" : "3%", 
                "福岡県" : "3%", 
                "佐賀県" : "3%", 
                "長崎県" : "3%", 
                "熊本県" : "3%", 
                "大分県" : "3%", 
                "宮崎県" : "3%", 
                "沖縄県" : "3%", 
                "秋田県" : "3%", 
                "山形県" : "3%", 
                "福島県" : "3%", 
                "茨城県" : "3%", 
                "栃木県" : "3%", 
                "鹿児島県" : "3%"
            },
            "job":{
                "公務員" : "12%",
                "学生" : "8%",
                "その他" : "2%",
                "無職" : "3%",
                "経営者・役員" : "13%",
                "会社員(事務系)" : "12%",
                "会社員(技術系)" : "15%",
                "会社員(その他)" : "15%",
                "自営業" : "10%",
                "自由業" : "13%",
                "専業主婦(主夫)" : "5%",
                "パート・アルバイト" : "7%"
            },
            "student":{
                "小学生" : "7%", 
                "中学生" : "8%", 
                "専門学校生" : "5%", 
                "短大生" : "15%", 
                "大学生" : "13%", 
                "大学院生" : "15%", 
                "その他学生" : "20%", 
                "高校生・高専生" : "17%"
            },
            "age":{
                "12才未満" : "3%",
                "12才〜19才" : "5%",
                "20才〜24才" : "14%",
                "25才〜29才" : "13%",
                "30才〜34才" : "19%",
                "35才〜39才" : "23%",
                "40才〜44才" : "6%",
                "45才〜49才" : "5%",
                "50才〜54才" : "4%",
                "55才〜59才" : "3%",
                "60才以上" : "5%"
            }
        }
        logging.debug("6. Done predicting everything!")
        # print(result)
        return result