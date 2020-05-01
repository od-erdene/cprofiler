import argparse
import numpy as np
import pandas as pd


class PrepareData:

    def __init__(self, data_path):

        self.data = pd.read_csv(data_path, low_memory=False)
        
        self.keyword_columns = [column for column in self.data.columns if 'k_' in column]
        self.time_columns = [column for column in self.data.columns if 'a_' in column]
        self.device_columns = ['os', 'browser', 'device']

        all_feature_colums = self.keyword_columns + self.time_columns + self.device_columns
        
        self.target_columns = [column for column in self.data.columns if not column in all_feature_colums]

    def return_features(self, feat_cols, target_cols):

        datas = self.data[feat_cols].values
        targets = self.data[target_cols].values

        return datas, targets


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, default='')
    args = parser.parse_args()

    data_preparer = PrepareData(args.data_path)

    data, target = data_preparer.return_features(data_preparer.keyword_columns, 'SEX')
    
    
    
