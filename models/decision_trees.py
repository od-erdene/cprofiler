import argparse
import numpy as np
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score
from make_data import PrepareData


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')
    parser.add_argument('--target_column', type=str, default='SEX')
    parser.add_argument('--data_cols', type=str, default = 'keyword')
    
    args = parser.parse_args()
    return args


def get_data(args, data_columns):

    train_prepare = PrepareData(args.train_path)
    train_data, train_label = train_prepare.return_features(data_columns,
                                                            args.target_column)

    test_prepare = PrepareData(args.test_path)
    test_data, test_label = test_prepare.return_features(data_columns,
                                                         args.target_column)
    
    if args.target_column in ['SEX', 'CHILD', 'PREFECTURE', 'MARRIED', 'AREA']:
        train_label = train_label -1
        test_label = test_label - 1

    return train_data, train_label, test_data, test_label


def get_model():

    return LGBMClassifier()


def get_columns(args):

    test_prepare = PrepareData(args.test_path)
    dat_cols = []
    columns = args.data_cols.split(',')

    if 'keyword' in columns:
        dat_cols = dat_cols + test_prepare.keyword_columns
        
    if 'time' in columns:
        dat_cols = dat_cols + test_prepare.time_columns

    if 'device' in columns:
        dat_cols = dat_cols + test_prepare.device_columns
    
    return dat_cols
    

if __name__ == '__main__':

    args = get_args()

    data_columns = get_columns(args)
    target_data, target_label, test_data, test_label = get_data(args, data_columns)
    
    model = get_model()
    print (target_data)
    print ('xxxx ---> yyy ')
    print (args.data_cols)
    model.fit(target_data, target_label)
    print ('training is done')

    y_pred = model.predict(test_data)

    acc = accuracy_score(test_label, y_pred)
    print (acc)
