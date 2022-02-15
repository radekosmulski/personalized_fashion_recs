import pandas as pd
import datetime
from average_precision import apk
import numpy as np


customers = pd.read_csv('data/customers.csv')
sample_sub = pd.read_csv('data/sample_submission.csv')
validation_set = pd.read_pickle('data/validation_set.pkl')

transactions = pd.read_csv('data/transactions_train.csv', dtype={'article_id': str})
transactions.t_dat = pd.to_datetime(transactions.t_dat)
most_recent_transaction = transactions.t_dat.max()
last_7_days = transactions.t_dat > most_recent_transaction - datetime.timedelta(days=7)
transactions_without_last_7_days = transactions[~last_7_days]

def calculate_apk(list_of_preds, list_of_gts):
    apks = []
    for preds, gt in zip(list_of_preds, list_of_gts):
        apks.append(apk(preds, gt))
    return np.mean(apks)

def eval_sub(sub_csv, skip_cust_with_no_purchases=True):
    sub=pd.read_csv(sub_csv)
    validation_set=pd.read_csv('data/subs/validation_set.csv')

    apks = []

    no_purchases_pattern = ['0']*12
    for pred, gt in zip(sub.prediction.str.split(), validation_set.prediction.str.split()):
        if skip_cust_with_no_purchases & (gt == no_purchases_pattern): continue
        apks.append(apk(pred, gt))
    return np.mean(apks)
