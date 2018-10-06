#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB

from utils import get_data

warnings.filterwarnings('ignore')


class Stacking(object):

    @staticmethod
    def _get_kfold(train_features, train_labels, n_folds=5):
        kfolder = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2018)
        kfold = kfolder.split(train_features, train_labels)
        return kfold

    @staticmethod
    def _get_num(num):
        if num > 0.5:
            return 1
        return 0

    def _stcaking_lr(self, train_features, train_labels, test_features):
        train_features = train_features.copy()
        train_labels = train_labels.copy()
        test_features = test_features.copy()

        clf = LogisticRegression(random_state=2018, C=8.0)

        train_features = train_features.fillna(-1)
        test_features = test_features.fillna(-1)

        kfold = self._get_kfold(train_features, train_labels)
        preds_train_list = list()
        preds_test_list = list()
        for train_index, test_index in kfold:
            k_x_train = train_features.loc[train_index]
            k_y_train = train_labels.loc[train_index]
            k_x_test = train_features.loc[test_index]
            k_y_test = train_labels.loc[test_index]

            clf.fit(k_x_train, k_y_train)

            preds_valid = clf.predict_proba(k_x_test)
            logloss = log_loss(k_y_test, preds_valid)
            print("lr model logloss is {logloss}".format(logloss=logloss))

            preds_train = clf.predict_proba(train_features)[:, 1]
            preds_train_list.append(preds_train)

            preds_test = clf.predict_proba(test_features)[:, 1]
            preds_test_list.append(preds_test)

        length = len(preds_train_list)
        preds_columns = ["preds_{id}".format(id=i) for i in range(length)]

        preds_train_df = pd.DataFrame(data=preds_train_list)
        preds_train_df = preds_train_df.T
        preds_train_df.columns = preds_columns
        preds_train_df = preds_train_df.copy()
        preds_train_df["stacking_lr"] = preds_train_df.mean(axis=1)

        preds_test_df = pd.DataFrame(data=preds_test_list)
        preds_test_df = preds_test_df.T
        preds_test_df.columns = preds_columns
        preds_test_df = preds_test_df.copy()
        preds_test_df["stacking_lr"] = preds_test_df.mean(axis=1)

        stacking_lr_df = pd.concat([preds_train_df["stacking_lr"], preds_test_df["stacking_lr"]], axis=0,
                                   ignore_index=True)
        stacking_lr_df = stacking_lr_df.apply(self._get_num)
        return stacking_lr_df

    def _stacking_bnb(self, train_features, train_labels, test_features):
        train_features = train_features.copy()
        train_labels = train_labels.copy()
        test_features = test_features.copy()

        clf = BernoulliNB()

        train_features = train_features.fillna(-1)
        test_features = test_features.fillna(-1)

        kfold = self._get_kfold(train_features, train_labels)
        preds_train_list = list()
        preds_test_list = list()
        for train_index, test_index in kfold:
            k_x_train = train_features.loc[train_index]
            k_y_train = train_labels.loc[train_index]
            k_x_test = train_features.loc[test_index]
            k_y_test = train_labels.loc[test_index]

            clf.fit(k_x_train, k_y_train)

            preds_valid = clf.predict_proba(k_x_test)
            logloss = log_loss(k_y_test, preds_valid)
            print("bnb model logloss is {logloss}".format(logloss=logloss))

            preds_train = clf.predict_proba(train_features)[:, 1]
            preds_train_list.append(preds_train)

            preds_test = clf.predict_proba(test_features)[:, 1]
            preds_test_list.append(preds_test)

        length = len(preds_train_list)
        preds_columns = ["preds_{id}".format(id=i) for i in range(length)]

        preds_train_df = pd.DataFrame(data=preds_train_list)
        preds_train_df = preds_train_df.T
        preds_train_df.columns = preds_columns
        preds_train_df = preds_train_df.copy()
        preds_train_df["stacking_bnb"] = preds_train_df.mean(axis=1)

        preds_test_df = pd.DataFrame(data=preds_test_list)
        preds_test_df = preds_test_df.T
        preds_test_df.columns = preds_columns
        preds_test_df = preds_test_df.copy()
        preds_test_df["stacking_bnb"] = preds_test_df.mean(axis=1)

        stacking_bnb_df = pd.concat([preds_train_df["stacking_bnb"], preds_test_df["stacking_bnb"]], axis=0,
                                    ignore_index=True)

        stacking_bnb_df = stacking_bnb_df.apply(self._get_num)

        return stacking_bnb_df

    def _stacking_rf(self, train_features, train_labels, test_features):
        train_features = train_features.copy()
        train_labels = train_labels.copy()
        test_features = test_features.copy()

        clf = RandomForestClassifier()

        train_features = train_features.fillna(-1)
        test_features = test_features.fillna(-1)

        kfold = self._get_kfold(train_features, train_labels)
        preds_train_list = list()
        preds_test_list = list()
        for train_index, test_index in kfold:
            k_x_train = train_features.loc[train_index]
            k_y_train = train_labels.loc[train_index]
            k_x_test = train_features.loc[test_index]
            k_y_test = train_labels.loc[test_index]

            clf.fit(k_x_train, k_y_train)

            preds_valid = clf.predict_proba(k_x_test)
            logloss = log_loss(k_y_test, preds_valid)
            print("rf model logloss is {logloss}".format(logloss=logloss))

            preds_train = clf.predict_proba(train_features)[:, 1]
            preds_train_list.append(preds_train)

            preds_test = clf.predict_proba(test_features)[:, 1]
            preds_test_list.append(preds_test)

        length = len(preds_train_list)
        preds_columns = ["preds_{id}".format(id=i) for i in range(length)]

        preds_train_df = pd.DataFrame(data=preds_train_list)
        preds_train_df = preds_train_df.T
        preds_train_df.columns = preds_columns
        preds_train_df = preds_train_df.copy()
        preds_train_df["stacking_rf"] = preds_train_df.mean(axis=1)

        preds_test_df = pd.DataFrame(data=preds_test_list)
        preds_test_df = preds_test_df.T
        preds_test_df.columns = preds_columns
        preds_test_df = preds_test_df.copy()
        preds_test_df["stacking_rf"] = preds_test_df.mean(axis=1)

        stacking_rf_df = pd.concat([preds_train_df["stacking_rf"], preds_test_df["stacking_rf"]], axis=0,
                                   ignore_index=True)

        stacking_rf_df = stacking_rf_df.apply(self._get_num)

        return stacking_rf_df

    def stacking(self):
        train_data = get_data(name="train", qcut=False, filter_=False)
        test_data = get_data(name="test", qcut=False, filter_=False)

        columns = train_data.columns
        remove_fields = ["instance_id", "click"]
        features_fields = [column for column in columns if column not in remove_fields]

        train_data_length = train_data.shape[0]
        df = pd.concat([train_data, test_data], axis=0, ignore_index=True, sort=False)

        train_features = train_data[features_fields]
        train_labels = train_data["click"]
        test_features = test_data[features_fields]

        stacking_lr = self._stcaking_lr(train_features, train_labels, test_features)
        stacking_bnb = self._stacking_bnb(train_features, train_labels, test_features)
        stacking_rf = self._stacking_rf(train_features, train_labels, test_features)

        df = pd.concat([df, stacking_lr, stacking_bnb, stacking_rf], axis=1)

        train_df = df[:train_data_length]
        test_df = df[train_data_length:]

        train_df["click"] = train_df["click"].apply(int)
        test_df = test_df.drop(columns=["click"])

        etl_path = os.path.join("data", "EtlData")
        train_name = os.path.join(etl_path, "stacking_train_features.csv")
        test_name = os.path.join(etl_path, "stacking_test_features.csv")

        train_df.to_csv(train_name, index=False)
        test_df.to_csv(test_name, index=False)


if __name__ == "__main__":
    Stacking().stacking()
