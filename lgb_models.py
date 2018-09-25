#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from utils import get_data

warnings.filterwarnings('ignore')


def lgb_model(train_data, test_data, parms, n_flods=5):
    columns = train_data.columns
    remove_fields = ["instance_id", "click"]
    features_fields = [column for column in columns if column not in remove_fields]

    train_features = train_data[features_fields]
    train_labels = train_data["click"]
    test_features = test_data[features_fields]

    clf = lgb.LGBMClassifier(**parms)
    kfloder = StratifiedKFold(n_splits=n_flods, shuffle=True, random_state=2018)
    kflod = kfloder.split(train_features, train_labels)

    preds_list = list()
    for train_index, test_index in kflod:
        k_x_train = train_features.loc[train_index]
        k_y_train = train_labels.loc[train_index]
        k_x_test = train_features.loc[test_index]
        k_y_test = train_labels.loc[test_index]

        lgb_clf = clf.fit(k_x_train, k_y_train,
                          eval_names=["train", "valid"],
                          eval_metric="logloss",
                          eval_set=[(k_x_train, k_y_train),
                                    (k_x_test, k_y_test)],
                          early_stopping_rounds=100,
                          verbose=False)

        preds = lgb_clf.predict_proba(test_features, num_iteration=lgb_clf.best_iteration_)[:, 1]

        preds_list.append(preds)

    preds_columns = ["preds_{id}".format(id=i) for i in range(n_flods)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_df = preds_df.copy()
    preds_df["mean"] = preds_df.mean(axis=1)

    predictions = pd.DataFrame({"instance_id": test_data["instance_id"],
                                "predicted_score": preds_df["mean"]})

    predictions.to_csv("predict.csv", index=False)


def offline_tests(data_df, parms, n_flods=5):
    columns = data_df.columns
    remove_fields = ["instance_id", "click"]
    features_fields = [column for column in columns if column not in remove_fields]

    train_data = data_df[data_df["times_week_6"] == 0]
    test_data = data_df[data_df["times_week_6"] == 1]

    train_data.reset_index(inplace=True)
    test_data.reset_index(inplace=True)

    train_features = train_data[features_fields]
    train_labels = train_data["click"]
    test_features = test_data[features_fields]
    test_labels = test_data["click"]

    clf = lgb.LGBMClassifier(**parms)
    kfloder = StratifiedKFold(n_splits=n_flods, shuffle=True, random_state=2018)
    kflod = kfloder.split(train_features, train_labels)

    preds_list = list()

    for train_index, test_index in kflod:
        k_x_train = train_features.loc[train_index]
        k_y_train = train_labels.loc[train_index]
        k_x_test = train_features.loc[test_index]
        k_y_test = train_labels.loc[test_index]

        lgb_clf = clf.fit(k_x_train, k_y_train,
                          eval_names=["train", "valid"],
                          eval_metric="logloss",
                          eval_set=[(k_x_train, k_y_train),
                                    (k_x_test, k_y_test)],
                          early_stopping_rounds=100,
                          verbose=False)

        preds = lgb_clf.predict_proba(test_features, num_iteration=lgb_clf.best_iteration_)[:, 1]
        preds_list.append(preds)

    preds_columns = ["preds_{id}".format(id=i) for i in range(n_flods)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_df = preds_df.copy()
    preds_df["mean"] = preds_df.mean(axis=1)

    predictions = pd.DataFrame({"instance_id": test_data["instance_id"],
                                "y_true": test_labels,
                                "y_pred": preds_df["mean"]})

    predictions.to_csv("offline_predict.csv", index=False)
    logloss = log_loss(predictions["y_true"], predictions["y_pred"])
    print(logloss)


if __name__ == "__main__":
    lgb_parms = {
        "boosting_type": "gbdt",
        "num_leaves": 128,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "max_bin": 425,
        "subsample_for_bin": 50000,
        "objective": 'binary',
        "min_split_gain": 0,
        "min_child_weight": 5,
        "min_child_samples": 10,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 1,
        "reg_alpha": 3,
        "reg_lambda": 5,
        "seed": 2018,
        "n_jobs": 5,
        "verbose": 0,
        "silent": True
    }

    train_df = get_data(name="train")
    test_df = get_data(name="test")
    offline_tests(train_df, lgb_parms)
    lgb_model(train_df, test_df, lgb_parms)
