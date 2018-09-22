#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


def get_data(name, sample=1.0):
    etl_path = os.path.join("data", "EtlData")
    if name == "train":
        data_name = os.path.join(etl_path, "train_features.csv")
    elif name == "test":
        data_name = os.path.join(etl_path, "test_features.csv")
    else:
        raise ValueError("name must be `train` or `test`!")
    df = pd.read_csv(data_name, header=0)
    if sample != 1.0:
        df = df.sample(frac=sample, random_state=2018)
    return df


def offline_tests(data_df, parms, n_flods=5):
    columns = data_df.columns
    remove_fields = ["instance_id", "click"]
    features_fields = [column for column in columns if column not in remove_fields or "user_tags" not in column]

    train_data = data_df[data_df["times_week_6"] == 0]
    test_data = data_df[data_df["times_week_6"] == 1]

    train_features = train_data[features_fields]
    train_labels = train_data["click"]
    test_features = test_data[features_fields]
    test_labels = test_data["click"]

    clf = lgb.LGBMClassifier(**parms)
    kfloder = StratifiedKFold(n_splits=n_flods, shuffle=True, random_state=2018)
    kflod = kfloder.split(train_features, train_labels)

    base_loss = list()
    loss = 0.0
    preds_list = list()
    for train_index, test_index in kflod:
        lgb_clf = clf.fit(train_features[train_index], train_labels[train_index],
                          eval_names=["train", "valid"],
                          eval_metric="logloss",
                          eval_set=[(train_features[train_index], train_labels[train_index]),
                                    (train_features[test_index], train_labels[test_index])],
                          early_stopping_rounds=100)
        _logloss = lgb_clf.best_score_['valid']['binary_logloss']
        print(_logloss)
        base_loss.append(_logloss)
        loss += _logloss
        preds = lgb_clf.predict_proba(test_features, num_iteration=lgb_clf.best_iteration_)[:, 1]
        preds_list.append(preds)

    preds_columns = ["preds_{id}".format(id=i) for i in range(n_flods)]
    preds_df = pd.DataFrame(data=preds_list, columns=preds_columns)
    preds_df = preds_df.copy()
    preds_df["mean"] = preds_df.mean(axis=1)

    predictions = pd.DataFrame({"instance_id": test_data["instance_id"],
                                "y_true": test_labels,
                                "y_pred": preds_df["mean"]})

    predictions.to_csv("predict.csv", index=False)
    logloss = log_loss(predictions["y_true"], predictions["y_pred"])

    return logloss


if __name__ == "__main__":
    lgb_parms = {
        "boosting_type": "gbdt",
        "num_leaves": 48,
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
        "silent": True
    }

    train_df = get_data(name="train")
    print(offline_tests(train_df, lgb_parms))