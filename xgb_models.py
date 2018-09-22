#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss


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


def fit_xgb_model(train_data, test_data, parms):
    columns = train_data.columns
    remove_fields = ["instance_id", "click"]
    features_fields = [column for column in columns if column not in remove_fields]

    train_features = train_data[features_fields]
    train_labels = train_data["click"]
    test_features = test_data[features_fields]

    dtrain = xgb.DMatrix(train_features, label=train_labels)
    dtest = xgb.DMatrix(test_features)
    bst = xgb.train(parms, dtrain)

    preds = bst.predict(dtest)
    predictions = pd.DataFrame({"instance_id": test_data["instance_id"],
                                "predicted_score": preds})
    predictions.to_csv("predict.csv", index=False)


def offline_tests(data_df, parms):
    columns = data_df.columns
    remove_fields = ["instance_id", "click"]
    features_fields = [column for column in columns if column not in remove_fields or "user_tags" not in column]

    train_data = data_df[data_df["times_week_6"] == 0]
    test_data = data_df[data_df["times_week_6"] == 1]

    train_features = train_data[features_fields]
    train_labels = train_data["click"]
    test_features = test_data[features_fields]
    test_labels = test_data["click"]

    # dtrain = xgb.DMatrix(train_features, label=train_labels)
    # dtest = xgb.DMatrix(test_features, label=test_labels)
    # bst = xgb.train(parms, dtrain)
    #
    # preds = bst.predict(dtest)
    # logloss = log_loss(test_labels, preds)
    #
    # predictions = pd.DataFrame({"instance_id": test_data["instance_id"],
    #                             "y_true": test_labels,
    #                             "y_pred": preds})
    # predictions.to_csv("predict.csv", index=False)
    # return logloss


if __name__ == "__main__":
    xgb_parms = {
        "booster": "gbtree",
        "objective": "reg:linear",
        "eta": 0.01,
        "n_estimators": 500,
        "max_depth": 10,
        "silent": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "nthread": 10
    }
    train_df = get_data(name="train")
    # test_df = get_data(name="test")

    print(offline_tests(train_df, xgb_parms))
