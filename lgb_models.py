#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils import get_data

warnings.filterwarnings('ignore')


def lgb_model(train_data, test_data, parms, n_folds=5):
    columns = train_data.columns
    remove_fields = ["instance_id", "click"]
    features_fields = [column for column in columns if column not in remove_fields]

    train_features = train_data[features_fields]
    train_labels = train_data["click"]
    test_features = test_data[features_fields]

    clf = lgb.LGBMClassifier(**parms)
    kfolder = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2018)
    kfold = kfolder.split(train_features, train_labels)

    preds_list = list()
    for train_index, test_index in kfold:
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
                          verbose=True)

        preds = lgb_clf.predict_proba(test_features, num_iteration=lgb_clf.best_iteration_)[:, 1]

        preds_list.append(preds)

    length = len(preds_list)
    preds_columns = ["preds_{id}".format(id=i) for i in range(length)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_df = preds_df.copy()
    preds_df["mean"] = preds_df.mean(axis=1)

    predictions = pd.DataFrame({"instance_id": test_data["instance_id"],
                                "predicted_score": preds_df["mean"]})

    predictions.to_csv("predict.csv", index=False)


if __name__ == "__main__":
    lgb_parms = {
        "boosting_type": "gbdt",
        "num_leaves": 1024,
        "max_depth": -1,
        "learning_rate": 0.01,
        "n_estimators": 10000,
        "max_bin": 425,
        "subsample_for_bin": 50000,
        "objective": 'binary',
        "min_split_gain": 0,
        "min_child_weight": 5,
        "min_child_samples": 10,
        "subsample": 0.8,
        "subsample_freq": 3,
        "colsample_bytree": 0.8,
        "reg_alpha": 1,
        "reg_lambda": 3,
        "seed": 2018,
        "n_jobs": 5,
        "verbose": 1,
        "silent": False
    }

    train_df = get_data(name="train", stacking=False)
    test_df = get_data(name="test", stacking=False)
    lgb_model(train_df, test_df, lgb_parms)
