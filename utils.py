#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import pandas as pd


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


def get_col_corr(df):
    corr_df = df.corr().abs().unstack().sort_values(ascending=False).reset_index()
    corr_df = corr_df.loc[corr_df.level_0 != corr_df.level_1]
    corr_df2 = pd.DataFrame([sorted(i) for i in corr_df[['level_0', 'level_1']].values])
    corr_df2['cor'] = corr_df[0].values
    corr_df2.columns = ['col_1', 'col_2', 'corr']
    return corr_df2.drop_duplicates()
