#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

from utils import get_data

warnings.filterwarnings('ignore')

REMOVE_COLUMN = ["make", "model", "app_paid", "creative_is_js", "creative_is_voicead"]
DUMPS_COLUMN = ["province", "carrier", "devtype", "nnt", "os_name", "advert_id", "campaign_id", "creative_tp_dnf",
                "app_cate_id", "f_channel", "creative_type", "creative_is_jump", "creative_is_download",
                "creative_has_deeplink", "advert_name"]

RATE_COLUMN = ["province", "carrier", "devtype", "nnt", "os_name", "advert_id", "campaign_id", "creative_tp_dnf",
               "app_cate_id", "f_channel", "creative_type", "creative_is_jump", "creative_is_download",
               "creative_has_deeplink", "advert_name", "city", "adid", "orderid", "creative_id", "inner_slot_id",
               "app_id"]


class TagsProcessing(object):
    # user_tags字段特殊处理

    @staticmethod
    def _get_tags_dict(tags):
        tags_list = list()
        for _, item in tags.iteritems():
            try:
                tag_list = item.split(",")
                tags_list.extend(tag_list)
            except AttributeError:
                tags_list.append(str(item))

        tags_set = set(tags_list)
        tags_dict = {k: v for v, k in enumerate(tags_set)}
        return tags_dict

    def _get_ont_hot_list(self, tags):
        tags_dict = self._get_tags_dict(tags)

        length = len(tags_dict)
        ont_hot_list = list()

        for _, item in tags.iteritems():
            item_list = [0] * length

            try:
                item_tags_list = item.split(",")
                for tag in item_tags_list:
                    idx = tags_dict[tag]
                    item_list[idx] = 1
            except AttributeError:
                tag = str(item)
                idx = tags_dict[tag]
                item_list[idx] = 1
            ont_hot_list.append(item_list)

        return ont_hot_list

    def _get_svd_df(self, tags, n_components):
        ont_hot_list = self._get_ont_hot_list(tags)
        svd = TruncatedSVD(n_components=n_components)
        csr = sparse.csr_matrix(ont_hot_list)
        svd_data = svd.fit_transform(csr)
        svd_columns = ["svd_{i}".format(i=i) for i in range(n_components)]
        svd_df = pd.DataFrame(data=svd_data, columns=svd_columns)
        return svd_df

    def _get_rate_svd_df(self, tags, click, n_components):
        ont_hot_list = self._get_ont_hot_list(tags)
        item_length = len(ont_hot_list[0])
        columns = ["one_hot_{i}".format(i=i) for i in range(item_length)]
        ont_hot_df = pd.DataFrame(data=ont_hot_list, columns=columns)
        tags_click_df = pd.concat([ont_hot_df, click], axis=1)

        for column in columns:
            sub_df = tags_click_df[tags_click_df[column] == 1]
            sub_df_rate = np.nanmean(sub_df["click"])
            tags_click_df[tags_click_df[column] == 1][column] = sub_df_rate

        tags_click_df = tags_click_df.drop(columns=["click"])
        svd = TruncatedSVD(n_components=n_components)
        svd_data = svd.fit_transform(tags_click_df)
        svd_columns = ["svd_{i}".format(i=i) for i in range(n_components)]
        rate_svd_df = pd.DataFrame(data=svd_data, columns=svd_columns)
        return rate_svd_df

    def get_tags_df(self, tags, cilck, n_components=50):
        svd_df = self._get_svd_df(tags, n_components)

        # FIXME: 内存不够，无法运行
        rate_svd_df = self._get_rate_svd_df(tags, cilck, n_components)
        tags_df = pd.concat([svd_df, rate_svd_df], axis=1)

        return tags_df


class TimeProcessing(object):
    # time字段特殊处理
    @staticmethod
    def _get_init_timestamp():
        # 手动设定初始化时间戳
        # 表示x日0点0分0秒
        init_timestamp = 2190038402
        return init_timestamp

    def _get_week_features(self, item):
        # train为7天数据,test为1天数据
        init_timestamp = self._get_init_timestamp()
        interval_seconds = item - init_timestamp
        interval_days = int(interval_seconds / (3600 * 24))
        week_num = interval_days % 7
        return week_num

    def _get_hour_features(self, item):
        # 通过时间戳换算小时数
        init_timestamp = self._get_init_timestamp()
        interval_seconds = item - init_timestamp
        interval_hours_seconds = interval_seconds % (3600 * 24)
        hour_num = int(interval_hours_seconds / 3600)
        return hour_num

    def _get_period_features(self, item):
        # 通过时间戳换算天数
        # train为7天数据,test为1天数据
        init_timestamp = self._get_init_timestamp()
        interval_seconds = item - init_timestamp
        interval_days = int(interval_seconds / (3600 * 24))
        day_num = interval_days
        return day_num

    def get_times_df(self, times):
        if not isinstance(times, pd.Series):
            raise ValueError("times must be pd.Series!")

        df = pd.DataFrame()
        df["times_week"] = times.apply(self._get_week_features)
        df["times_hours"] = times.apply(self._get_hour_features)
        df["times_days"] = times.apply(self._get_period_features)
        df["times_period"] = df["times_days"]
        df = pd.get_dummies(df, columns=["times_week", "times_hours", "times_days"])
        return df


class TrendProcessing(object):

    @staticmethod
    def get_trend_df(df):
        df = df.copy()
        trend_columns = RATE_COLUMN.copy()
        trend_columns.append("advert_industry_1")
        trend_columns.append("advert_industry_2")
        trend_columns.append("creative_area")

        for column in trend_columns:
            trend_column_name = "{column}_trend".format(column=column)
            df[trend_column_name] = 0.0
            for i in range(8):
                if i == 0:
                    sub_df = df[df["times_period"] == 0]
                    sub_df_rate = np.nanmean(sub_df["click"])
                    df[df["times_period"] == 0][trend_column_name] = sub_df_rate
                else:
                    sub_df = df[df["times_period"] < i]
                    sub_df_rate = np.nanmean(sub_df["click"])
                    df[df["times_period"] == i][trend_column_name] = sub_df_rate

        return df


class Processing(object):
    @staticmethod
    def _get_data(name):
        raw_path = os.path.join("data", "RawData")

        if name == "train":
            file_name = os.path.join(raw_path, "round1_iflyad_train.txt")
        elif name == "test":
            file_name = os.path.join(raw_path, "round1_iflyad_test.txt")
        else:
            raise ValueError("name must be `train` or `test`!")

        df = pd.read_csv(file_name, header=0, sep="\t", encoding="utf-8")
        df = df.drop(columns=REMOVE_COLUMN)
        return df

    @staticmethod
    def _get_filter_num(x, threshold=20):
        x_counter = Counter(x)
        filter_num_list = list()

        for item, cnt in x_counter.items():
            if cnt > threshold:
                continue
            filter_num_list.append(item)
        return filter_num_list

    @staticmethod
    def _filter_num(x, filter_num_list):
        x = x.copy()

        for item in filter_num_list:
            x = x.apply(lambda _item: -1 if _item == item else _item)
        return x

    @staticmethod
    def _get_rate_dict(x, y, threshold=20):
        avg_rate = np.nanmean(y)
        x_set = set(x)

        rate_dict = dict()
        for item in x_set:
            sub_y = y[x == item]
            if len(sub_y) <= threshold:
                rate_dict[item] = avg_rate
            sub_y_rate = np.nanmean(sub_y)
            rate_dict[item] = sub_y_rate
        return rate_dict

    def _get_label_encoder(self, df):
        df = df.copy()
        columns = df.columns

        label_columns = list()
        for column in columns:
            if column in ["instance_id", "click", "user_tags"]:
                continue
            if "times_" in column:
                continue
            label_columns.append(column)

        encoder = LabelEncoder()
        for column in label_columns:
            column_item = df[column]
            filter_num_list = self._get_filter_num(column_item)
            df[column] = self._filter_num(df[column], filter_num_list)
            df[column] = encoder.fit_transform(df[column].astype(np.str_))

        return df

    def get_processing(self):
        train_df = self._get_data(name="train")
        test_df = self._get_data(name="test")
        labels = train_df["click"]

        etl_path = os.path.join("data", "EtlData")
        train_name = os.path.join(etl_path, "train_features.csv")
        test_name = os.path.join(etl_path, "test_features.csv")

        train_df_length = train_df.shape[0]
        df = pd.concat([train_df, test_df], axis=0, ignore_index=True, sort=False)

        # 特殊处理
        df["app_id"] = df["app_id"].fillna(-1).astype(np.int_)
        df["app_cate_id"] = df["app_cate_id"].fillna(-1).astype(np.int_)
        df["creative_area"] = df["creative_width"] * df["creative_height"]
        df["advert_industry_1"], df["advert_industry_2"] = df["advert_industry_inner"].str.split("_").str
        df = df.drop(columns=["advert_industry_inner"])
        df["os_osv"] = df['os'].astype(str).values + '_' + df['osv'].astype(str).values
        df = df.drop(columns=["os", "osv"])

        time_processer = TimeProcessing()
        times_df = time_processer.get_times_df(df["time"])
        df = pd.concat([df, times_df], axis=1)
        df = df.drop(columns=["time"])

        df = self._get_label_encoder(df)

        tags_processer = TagsProcessing()
        tags_df = tags_processer.get_tags_df(df["user_tags"], df["click"])
        df = pd.concat([df, tags_df], axis=1)
        df = df.drop(columns=["user_tags"])

        trend_processer = TrendProcessing()
        df = trend_processer.get_trend_df(df)
        df = df.drop(columns=["times_period"])

        for column in RATE_COLUMN:
            rate_dict = self._get_rate_dict(df[column], labels)
            column_name = "{column}_rate".format(column=column)
            df[column_name] = df[column].map(rate_dict)

        DUMPS_COLUMN.append("advert_industry_1")
        DUMPS_COLUMN.append("advert_industry_2")
        DUMPS_COLUMN.append("creative_area")
        df = pd.get_dummies(df, columns=DUMPS_COLUMN)

        train_df = df[:train_df_length]
        test_df = df[train_df_length:]

        train_df["click"] = train_df["click"].apply(int)
        test_df = test_df.drop(columns=["click"])

        train_df.to_csv(train_name, index=False)
        test_df.to_csv(test_name, index=False)


def binning(q=20):
    # 分箱
    train_data = get_data(name="train")
    test_data = get_data(name="test")

    train_data_length = train_data.shape[0]
    df = pd.concat([train_data, test_data], axis=0, ignore_index=True, sort=False)

    remove_fields = ["instance_id", "click"]
    columns = df.columns
    for column in columns:
        if column in remove_fields:
            continue
        if df[column].dtype == np.float64:
            unique_count = df[column].unique().shape[0]
            q = min(q, unique_count)
            df[column] = pd.qcut(df[column], q=q, labels=False, duplicates="drop")

    train_data = df[:train_data_length]
    test_data = df[train_data_length:]

    train_data["click"] = train_data["click"].apply(int)
    test_data = test_data.drop(columns=["click"])

    etl_path = os.path.join("data", "EtlData")
    train_name = os.path.join(etl_path, "qcut_train_features.csv")
    test_name = os.path.join(etl_path, "qcut_test_features.csv")

    train_data.to_csv(train_name, index=False)
    test_data.to_csv(test_name, index=False)


def get_variance_selector_columns(data, threshold):
    selector = VarianceThreshold(threshold=threshold)
    x_reduced = selector.fit(data)
    selector_index = x_reduced.get_support(indices=True)

    selector_columns = data.columns[selector_index]
    return selector_columns


def get_model_selector_columns(data, labels):
    clf = ExtraTreesClassifier()
    clf = clf.fit(data, labels)
    selector = SelectFromModel(clf, threshold="1.25*mean", prefit=True)
    selector_index = selector.get_support(indices=True)

    selector_columns = data.columns[selector_index]
    return selector_columns


def features_selector():
    train_data = get_data(name="train", qcut=True, filter_=False)
    test_data = get_data(name="test", qcut=True, filter_=False)

    columns = train_data.columns
    remove_fields = ["instance_id", "click"]
    features = [column for column in columns if column not in remove_fields]

    # VarianceThreshold
    # FIXME:
    selector_columns = get_variance_selector_columns(train_data[features], threshold=0.01)
    drop_columns = [column for column in features if column not in selector_columns]
    train_data = train_data.drop(columns=drop_columns)
    test_data = test_data.drop(columns=drop_columns)

    # Model Select
    columns = train_data.columns
    features = [column for column in columns if column not in remove_fields]
    selector_columns = get_model_selector_columns(train_data[features], train_data["click"])
    drop_columns = [column for column in features if column not in selector_columns]

    train_data = train_data.drop(columns=drop_columns)
    test_data = test_data.drop(columns=drop_columns)

    etl_path = os.path.join("data", "EtlData")
    train_name = os.path.join(etl_path, "filter_qcut_train_features.csv")
    test_name = os.path.join(etl_path, "filter_qcut_test_features.csv")

    train_data.to_csv(train_name, index=False)
    test_data.to_csv(test_name, index=False)


if __name__ == "__main__":
    Processing().get_processing()
    # features_selector()
