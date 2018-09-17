#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import Counter

import numpy as np
import pandas as pd

REMOVE_COLUMN = ["make", "model", "os", "osv", "app_paid"]
DUMPS_COLUMN = ["province", "carrier", "devtype", "nnt", "os_name", "advert_id", "campaign_id", "creative_tp_dnf",
                "app_cate_id", "f_channel", "app_id", "creative_type", "creative_is_jump", "creative_is_download",
                "creative_is_js", "creative_is_voicead", "creative_has_deeplink", "advert_name"]
RATE_COLUMN = ["city", "adid", "orderid", "creative_id", "inner_slot_id", ]
DONNET_COLUMN = ["instance_id", "time", "user_tags", "creative_width", "creative_height", "advert_industry_inner",
                 "click"]


class TagsProcessing(object):
    # user_tags字段特殊处理
    def __init__(self, user_tags):
        if not isinstance(user_tags, pd.Series):
            raise ValueError("user_tags must be pd.Series!")
        self.user_tags = user_tags

    @staticmethod
    def _sorted_tags(tags):
        try:
            tags_list = tags.split(",")
            return sorted(tags_list)
        except AttributeError:
            return tags

    def _get_topn_tags(self, topn=10):
        # 获取频率最高的user_tags
        tags_list = list()
        for _, tags in self.user_tags.iteritems():
            try:
                _tags_list = tags.split(",")
                tags_list.extend(_tags_list)
            except AttributeError:
                tags_list.append(tags)
        tags_counter = Counter(tags_list)
        topn_counter = tags_counter.most_common(topn)
        topn_tags = [item[0] for item in topn_counter]
        return topn_tags

    def _get_topn_tags_group(self, topn=10):
        # 获取频率最高的user_tags组合
        tags_group_list = list()
        for _, tags in self.user_tags.iteritems():
            sorted_tags = self._sorted_tags(tags)
            tags_group_list.append(sorted_tags)
        tags_group_counter = Counter(tags_group_list)
        topn_counter = tags_group_counter.most_common(topn)
        topn_group_tags = [item[0] for item in topn_counter]
        return topn_group_tags


class TimeProcessing(object):
    # time字段特殊处理
    pass


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
        avg_rate = np.average(y)
        x_set = set(x)

        rate_dict = dict()
        for item in x_set:
            sub_y = y[x == item]
            if len(sub_y) <= threshold:
                rate_dict[item] = avg_rate
            sub_y_rate = np.average(sub_y)
            rate_dict[item] = sub_y_rate
        return rate_dict

    def get_processing(self):
        train_df = self._get_data(name="train")
        test_df = self._get_data(name="test")
        labels = train_df["click"]

        etl_path = os.path.join("data", "EtlData")
        train_name = os.path.join(etl_path, "train_features.csv")
        test_name = os.path.join(etl_path, "test_features.csv")

        for column in RATE_COLUMN:
            rate_dict = self._get_rate_dict(train_df[column], labels)
            train_df[column] = train_df[column].map(rate_dict)
            test_df[column] = test_df[column].map(rate_dict)

        for column in DUMPS_COLUMN:
            train_column = train_df[column]

            if train_column.dtype == "bool":
                train_df[column] = train_df[column].map({False: 0, True: 1})
                test_df[column] = test_df[column].map({False: 0, True: 1})
                continue

            filter_num_list = self._get_filter_num(train_column)
            train_df[column] = self._filter_num(train_df[column], filter_num_list)
            test_df[column] = self._filter_num(test_df[column], filter_num_list)

        train_df = pd.get_dummies(train_df, columns=DUMPS_COLUMN)
        test_df = pd.get_dummies(test_df, columns=DUMPS_COLUMN)

        train_df.to_csv(train_name, index=False)
        test_df.to_csv(test_name, index=False)


if __name__ == "__main__":
    Processing().get_processing()
