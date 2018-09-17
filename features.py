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
            return str(tags)

    def get_topn_tags(self, topn=10):
        # 获取频率最高的user_tags
        tags_list = list()
        for _, tags in self.user_tags.iteritems():
            try:
                _tags_list = tags.split(",")
                tags_list.extend(_tags_list)
            except AttributeError:
                tags_list.append(str(tags))
        tags_counter = Counter(tags_list)
        topn_counter = tags_counter.most_common(topn)
        topn_tags = [item[0] for item in topn_counter]
        return topn_tags

    def get_topn_tags_group(self, topn=10):
        # 获取频率最高的user_tags组合
        tags_group_list = list()
        for _, tags in self.user_tags.iteritems():
            sorted_tags = self._sorted_tags(tags)
            tags_group_list.append(sorted_tags)
        tags_group_counter = Counter(tags_group_list)
        topn_counter = tags_group_counter.most_common(topn)
        topn_tags_group = [item[0] for item in topn_counter]
        return topn_tags_group

    def get_topn_tags_df(self, topn_tags):
        # 获取topn_tags构造的DataFrame
        columns = list()
        tags_df_list = list()

        for topn_tag in topn_tags:
            coumn = "user_tags_{topn_tag}".format(topn_tag=topn_tag)
            columns.append(coumn)

        for user_tag in self.user_tags:
            try:
                user_tag_list = user_tag.split(",")
            except AttributeError:
                user_tag_list = [str(user_tag)]

            tags_list = list()
            for topn_tag in topn_tags:
                if topn_tag in user_tag_list:
                    tags_list.append(1)
                else:
                    tags_list.append(0)

            tags_df_list.append(tags_list)

        tags_df = pd.DataFrame(data=tags_df_list, columns=columns)
        return tags_df

    def get_topn_tags_group_df(self, topn_tags_group):
        # 获取topn_tags_group构造的DataFrame
        columns = list()
        tags_group_df_list = list()

        for idx, _ in enumerate(topn_tags_group):
            coumn = "topn_tags_group_{idx}".format(idx=idx)
            columns.append(coumn)

        for user_tag in self.user_tags:
            sorted_user_tags = self._sorted_tags(user_tag)

            tags_group_list = list()
            for tags_group in topn_tags_group:
                if tags_group == sorted_user_tags:
                    tags_group_list.append(1)
                else:
                    tags_group_list.append(0)

            tags_group_df_list.append(tags_group_list)

        tags_group_df = pd.DataFrame(data=tags_group_df_list, columns=columns)
        return tags_group_df


class TimeProcessing(object):
    # time字段特殊处理
    def __init__(self, times):
        if not isinstance(times, pd.Series):
            raise ValueError("times must be pd.Series!")
        self.times = times

    @staticmethod
    def _get_init_timestamp():
        # 手动设定初始化时间戳
        # 表示x日0点0分0秒
        init_timestamp = 0
        return init_timestamp

    def get_week_features(self):
        # train为7天数据,test为1天数据
        pass

    def get_hour_features(self):
        # 通过时间戳换算小时数
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
    pass
    # test_df = Processing()._get_data(name="test")
    # tag_processing = TagsProcessing(test_df["user_tags"])
    # topn_tags = tag_processing.get_topn_tags(topn=10)
    # df = tag_processing.get_topn_tags_df(topn_tags)
    # print(df.head(10))
