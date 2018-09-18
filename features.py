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
    @staticmethod
    def _sorted_tags(tags):
        try:
            tags_list = tags.split(",")
            return sorted(tags_list)
        except AttributeError:
            return str(tags)

    @staticmethod
    def get_topn_tags(user_tags, topn=100):
        # 获取频率最高的user_tags
        if not isinstance(user_tags, pd.Series):
            raise ValueError("user_tags must be pd.Series!")

        tags_list = list()
        for _, tags in user_tags.iteritems():
            try:
                _tags_list = tags.split(",")
                tags_list.extend(_tags_list)
            except AttributeError:
                tags_list.append(str(tags))
        tags_counter = Counter(tags_list)
        topn_counter = tags_counter.most_common(topn)
        topn_tags = [item[0] for item in topn_counter]
        return topn_tags

    def get_topn_tags_group(self, user_tags, topn=100):
        # 获取频率最高的user_tags组合
        if not isinstance(user_tags, pd.Series):
            raise ValueError("user_tags must be pd.Series!")

        tags_group_list = list()
        for _, tags in user_tags.iteritems():
            sorted_tags = self._sorted_tags(tags)
            if isinstance(sorted_tags, list):
                sorted_tags = ",".join(map(str, sorted_tags))
            tags_group_list.append(sorted_tags)
        tags_group_counter = Counter(tags_group_list)
        topn_counter = tags_group_counter.most_common(topn)
        topn_tags_group = [item[0] for item in topn_counter]
        return topn_tags_group

    @staticmethod
    def get_topn_tags_df(user_tags, topn_tags):
        # 获取topn_tags构造的DataFrame
        columns = list()
        tags_df_list = list()

        for topn_tag in topn_tags:
            coumn = "user_tags_{topn_tag}".format(topn_tag=topn_tag)
            columns.append(coumn)

        for user_tag in user_tags:
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

    def get_topn_tags_group_df(self, user_tags, topn_tags_group):
        # 获取topn_tags_group构造的DataFrame
        if not isinstance(user_tags, pd.Series):
            raise ValueError("user_tags must be pd.Series!")

        columns = list()
        tags_group_df_list = list()

        for idx, _ in enumerate(topn_tags_group):
            coumn = "topn_tags_group_{idx}".format(idx=idx)
            columns.append(coumn)

        for user_tag in user_tags:
            sorted_user_tags = self._sorted_tags(user_tag)
            if isinstance(sorted_user_tags, list):
                sorted_user_tags = ",".join(map(str, sorted_user_tags))

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

    def get_times_df(self, times):
        if not isinstance(times, pd.Series):
            raise ValueError("times must be pd.Series!")

        df = pd.DataFrame()
        df["times_week"] = times.apply(self._get_week_features)
        df["times_hours"] = times.apply(self._get_hour_features)
        df = pd.get_dummies(df, columns=["times_week", "times_hours"])
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

    def get_processing(self):
        train_df = self._get_data(name="train")
        test_df = self._get_data(name="test")
        labels = train_df["click"]

        etl_path = os.path.join("data", "EtlData")
        train_name = os.path.join(etl_path, "train_features.csv")
        test_name = os.path.join(etl_path, "test_features.csv")

        train_df_length = train_df.shape[0]
        df = pd.concat([train_df, test_df], axis=0, ignore_index=True, sort=False)

        df["creative_area"] = df["creative_width"] * df["creative_height"]
        df["advert_industry_1"], df["advert_industry_2"] = df["advert_industry_inner"].str.split("_").str
        df = df.drop(columns=["advert_industry_inner"])

        for column in RATE_COLUMN:
            rate_dict = self._get_rate_dict(df[column], labels)
            df[column] = df[column].map(rate_dict)

        for column in DUMPS_COLUMN:
            column_item = df[column]

            if df[column].dtype == "bool":
                df[column] = df[column].map({False: 0, True: 1})
                continue

            filter_num_list = self._get_filter_num(column_item)
            df[column] = self._filter_num(df[column], filter_num_list)

        tags_processer = TagsProcessing()
        topn_tags = tags_processer.get_topn_tags(df["user_tags"])
        topn_tags_group = tags_processer.get_topn_tags_group(df["user_tags"])
        topn_tags_df = tags_processer.get_topn_tags_df(df["user_tags"], topn_tags)
        topn_tags_group_df = tags_processer.get_topn_tags_group_df(df["user_tags"], topn_tags_group)

        df = pd.concat([df, topn_tags_df, topn_tags_group_df], axis=1)
        df = df.drop(columns=["user_tags"])

        time_processer = TimeProcessing()
        times_df = time_processer.get_times_df(df["time"])
        df = pd.concat([df, times_df], axis=1)
        df = df.drop(columns=["time"])

        DUMPS_COLUMN.append("advert_industry_1")
        DUMPS_COLUMN.append("advert_industry_2")
        df = pd.get_dummies(df, columns=DUMPS_COLUMN)

        train_df = df[:train_df_length]
        test_df = df[train_df_length:]

        train_df["click"] = train_df["click"].apply(int)
        test_df = test_df.drop(columns=["click"])

        train_df.to_csv(train_name, index=False)
        test_df.to_csv(test_name, index=False)


if __name__ == "__main__":
    Processing().get_processing()
