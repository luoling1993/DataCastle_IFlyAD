#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np
import pandas as pd


class Entropy(object):
    @staticmethod
    def get_entropy(x):
        # 计算信息熵
        x_counter = Counter(x)
        length = len(x)

        entropy = 0.0
        for _, cnt in x_counter.items():
            p = float(cnt / length)
            logp = np.log2(p)
            entropy -= p * logp

        return entropy

    def get_split_info(self, x):
        # 计算分裂信息:同计算信息熵相同
        return self.get_entropy(x)

    def get_condition_entropy(self, x, y):
        # 计算条件信息熵
        if len(x) != len(y):
            raise ValueError("The length of x anf y should equal!")
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        x_set = set(x)
        length = len(x)

        condition_entropy = 0.0
        for item in x_set:
            sub_y = y[x == item]
            sub_length = len(sub_y)
            sub_y_entropy = self.get_entropy(sub_y)
            condition_entropy += (sub_length / length) * sub_y_entropy

        return condition_entropy

    def get_entropy_gain(self, x, y):
        # 计算信息增益
        base_entropy = self.get_entropy(y)
        condition_entropy = self.get_condition_entropy(x, y)
        entropy_gain = condition_entropy - base_entropy
        return entropy_gain

    def get_entropy_gain_ratio(self, x, y):
        # 计算信息增益率
        entropy_gain = self.get_entropy_gain(x, y)
        split_info = self.get_split_info(x)
        entropy_gain_ratio = entropy_gain / split_info
        return entropy_gain_ratio


