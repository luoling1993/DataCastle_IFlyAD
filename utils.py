#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def get_rate_dict(x, y, threshold=50):
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