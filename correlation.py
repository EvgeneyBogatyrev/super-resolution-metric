import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr


def correlation(gt, result, test_cases):
    data = {}
    for gt_sample, result_sample, case in zip(gt, result, test_cases):
        video = list(case.split("@"))[1]
        codec = list(case.split("@"))[0]
        if video + "@" + codec not in data.keys():
            data[video + "@" + codec] = []
        data[video + "@" + codec].append((gt_sample, result_sample))
    
    pearson = []
    spearman = []

    for video in data.keys():
        cur_gt = []
        cur_result = []
        for pair in data[video]:
            cur_gt.append(pair[0])
            cur_result.append(pair[1])
        pear = pearsonr(cur_gt, cur_result)[0]
        spear = spearmanr(cur_gt, cur_result)[0]

        pearson.append(pear)
        spearman.append(spear)

    return pearson, spearman
