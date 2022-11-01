from itertools import accumulate
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path


def get_lost():
    return ["MS-SSIM", "corrected_MS-SSIM", "MDTVSFA", "corrected_MDTVSFA", "corrected_ERQA", "bitrate", "std_colorfulness", "SI", "TI"]

def get_datasets():
    data_file = "updated_data.csv"

    data = pd.read_csv(data_file)

    data = data.drop(columns=get_lost())
    #data["LPIPS"] = data["LPIPS"].apply(lambda x: x * -1)
    #data["corrected_LPIPS"] = data["corrected_LPIPS"].apply(lambda x: x * -1)


    subjective_scores = data.iloc[:, -1]
    cases = data.iloc[:, 0]
    data = data.iloc[: , 1:-1]

    accumulative_data = {}

    for _case, _score in zip(cases, subjective_scores):
        _video = list(_case.split("@"))[1]
        if _video not in accumulative_data.keys():
            accumulative_data[_video] = {
                "min" : 100,
                "max" : -1
            }
        if accumulative_data[_video]["min"] > _score:
            accumulative_data[_video]["min"] = _score
        if accumulative_data[_video]["max"] < _score:
            accumulative_data[_video]["max"] = _score

    #print(accumulative_data)
    
    for i in range(len(subjective_scores)):
        _video = list(cases[i].split("@"))[1]
        subjective_scores[i] -= accumulative_data[_video]["min"]
        subjective_scores[i] /= (accumulative_data[_video]["max"] - accumulative_data[_video]["min"])
        #subjective_scores[i] *= 5
    

    videos = []
    for case in cases:
        words = list(case.split("@"))
        video = words[1]
        videos.append(video)

    video_pull = np.unique(videos)

    test_cases = [
        ["cuphead", "pig", "beach"],
        ["colors", "statue", "bridge"],
        ["dancing", "camera", "classroom"]
    ]

    train_cases = []

    for case in test_cases:
        cur_videos = list(video_pull)
        for video in case:
            cur_videos.remove(video)
            
        train_cases.append(cur_videos)


    datasets = []

    for train_case, test_case in zip(train_cases, test_cases):
        indices = []
        for i, video in enumerate(videos):
            if video in train_case:
                indices.append(i)
        data_train = data.iloc[indices]
        cases_train = cases.iloc[indices]       
        subjective_train = subjective_scores.iloc[indices]

        test_indices = list(range(len(videos)))
        for ind in indices:
            test_indices.remove(ind)

        data_test = data.iloc[test_indices]
        cases_test = cases.iloc[test_indices]
        subjective_test = subjective_scores.iloc[test_indices]

        dataset = {
            "X_train" : data_train,
            "y_train" : subjective_train,
            "X_test" : data_test,
            "y_test" : subjective_test,
            "test_cases" : cases_test
        }

        datasets.append(dataset)

    return datasets

def format_dataset(dataset):
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    #print(np.concatenate([y_train, y_test], axis=0))

    return X_train, y_train, X_test, y_test, np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test], axis=0)