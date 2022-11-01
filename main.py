import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from correlation import correlation
from data import get_datasets, format_dataset, get_lost


datasets = get_datasets()

print(get_lost())

for md in ["SVR", "LinearRegression"]:
    for i, dataset in enumerate(datasets):
        X_train, y_train, X_test, y_test, whole_data, whole_test = format_dataset(dataset)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if md == "SVR":
            model = SVR(kernel='rbf')
        else:
            model = LinearRegression()
        model.fit(X_train, y_train)

        res = model.predict(X_test)
        print(f"=={i + 1}==")
        print(md)
        pearson, spearman = correlation(y_test, res, dataset["test_cases"])
        print(f"Mean pearson: {np.mean(pearson)}\nMean spearman: {np.mean(spearman)}\n")
        
        if i == 0 and md == "SVR":
            pickle.dump(model, open("model_image.pkl", 'wb'))
            pickle.dump(scaler, open("scaler_image.pkl", 'wb'))
            
            
        
        '''
        if md == "SVR":
            importances = model.coef_[0]
        else:
            importances = model.coef_
        print(importances)
        '''
        feature_names = ["ERQA","LPIPS","MDTVSFA","MS-SSIM",
            "corrected_ERQA", "corrected_LPIPS", "corrected_MDTVSFA",
                "corrected_MS-SSIM","SI","TI","bitrate","mean_colorfulness","std_colorfulness"]
        
        for name in get_lost():
            feature_names.remove(name)
        print(feature_names)
        '''
        comb = zip(feature_names, importances)
        comb = sorted(comb, key = lambda x: abs(x[1]), reverse=True)

        feature_names = [x[0] for x in comb]
        importances = [x[1] for x in comb]

        print(feature_names[:6])
        
        forest_importances = pd.Series(importances, index=feature_names)
        
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=importances, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout() 
        #plt.show()
        '''
        
    

loaded_model = pickle.load(open("model_image.pkl", 'rb'))
loaded_scaler = pickle.load(open("scaler_image.pkl", 'rb'))

datasets = get_datasets()
for dataset in datasets:
    scaler = MinMaxScaler()
    X_train, y_train, X_test, y_test, whole_data, whole_test = format_dataset(dataset)
    X_test = loaded_scaler.transform(X_test)
    res = loaded_model.predict(X_test)
    pearson, spearman = correlation(y_test, res, dataset["test_cases"])
    print(f"Mean pearson: {np.mean(pearson)}\nMean spearman: {np.mean(spearman)}\n")

