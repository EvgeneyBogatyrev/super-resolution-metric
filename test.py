import pickle
import numpy as np
from scipy.stats import spearmanr, pearsonr
from data import get_datasets, format_dataset
from correlation import correlation

with open("models/model.pkl", "rb") as f:
    s = f.read()

model = pickle.loads(s)

datasets = get_datasets()
dataset = datasets[0]
X_train, y_train, X_test, y_test, whole_data, whole_test = format_dataset(dataset)
res = model.predict(X_test)
pearson, spearman = correlation(y_test, res, dataset["test_cases"])
print(f"Mean pearson: {np.mean(pearson)}\nMean spearman: {np.mean(spearman)}\n")