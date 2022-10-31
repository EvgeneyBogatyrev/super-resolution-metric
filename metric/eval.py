import pickle
import numpy as np
import lpips

metric = pickle.load(open("models/model.pkl", 'rb'))
scaler = pickle.load(open("models/scaler.pkl", 'rb'))

ERQA_value = 0.5323870049464775
LPIPS_value = 0.22963963157894735
ERQAxLPIPS_value = 0.37344376365681164
ERQAxMDTVSFA_value = 0.2571909805397002
SI = 0.01128343621399177
TI = 0.5934244224245432
colorfulness = 10.601520481769922

tensor = np.array([[ERQA_value, LPIPS_value, ERQAxLPIPS_value,
    ERQAxMDTVSFA_value, SI, TI, colorfulness]])

tensor = scaler.transform(tensor)

result = metric.predict(tensor)[0]
print(result)
