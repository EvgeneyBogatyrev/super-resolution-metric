import pickle
import numpy as np
from get_values import calculate_ERQA_and_LPIPS_and_color_on_video, \
    calculate_SI_TI, calculate_MDTVSFA, calculate_ERQA_and_LPIPS_and_color_on_frame


def mymetric(gt, dist, video=True, gt_params=None, dist_params=None):
    if video:
        metric = pickle.load(open("models/model.pkl", 'rb'))
        scaler = pickle.load(open("models/scaler.pkl", 'rb'))

        ERQA_value, LPIPS_value, colorfulness = calculate_ERQA_and_LPIPS_and_color_on_video(gt, dist, gt_params, dist_params)
        SI, TI = calculate_SI_TI(dist, dist_params)
        MDTVSFA_value = calculate_MDTVSFA(dist, dist_params)
        ERQAxLPIPS_value = ERQA_value * (1 - LPIPS_value)
        ERQAxMDTVSFA_value = ERQA_value * MDTVSFA_value
        
        tensor = np.array([[ERQA_value, LPIPS_value, ERQAxLPIPS_value,
            ERQAxMDTVSFA_value, SI, TI, colorfulness]])

        tensor = scaler.transform(tensor)

        result = metric.predict(tensor)[0]
        return result
    else:
        metric = pickle.load(open("models/model_image.pkl", 'rb'))
        scaler = pickle.load(open("models/scaler_image.pkl", 'rb'))

        ERQA_value, LPIPS_value, colorfulness = calculate_ERQA_and_LPIPS_and_color_on_frame(gt, dist)
        ERQAxLPIPS_value = ERQA_value * (1 - LPIPS_value)

        tensor = np.array([[ERQA_value, LPIPS_value, ERQAxLPIPS_value,
            colorfulness]])

        tensor = scaler.transform(tensor)

        result = metric.predict(tensor)[0]
        return result