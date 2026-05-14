import cv2
import numpy as np
import joblib
import tensorflow.keras as keras
from config import IMG_SIZE, CLASS_MAP

fundus_model = keras.models.load_model("models/fundus_model.h5", compile=False)
oct_model    = keras.models.load_model("models/oct_model.h5",    compile=False)

scaler = joblib.load("models/scaler.pkl")
lgbm   = joblib.load("models/lgbm_model.pkl")


def preprocess(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, 0)


def predict_dr(fundus_path, oct_path):
    f = preprocess(fundus_path)
    o = preprocess(oct_path)

    f_feat = fundus_model.predict(f)
    o_feat = oct_model.predict(o)

    X = np.concatenate([f_feat, o_feat], axis=1)
    X = scaler.transform(X)

    pred  = lgbm.predict(X)[0]
    proba = lgbm.predict_proba(X)[0]

    return CLASS_MAP[pred], round(float(max(proba)) * 100, 2)