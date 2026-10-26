"""F1.2–F1.3: entrena el Pipeline (features + XGBoost) y lo guarda con joblib.

Uso, desde deploy/:  .venv/bin/python train.py
"""
import os

import kagglehub
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib

from features import RAW_COLS, add_features

TARGET = "isFraud"


def load_data() -> pd.DataFrame:
    path = kagglehub.dataset_download("ealaxi/paysim1")  
    df = pd.read_csv(os.path.join(path, "PS_20174392719_1491204439457_log.csv"),
                     usecols=RAW_COLS + [TARGET])        
    df = df[df["type"].isin(["TRANSFER","CASH_OUT"])]
    return df


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(add_features)),  # paso 1: crudo -> features
        ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
                            n_jobs=-1, random_state=42))
    ])


if __name__ == "__main__":
    df = load_data()
    X, y = df[RAW_COLS], df[TARGET]

    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
    # Compute scale scale_pos = negatives for every positive
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    print("scale_pos_weight :", round(scale_pos, 1))
    pipe = build_pipeline(scale_pos)
    pipe.fit(X_train, y_train)
    proba_pipe = pipe.predict_proba(X_test)[:, 1]
    print(f'\nXGBoost: ROC-AUC = {roc_auc_score(y_test, proba_pipe):.4f}, PR-AUC = {average_precision_score(y_test, proba_pipe):.4f}')
    os.makedirs("model", exist_ok=True)
    joblib.dump(pipe, "model/fraud_xgb.joblib")
    print("Model stored in model/fraud_xgb.joblib")