"""F1.4: carga el modelo guardado y predice una transacción.

Contrato:
    entra  {"step": 26, "type": "TRANSFER", "amount": 500000.0}
    sale   {"fraud_score": 0.98, "is_fraud": True}
"""
from pathlib import Path

import joblib
import pandas as pd

from features import RAW_COLS

MODEL_PATH = Path(__file__).parent / "model" / "fraud_xgb.joblib"
THRESHOLD = 0.9 #Selected for saving 83.8% of defrauded funds flagging 0.85% of transactions
VALID_TYPES = {"TRANSFER", "CASH_OUT"}

# Se carga UNA vez al importar el módulo, no en cada petición
# (en Lambda: al arrancar el contenedor; las siguientes peticiones lo reutilizan)
_model = joblib.load(MODEL_PATH)


def predict(transaction: dict) -> dict:
    # 1. Validar
    for col in RAW_COLS:
        if col not in transaction:
            raise ValueError(f"Missing key: {col}")
    if transaction["type"] not in VALID_TYPES:
        raise ValueError(f"Unsupported type: {transaction['type']}")
    if transaction["amount"] < 0:
        raise ValueError("amount must be >= 0")

    # 2. DataFrame de UNA fila, armado con lo que llegó en la petición
    df = pd.DataFrame([transaction])[RAW_COLS]

    # 3. Score
    score = float(_model.predict_proba(df)[:, 1][0])

    # 4. Respuesta
    return {"fraud_score": score, "is_fraud": score >= THRESHOLD}
