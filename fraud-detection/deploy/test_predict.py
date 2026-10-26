"""Pruebas de predict.py. Correr desde deploy/:  .venv/bin/pytest -v"""
import json

import pytest

from predict import predict

TRANSFER_NOCHE = {"step": 26, "type": "TRANSFER", "amount": 500000.0}


def test_respuesta_cumple_el_contrato():
    r = predict(TRANSFER_NOCHE)
    assert set(r) == {"fraud_score", "is_fraud"}  # exactamente esas dos llaves
    assert 0.0 <= r["fraud_score"] <= 1.0          # un score válido
    assert isinstance(r["is_fraud"], bool)
    json.dumps(r)                                  # truena si algo no se puede mandar como JSON


def test_falta_una_llave():
    with pytest.raises(ValueError):
        predict({"step": 26, "amount": 5.0})


def test_tipo_no_soportado():
    with pytest.raises(ValueError):
        predict({"step": 26, "type": "PAYMENT", "amount": 5.0})


def test_monto_negativo():
    with pytest.raises(ValueError):
        predict({"step": 26, "type": "TRANSFER", "amount": -1.0})
