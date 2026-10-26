"""Features del modelo de fraude.

Las usan train.py (al entrenar) y predict.py (en la API): una sola definición
para que entrenamiento y producción calculen exactamente lo mismo.
"""
import pandas as pd

RAW_COLS = ["step", "type", "amount"]  # lo único que recibe la API


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Columnas crudas -> features del modelo, siempre en el mismo orden."""
    X = X.copy()
    X['hour_of_day'] = X['step'].mod(24)
    X['is_night'] = X['hour_of_day'].isin([2,3,4,5,6]).astype(int)
    X['type_TRANSFER'] = X['type'].isin(["TRANSFER"]).astype(int)
    return X[['amount', 'step', 'hour_of_day', 'is_night','type_TRANSFER']]  # TODO lista fija de columnas


if __name__ == "__main__":
    # Prueba rápida: python features.py
    muestra = pd.DataFrame({
        "step": [1, 26, 743],
        "type": ["TRANSFER", "CASH_OUT", "TRANSFER"],
        "amount": [181.0, 5000.0, 1_000_000.0],
    })
    print(add_features(muestra))
    print(add_features(muestra.iloc[[0]]))  # una sola fila: ¿salen las mismas columnas?
