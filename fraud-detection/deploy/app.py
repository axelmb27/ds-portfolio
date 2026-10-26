"""F3: handler de Lambda. Lambda llama a handler(event, context) en cada petición."""
import base64
import json

from predict import predict  # al importar, predict.py carga el modelo UNA vez


def _read_body(event: dict) -> dict:
    """La Function URL manda el JSON como TEXTO en event["body"] (a veces en base64)."""
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    return json.loads(body)  # texto JSON -> diccionario de Python


def _response(status: int, payload: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),  # diccionario de Python -> texto JSON
    }


def handler(event, context):
    try:
        body =_read_body(event)
        p = predict(body)
        return _response(200,p)
    except ValueError as e:
        return _response(400,{"error": str(e)}) 

if __name__ == "__main__":
    # Prueba rápida sin Docker: python app.py
    buena = {"body": json.dumps({"step": 26, "type": "TRANSFER", "amount": 500000.0})}
    mala = {"body": json.dumps({"step": 26, "type": "PAYMENT", "amount": 5.0})}
    rota = {"body": "esto no es JSON"}
    for nombre, evento in [("buena", buena), ("mala", mala), ("rota", rota)]:
        print(nombre, "->", handler(evento, None))
