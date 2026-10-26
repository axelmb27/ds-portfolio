# 1. Base: Python 3.12 + runtime de Lambda
FROM public.ecr.aws/lambda/python:3.12

# 2. Primero SOLO las dependencias
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt

# 3. Tu código y tu modelo
COPY app.py predict.py features.py ${LAMBDA_TASK_ROOT}/
COPY model/ ${LAMBDA_TASK_ROOT}/model/

# 4. La función que Lambda llama en cada petición
CMD [ "app.handler" ]