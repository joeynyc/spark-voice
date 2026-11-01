# The Spark Voice web app (no GPU needed: it talks to the model servers over HTTP).
FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY web web
COPY voices voices
COPY config.yaml .
ENV APP_PORT=7799 APP_BIND=0.0.0.0
USER 1000:1000
CMD uvicorn app.main:app --host ${APP_BIND} --port ${APP_PORT}
