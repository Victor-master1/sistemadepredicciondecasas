FROM python:3.10-slim-buster
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"]
