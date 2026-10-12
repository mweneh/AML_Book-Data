FROM python:3.12-slim

WORKDIR /app

# Install the API's packages first (Docker caches this layer)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the API code and the trained model (run train.py before building)
COPY api/   api/
COPY model/ model/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
