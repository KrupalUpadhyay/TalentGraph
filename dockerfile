FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY dashboard ./dashboard
COPY models ./models
COPY data/raw/role-radar/scraped_jobs.json ./data/raw/role-radar/scraped_jobs.json

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
