FROM python:3.11-slim

WORKDIR /app
ENV HF_HOME=/app/model_cache

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model build ke time hi image mein save kar lo taaki cold start fast rahe
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
m = 'cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual'; \
AutoTokenizer.from_pretrained(m); \
AutoModelForSequenceClassification.from_pretrained(m)"

COPY core.py cloudrun_app.py .

CMD exec uvicorn cloudrun_app:app --host 0.0.0.0 --port ${PORT:-8080}
