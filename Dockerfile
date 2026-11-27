FROM python:3.11-slim

WORKDIR /app

# Hugging Face Spaces mein cache permission error se bachne ke liye
ENV HF_HOME=/tmp/huggingface

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]