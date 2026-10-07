FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

RUN mkdir -p model && python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='gowtham0046/parkitrace', filename='parkitrace_fp16.pt', local_dir='model')"

COPY app.py .
COPY parkitrace_model.py .
COPY templates ./templates

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 app:app"]
