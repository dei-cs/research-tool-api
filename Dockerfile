FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# System deps for OCR (Tesseract) + PDF images (Poppler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]