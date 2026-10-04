FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# مكتبات نظام لـ LightGBM + DuckDB
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 git curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install -e .

COPY . .

# مجلد التخزين المحلي
RUN mkdir -p /app/storage

EXPOSE 7860

# افتراضيًا: تشغيل الواجهة فقط (لـ HF Space)
# للتشغيل الكامل استبدل بـ: CMD ["python", "main.py"]
CMD ["python", "space_app.py"]
