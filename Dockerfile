# Используем многоэтапную сборку для оптимизации размера образа

# Этап 1: Сборка зависимостей
FROM python:3.11-slim AS builder

WORKDIR /app

# Устанавливаем необходимые пакеты для компиляции зависимостей
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Копируем только файл с зависимостями для лучшего кэширования
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install setuptools wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Этап 2: Основной образ
FROM python:3.11-slim

# Создаем директории и пользователя сразу (до копирования файлов)
RUN addgroup --system app && \
    adduser --system --group app && \
    mkdir -p /app/src /app/app /app/models /app/logs && \
    chown -R app:app /app

WORKDIR /app

# Устанавливаем переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.pkl \
    VECTORIZER_PATH=/app/models/vectorizer.pkl \
    LABEL_ENCODER_PATH=/app/models/label_encoder.pkl \
    LOG_FILE=/app/logs/app.log \
    PORT=5000 \
    ENABLE_METRICS=true \
    METRICS_PORT=9090

# Копируем собранные пакеты из предыдущего этапа
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/* && \
    rm -rf /wheels

# Скачиваем ресурсы NLTK под непривилегированным пользователем
USER app
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/app/nltk_data')"
ENV NLTK_DATA=/app/nltk_data

# Возвращаемся к root для копирования приложения с правильными правами
USER root

# Копируем приложение
COPY --chown=app:app src/ /app/src/
COPY --chown=app:app app/ /app/app/

# Создаем пустую директорию для моделей, если они будут примонтированы
RUN mkdir -p /app/models && chown -R app:app /app/models

# Копируем скрипт для проверки здоровья
COPY --chown=app:app docker-healthcheck.sh /app/
RUN chmod +x /app/docker-healthcheck.sh

# Добавляем проверку здоровья контейнера
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["/app/docker-healthcheck.sh"]

# Переключаемся на непривилегированного пользователя
USER app

# Указываем порты, которые будет прослушивать приложение
EXPOSE ${PORT}
EXPOSE ${METRICS_PORT}

# Запускаем приложение
CMD ["python", "-m", "app.main"]