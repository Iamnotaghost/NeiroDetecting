#!/bin/bash
# Скрипт для проверки работоспособности API

set -e

# Получаем порт из переменной окружения или используем значение по умолчанию
PORT=${PORT:-5000}

# Выполняем запрос к эндпоинту /api/health
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/api/health)

# Проверяем, что код ответа 200
if [ "$RESPONSE" -eq 200 ]; then
    exit 0
else
    exit 1
fi