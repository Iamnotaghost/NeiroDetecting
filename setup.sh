#!/bin/bash

# Скрипт для настройки окружения проекта

# Выход при ошибке
set -e

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Начинаем настройку окружения для классификатора товаров${NC}"

# Проверка конкретно Python 3.11
if command -v python3.11 &>/dev/null; then
    PYTHON=python3.11
    echo -e "${GREEN}Используется $(${PYTHON} --version)${NC}"
elif command -v python3 &>/dev/null; then
    PYTHON=python3
    PYTHON_VERSION=$(${PYTHON} --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -eq 11 ]; then
        echo -e "${GREEN}Используется $(${PYTHON} --version)${NC}"
    else
        echo -e "${YELLOW}Внимание: обнаружен Python $PYTHON_VERSION, но проект оптимизирован для Python 3.11${NC}"
        echo -e "${YELLOW}Продолжаем с $(${PYTHON} --version)${NC}"
    fi
elif command -v python &>/dev/null; then
    PYTHON=python
    PYTHON_VERSION=$(${PYTHON} --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ]; then
        echo -e "${YELLOW}Используется $(${PYTHON} --version), но проект оптимизирован для Python 3.11${NC}"
    else
        echo -e "${RED}Обнаружен Python $PYTHON_VERSION, но требуется Python 3.x${NC}"
        exit 1
    fi
else
    echo -e "${RED}Python не найден. Пожалуйста, установите Python 3.11${NC}"
    exit 1
fi

# Создание виртуального окружения
echo -e "${GREEN}Создание виртуального окружения...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Виртуальное окружение уже существует. Используем его.${NC}"
else
    ${PYTHON} -m venv venv
    echo -e "${GREEN}Виртуальное окружение создано.${NC}"
fi

# Определение команды для активации виртуального окружения
if [ -f "venv/Scripts/activate" ]; then
    # Windows
    ACTIVATE="venv/Scripts/activate"
else
    # Linux/Mac
    ACTIVATE="venv/bin/activate"
fi

# Активация виртуального окружения
echo -e "${GREEN}Активация виртуального окружения...${NC}"
source "${ACTIVATE}"

# Проверка версии pip и обновление при необходимости
echo -e "${GREEN}Обновление pip...${NC}"
pip install --upgrade pip

# Установка зависимостей из requirements.txt
echo -e "${GREEN}Установка зависимостей из requirements.txt...${NC}"
pip install -r requirements.txt

# Создание директорий проекта, если они не существуют
echo -e "${GREEN}Создание директорий проекта...${NC}"
mkdir -p src
mkdir -p app
mkdir -p models
mkdir -p logs

# Проверка, есть ли файл __init__.py в директориях src и app
if [ ! -f "src/__init__.py" ]; then
    echo -e "${GREEN}Создание src/__init__.py...${NC}"
    echo '"""Инициализационный файл для пакета src"""' > src/__init__.py
    echo '__version__ = "1.0.0"' >> src/__init__.py
fi

if [ ! -f "app/__init__.py" ]; then
    echo -e "${GREEN}Создание app/__init__.py...${NC}"
    echo '"""Инициализационный файл для пакета app"""' > app/__init__.py
    echo 'from flask import Flask' >> app/__init__.py
    echo '' >> app/__init__.py
    echo 'def create_app():' >> app/__init__.py
    echo '    """Создает Flask-приложение"""' >> app/__init__.py
    echo '    app = Flask(__name__)' >> app/__init__.py
    echo '    return app' >> app/__init__.py
fi

# Загрузка NLTK данных
echo -e "${GREEN}Загрузка ресурсов NLTK...${NC}"
if [ -f "venv/bin/python" ]; then
    PYTHON_BIN="venv/bin/python"
else 
    PYTHON_BIN="venv/Scripts/python"
fi

"$PYTHON_BIN" -c "import nltk; nltk.download('stopwords')"

# Завершение настройки
echo -e "${GREEN}Настройка окружения завершена!${NC}"
echo -e "${YELLOW}Для активации виртуального окружения используйте: source ${ACTIVATE}${NC}"
echo -e "${YELLOW}Для запуска приложения используйте: python -m app.main${NC}"