"""
Точка входа приложения классификатора товаров
"""

import os
import sys
import logging
import joblib

# Добавляем текущую директорию в путь поиска модулей
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app import create_app
from app.config import get_config
from app.logging_config import setup_logging
from app.api import initialize
from src.preprocess import TextPreprocessor

# Настраиваем логирование
logger = setup_logging()

def load_model_and_vectorizer():
    """
    Загружает модель, векторизатор и кодировщик меток
    
    Returns:
        tuple: (модель, векторизатор, предобработчик)
    """
    config = get_config()
    
    # Пути к файлам
    model_path = config.get('MODEL_PATH')
    vectorizer_path = config.get('VECTORIZER_PATH')
    label_encoder_path = config.get('LABEL_ENCODER_PATH')
    
    # Проверяем наличие файлов
    if not os.path.exists(model_path):
        logger.warning(f"Модель не найдена по пути: {model_path}")
        model_trainer = None
    else:
        # Загружаем модель и векторизатор
        from src.models import ModelTrainer
        model_trainer, vectorizer = ModelTrainer.load(
            model_path, 
            label_encoder_path,
            vectorizer_path
        )
        logger.info("Модель загружена успешно")
    
    # Создаем предобработчик текста
    preprocessor = TextPreprocessor()
    
    return model_trainer, vectorizer, preprocessor

def main():
    """
    Основная функция приложения
    """
    # Загружаем модель и векторизатор
    model_trainer, vectorizer, preprocessor = load_model_and_vectorizer()
    
    # Создаем Flask-приложение
    from app import create_app
    app = create_app()
    
    # Инициализируем API
    initialize(model_trainer, vectorizer, preprocessor)
    
    # Получаем конфигурацию
    config = get_config()
    
    # Настраиваем параметры запуска
    host = config.get('HOST', '0.0.0.0')
    port = config.get('PORT', 5000)
    debug = config.get('DEBUG', False)
    
    # Запускаем приложение
    logger.info(f"Запуск приложения на {host}:{port}, debug={debug}")
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    main()