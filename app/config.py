"""
Настройки конфигурации приложения
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

# Загружаем переменные окружения из .env файла (если он существует)
load_dotenv()

def get_config() -> Dict[str, Any]:
    """
    Получает конфигурацию приложения из переменных окружения
    
    Returns:
        Словарь с настройками конфигурации
    """
    # Определяем окружение (development, testing, production)
    environment = os.environ.get('ENVIRONMENT', 'development')
    
    # Базовая конфигурация для всех окружений
    config = {
        'ENVIRONMENT': environment,
        'API_VERSION': os.environ.get('API_VERSION', '1.0.0'),
        'DEBUG': os.environ.get('DEBUG', 'False').lower() == 'true',
        'PORT': int(os.environ.get('PORT', 5000)),
        
        # Пути к моделям и данным
        'MODEL_PATH': os.environ.get('MODEL_PATH', 'models/model.pkl'),
        'VECTORIZER_PATH': os.environ.get('VECTORIZER_PATH', 'models/vectorizer.pkl'),
        'LABEL_ENCODER_PATH': os.environ.get('LABEL_ENCODER_PATH', 'models/label_encoder.pkl'),
        
        # Настройки API
        'MAX_BATCH_SIZE': int(os.environ.get('MAX_BATCH_SIZE', 100)),
        'REQUEST_TIMEOUT': int(os.environ.get('REQUEST_TIMEOUT', 30)),
        'ENABLE_CACHING': os.environ.get('ENABLE_CACHING', 'False').lower() == 'true',
        'CACHE_TTL': int(os.environ.get('CACHE_TTL', 3600)),  # время жизни кэша в секундах
        
        # Настройки логирования
        'LOG_LEVEL': os.environ.get('LOG_LEVEL', 'INFO'),
        'LOG_FORMAT': os.environ.get(
            'LOG_FORMAT', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ),
        'LOG_FILE': os.environ.get('LOG_FILE', ''),
        'ENABLE_ACCESS_LOGS': os.environ.get('ENABLE_ACCESS_LOGS', 'True').lower() == 'true',
        
        # Настройки мониторинга
        'ENABLE_METRICS': os.environ.get('ENABLE_METRICS', 'False').lower() == 'true',
        'METRICS_PORT': int(os.environ.get('METRICS_PORT', 9090)),
    }
    
    # Дополнительные настройки для разных окружений
    if environment == 'development':
        config.update({
            'DEBUG': True,
            'ENABLE_ACCESS_LOGS': True,
        })
    elif environment == 'testing':
        config.update({
            'DEBUG': True,
            'ENABLE_ACCESS_LOGS': False,
        })
    elif environment == 'production':
        config.update({
            'DEBUG': False,
            'ENABLE_ACCESS_LOGS': True,
            'ENABLE_METRICS': True,
        })
    
    return config

def get_flask_config() -> Dict[str, Any]:
    """
    Получает конфигурацию для Flask-приложения
    
    Returns:
        Словарь с настройками конфигурации Flask
    """
    config = get_config()
    
    return {
        'DEBUG': config['DEBUG'],
        'TESTING': config['ENVIRONMENT'] == 'testing',
        'SECRET_KEY': os.environ.get('SECRET_KEY', 'default-secret-key'),
        'JSON_SORT_KEYS': False,  # Не сортировать ключи в JSON-ответах
        'JSONIFY_PRETTYPRINT_REGULAR': config['DEBUG'],  # Форматирование JSON в режиме отладки
        'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # Максимальный размер запроса (16 МБ)
        'TRAP_HTTP_EXCEPTIONS': False,
        'PRESERVE_CONTEXT_ON_EXCEPTION': config['DEBUG'],
    }