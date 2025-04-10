"""
Middleware для Flask-приложения
"""

import time
import uuid
import logging
import importlib
from functools import wraps
from flask import request, g, Flask
from werkzeug.exceptions import HTTPException

# Используем относительные импорты для модулей в рамках пакета
from .config import get_config
from .logging_config import setup_request_logger

# Получаем логгер для запросов
request_logger = setup_request_logger()

# Проверяем доступность дополнительных модулей
def check_module_available(module_name):
    """
    Проверяет доступность модуля
    
    Args:
        module_name: Имя модуля
    
    Returns:
        bool: True, если модуль доступен, иначе False
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

# Проверяем доступность модулей
FLASK_CORS_AVAILABLE = check_module_available('flask_cors')
PROMETHEUS_AVAILABLE = check_module_available('prometheus_client')
FLASK_CACHING_AVAILABLE = check_module_available('flask_caching')

def setup_middlewares(app: Flask) -> None:
    """
    Настраивает middleware для приложения
    
    Args:
        app: Flask-приложение
    """
    logger = logging.getLogger(__name__)
    
    # Регистрируем before_request и after_request хуки
    app.before_request(log_request_start)
    app.after_request(log_request_end)
    
    # Настраиваем дополнительные middleware в зависимости от конфигурации
    config = get_config()
    
    # Включаем CORS, если указано в конфигурации
    if config.get('ENABLE_CORS', True):
        if FLASK_CORS_AVAILABLE:
            setup_cors(app)
        else:
            logger.warning("flask-cors не установлен, CORS не будет настроен")
    
    # Включаем мониторинг, если указано в конфигурации
    if config.get('ENABLE_METRICS', False):
        if PROMETHEUS_AVAILABLE:
            setup_metrics(app)
        else:
            logger.warning("prometheus-client не установлен, метрики не будут собираться")
    
    # Включаем кэширование, если указано в конфигурации
    if config.get('ENABLE_CACHING', False):
        if FLASK_CACHING_AVAILABLE:
            setup_caching(app)
        else:
            logger.warning("flask-caching не установлен, кэширование не будет настроено")

def log_request_start():
    """
    Логирует начало обработки запроса
    
    Эта функция вызывается перед обработкой каждого запроса
    """
    # Генерируем уникальный идентификатор запроса
    g.request_id = str(uuid.uuid4())
    
    # Сохраняем время начала обработки запроса
    g.start_time = time.time()
    
    # Сохраняем информацию о запросе
    g.request_info = {
        'method': request.method,
        'path': request.path,
        'remote_addr': request.remote_addr,
        'user_agent': request.user_agent.string if request.user_agent else None
    }

def log_request_end(response):
    """
    Логирует окончание обработки запроса
    
    Args:
        response: Ответ на запрос
    
    Returns:
        Исходный ответ
    """
    # Вычисляем время обработки запроса
    if hasattr(g, 'start_time'):
        duration = (time.time() - g.start_time) * 1000  # в миллисекундах
        
        # Получаем статус-код ответа
        status_code = response.status_code
        
        # Проверяем, что у нас есть вся необходимая информация
        if hasattr(g, 'request_id') and hasattr(g, 'request_info'):
            # Формируем информацию для логирования
            log_data = {
                'request_id': g.request_id,
                'method': g.request_info['method'],
                'path': g.request_info['path'],
                'status': status_code,
                'duration': duration,
                'remote_addr': g.request_info['remote_addr'],
                'user_agent': g.request_info['user_agent']
            }
            
            # Логируем запрос с помощью специального логгера
            request_logger.info(
                f"Request completed: {log_data['method']} {log_data['path']} {log_data['status']} {log_data['duration']:.2f}ms",
                extra=log_data
            )
            
            # Добавляем Request ID в заголовок ответа
            response.headers['X-Request-ID'] = g.request_id
    
    return response

def setup_cors(app: Flask) -> None:
    """
    Настраивает CORS для приложения
    
    Args:
        app: Flask-приложение
    """
    if not FLASK_CORS_AVAILABLE:
        return
    
    from flask_cors import CORS
    
    config = get_config()
    
    CORS(
        app,
        resources={r"/*": {"origins": config.get('CORS_ORIGINS', '*')}},
        supports_credentials=config.get('CORS_CREDENTIALS', False)
    )
    
    logging.getLogger(__name__).info("CORS middleware enabled")

def setup_metrics(app: Flask) -> None:
    """
    Настраивает сбор метрик для приложения
    
    Args:
        app: Flask-приложение
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    from prometheus_client import Counter, Histogram, start_http_server
    
    # Получаем конфигурацию
    config = get_config()
    
    # Определяем метрики
    app.http_requests_total = Counter(
        'http_requests_total',
        'Total HTTP Requests',
        ['method', 'endpoint', 'status']
    )
    
    app.http_request_duration_seconds = Histogram(
        'http_request_duration_seconds',
        'HTTP Request Duration in seconds',
        ['method', 'endpoint', 'status']
    )
    
    # Создаем before_request и after_request хуки для сбора метрик
    @app.before_request
    def start_timer():
        g.start_time = time.time()
    
    # Исправленная версия функции record_metrics в middleware.py
    @app.after_request
    def record_metrics(response):
    # Получаем время обработки запроса
        duration = time.time() - g.start_time
    
        # Определяем endpoint (убираем переменные пути)
        endpoint = request.path
    
        # Используем правильный метод для поиска соответствия маршрута
        if request.endpoint:
            for rule in app.url_map.iter_rules():
                if rule.endpoint == request.endpoint:
                    endpoint = rule.rule
                    break
    
        # Инкрементируем счетчик запросов
        app.http_requests_total.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code
        ).inc()
    
        # Записываем время обработки запроса
        app.http_request_duration_seconds.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code
        ).observe(duration)
    
        return response
    
    # Запускаем HTTP-сервер для метрик
    metrics_port = config.get('METRICS_PORT', 9090)
    start_http_server(metrics_port)
    
    logging.getLogger(__name__).info(f"Metrics server started on port {metrics_port}")

def setup_caching(app: Flask) -> None:
    """
    Настраивает кэширование для приложения
    
    Args:
        app: Flask-приложение
    """
    if not FLASK_CACHING_AVAILABLE:
        return
    
    from flask_caching import Cache
    
    config = get_config()
    
    # Настройки кэша
    cache_config = {
        'CACHE_TYPE': 'SimpleCache',
        'CACHE_DEFAULT_TIMEOUT': config.get('CACHE_TTL', 3600)
    }
    
    # Создаем кэш
    cache = Cache(app, config=cache_config)
    
    # Сохраняем кэш в приложении
    app.cache = cache
    
    logging.getLogger(__name__).info("Caching middleware enabled")

def cache_middleware(timeout=None):
    """
    Декоратор для кэширования ответов API
    
    Args:
        timeout: Время жизни кэша в секундах
    
    Returns:
        Декорированная функция
    """
    def decorator(func):
        @wraps(func)  # Добавляем wraps для сохранения имени функции
        def wrapper(*args, **kwargs):
            # Импортируем current_app, чтобы получить доступ к приложению
            from flask import current_app
            
            # Получаем конфигурацию
            config = get_config()
            
            # Проверяем, включено ли кэширование и установлен ли flask-caching
            if not FLASK_CACHING_AVAILABLE or not config.get('ENABLE_CACHING', False):
                return func(*args, **kwargs)
            
            # Проверяем, есть ли у приложения кэш
            if not hasattr(current_app, 'cache'):
                return func(*args, **kwargs)
            
            # Генерируем ключ кэша
            cache_key = f"{request.path}:{request.query_string.decode()}"
            
            # Получаем данные из кэша
            cached_data = current_app.cache.get(cache_key)
            
            # Если данные есть в кэше, возвращаем их
            if cached_data is not None:
                return cached_data
            
            # Иначе выполняем функцию и кэшируем результат
            result = func(*args, **kwargs)
            
            # Определяем время жизни кэша
            cache_timeout = timeout or config.get('CACHE_TTL', 3600)
            
            # Кэшируем результат
            current_app.cache.set(cache_key, result, timeout=cache_timeout)
            
            return result
        return wrapper
    return decorator