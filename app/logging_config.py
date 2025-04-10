"""
Настройка логирования для приложения
"""

import os
import sys
import logging
import importlib
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# Используем относительные импорты для модулей в рамках пакета
from .config import get_config

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

# Проверяем доступность модуля python-json-logger
JSON_LOGGER_AVAILABLE = check_module_available('pythonjsonlogger')

def setup_logging():
    """
    Настраивает логирование для приложения
    
    Returns:
        Logger: Настроенный логгер
    """
    config = get_config()
    
    # Получаем уровень логирования из конфигурации
    log_level_str = config.get('LOG_LEVEL', 'INFO')
    log_level = getattr(logging, log_level_str.upper())
    
    # Получаем формат логирования
    log_format = config.get('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Создаем корневой логгер
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Очищаем существующие обработчики
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Создаем обработчик для вывода в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Определяем формат логов
    if config.get('ENVIRONMENT') == 'production' and JSON_LOGGER_AVAILABLE:
        # В production используем JSON-форматирование, если доступно
        from pythonjsonlogger import jsonlogger
        formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d',
            rename_fields={'levelname': 'level', 'pathname': 'path'}
        )
    else:
        # В других окружениях или если JSON-логгер недоступен, используем обычный текстовый формат
        formatter = logging.Formatter(log_format)
        
        if config.get('ENVIRONMENT') == 'production' and not JSON_LOGGER_AVAILABLE:
            logging.getLogger(__name__).warning(
                "python-json-logger не установлен, используется обычный формат логов"
            )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Если указан файл для логирования, добавляем обработчик файла
    log_file = config.get('LOG_FILE')
    if log_file:
        # Создаем директорию для лог-файла, если она не существует
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Создаем обработчик файла с ротацией
        if config.get('ENVIRONMENT') == 'production':
            # В production используем ротацию по времени (ежедневно)
            file_handler = TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=30,  # Хранить логи за 30 дней
                encoding='utf-8'
            )
        else:
            # В других окружениях используем ротацию по размеру
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 МБ
                backupCount=5,
                encoding='utf-8'
            )
        
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Отключаем логи от библиотек, которые слишком многословны
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Если включены логи доступа, настраиваем их
    if config.get('ENABLE_ACCESS_LOGS', True):
        werkzeug_logger = logging.getLogger('werkzeug')
        werkzeug_logger.setLevel(logging.INFO)
    
    return logger

def setup_request_logger():
    """
    Настраивает логгер для запросов
    
    Returns:
        Logger: Настроенный логгер для запросов
    """
    config = get_config()
    
    # Создаем отдельный логгер для запросов
    logger = logging.getLogger('request')
    logger.setLevel(logging.INFO)
    
    # Очищаем существующие обработчики
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Создаем обработчик для вывода в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Определяем формат логов
    if config.get('ENVIRONMENT') == 'production' and JSON_LOGGER_AVAILABLE:
        # В production используем JSON-форматирование, если доступно
        from pythonjsonlogger import jsonlogger
        formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(name)s %(levelname)s %(message)s %(method)s %(path)s %(status)s %(duration)s',
            rename_fields={'levelname': 'level'}
        )
    else:
        # В других окружениях или если JSON-логгер недоступен, используем обычный текстовый формат
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(method)s] %(path)s - %(status)s - %(duration).2fms'
        )
        
        if config.get('ENVIRONMENT') == 'production' and not JSON_LOGGER_AVAILABLE:
            logging.getLogger(__name__).warning(
                "python-json-logger не установлен, используется обычный формат логов"
            )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Если указан файл для логирования, добавляем обработчик файла
    log_file = config.get('REQUEST_LOG_FILE') or config.get('LOG_FILE')
    if log_file:
        # Если это тот же файл, что и для основных логов, добавляем суффикс
        if log_file == config.get('LOG_FILE'):
            log_dir = os.path.dirname(log_file)
            log_basename = os.path.basename(log_file)
            log_name, log_ext = os.path.splitext(log_basename)
            log_file = os.path.join(log_dir, f"{log_name}.request{log_ext}")
        
        # Создаем директорию для лог-файла, если она не существует
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Создаем обработчик файла с ротацией
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 МБ
            backupCount=5,
            encoding='utf-8'
        )
        
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name):
    """
    Получает настроенный логгер для модуля
    
    Args:
        name: Имя модуля
    
    Returns:
        Logger: Настроенный логгер
    """
    # Настраиваем корневой логгер, если он еще не настроен
    if not logging.getLogger().handlers:
        setup_logging()
    
    # Возвращаем логгер для указанного модуля
    return logging.getLogger(name)