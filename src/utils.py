"""
Вспомогательные функции для работы с данными, моделями и т.д.
"""

import os
import re
import json
import time
import pickle
import logging
from typing import Dict, List, Any, Union, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def timing_decorator(func):
    """
    Декоратор для измерения времени выполнения функции
    
    Args:
        func: Функция для измерения времени выполнения
    
    Returns:
        Обернутая функция
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"Функция {func.__name__} выполнена за {execution_time:.4f} сек.")
        
        # Если результат - словарь, добавляем время выполнения
        if isinstance(result, dict):
            result['execution_time'] = f"{execution_time:.4f} сек."
        
        return result
    
    return wrapper

def safe_makedirs(directory: str) -> None:
    """
    Безопасно создает директорию, если она не существует
    
    Args:
        directory: Путь к директории
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Создана директория: {directory}")

def load_pickle(file_path: str) -> Any:
    """
    Загружает объект из pickle-файла
    
    Args:
        file_path: Путь к файлу
    
    Returns:
        Загруженный объект
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj: Any, file_path: str) -> None:
    """
    Сохраняет объект в pickle-файл
    
    Args:
        obj: Объект для сохранения
        file_path: Путь к файлу
    """
    directory = os.path.dirname(file_path)
    safe_makedirs(directory)
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    
    logger.info(f"Объект сохранен в файл: {file_path}")

def load_json(file_path: str) -> Dict:
    """
    Загружает данные из JSON-файла
    
    Args:
        file_path: Путь к файлу
    
    Returns:
        Данные из JSON-файла
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Dict, file_path: str) -> None:
    """
    Сохраняет данные в JSON-файл
    
    Args:
        data: Данные для сохранения
        file_path: Путь к файлу
    """
    directory = os.path.dirname(file_path)
    safe_makedirs(directory)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Данные сохранены в файл: {file_path}")

def validate_text(text: str) -> bool:
    """
    Проверяет текст на наличие полезного содержимого
    
    Args:
        text: Текст для проверки
    
    Returns:
        True, если текст содержит полезное содержимое, иначе False
    """
    if not text or not isinstance(text, str):
        return False
    
    # Удаляем пробелы и специальные символы
    cleaned_text = re.sub(r'\s+', '', text)
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    
    # Проверяем, что осталось не менее 2 символов
    return len(cleaned_text) >= 2

def batch_generator(data, batch_size=32):
    """
    Генератор батчей данных
    
    Args:
        data: Данные для разбиения на батчи
        batch_size: Размер батча
    
    Yields:
        Батч данных
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def convert_to_sparse(X_dense):
    """
    Конвертирует плотную матрицу в разреженную (если есть много нулей)
    
    Args:
        X_dense: Плотная матрица
    
    Returns:
        Разреженная матрица, если в ней много нулей, иначе исходная матрица
    """
    from scipy.sparse import csr_matrix
    
    # Пороговое значение для конвертации (если более 50% элементов - нули)
    threshold = 0.5
    
    if isinstance(X_dense, np.ndarray):
        # Проверяем долю нулевых элементов
        zero_ratio = np.sum(X_dense == 0) / X_dense.size
        
        if zero_ratio > threshold:
            # Конвертируем в разреженную матрицу
            return csr_matrix(X_dense)
    
    # Возвращаем исходную матрицу
    return X_dense