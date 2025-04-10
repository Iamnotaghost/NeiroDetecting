"""
Определение маршрутов и обработчиков запросов API
"""

import time
import logging
import traceback
from flask import Blueprint, request, jsonify, make_response
from functools import wraps

# Используем относительные импорты для модулей в рамках пакета
from .config import get_config
from src.preprocess import TextPreprocessor
from src.utils import validate_text

# Создаем экземпляр Blueprint
api_bp = Blueprint('api', __name__)

# Получаем логгер
logger = logging.getLogger(__name__)

# Инициализируем глобальные переменные
model_trainer = None
vectorizer = None
preprocessor = None

# Исправленный декоратор, который сохраняет имя функции
def timing_decorator(func):
    @wraps(func)  # Сохраняет имя и другие атрибуты функции
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

def initialize(model, featurizer, text_processor):
    """
    Инициализирует глобальные переменные для использования в API
    
    Args:
        model: Модель для предсказаний
        featurizer: Векторизатор признаков
        text_processor: Предобработчик текста
    """
    global model_trainer, vectorizer, preprocessor
    
    model_trainer = model
    vectorizer = featurizer
    preprocessor = text_processor
    
    logger.info("API инициализировано")

@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Проверка работоспособности API
    
    Returns:
        JSON-ответ с информацией о статусе API
    """
    config = get_config()
    
    # Проверяем, что модель загружена
    if model_trainer is None or vectorizer is None or preprocessor is None:
        status = 'error'
        message = 'API не инициализировано'
    else:
        status = 'ok'
        message = 'API работает'
    
    return jsonify({
        'status': status,
        'message': message,
        'version': config.get('API_VERSION', 'unknown'),
        'environment': config.get('ENVIRONMENT', 'unknown')
    })

@api_bp.route('/predict', methods=['POST'])
@timing_decorator
def predict():
    """
    Предсказание категории товара по его названию
    
    Returns:
        JSON-ответ с предсказанной категорией
    """
    try:
        # Получаем данные из запроса
        data = request.get_json()
        
        if not data:
            return make_response(jsonify({
                'error': 'Не удалось получить данные из запроса'
            }), 400)
        
        if 'title' not in data:
            return make_response(jsonify({
                'error': 'В запросе отсутствует поле "title"'
            }), 400)
        
        title = data['title']
        
        # Валидация входных данных
        if not validate_text(title):
            return make_response(jsonify({
                'error': 'Название товара содержит недопустимые символы или слишком короткое'
            }), 400)
        
        # Предобработка текста
        processed_title = preprocessor.process(title)
        
        # Векторизация
        title_features = vectorizer.transform([processed_title])
        
        # Предсказание
        prediction = model_trainer.predict(title_features)[0]
        
        # Получаем уверенность модели (если доступно)
        confidence = None
        if hasattr(model_trainer.model, 'predict_proba'):
            proba = model_trainer.model.predict_proba(title_features)
            confidence = float(proba.max())
        
        # Формируем ответ
        response = {
            'category': prediction
        }
        
        # Добавляем уверенность, если доступна
        if confidence is not None:
            response['confidence'] = confidence
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Ошибка при выполнении предсказания: {e}")
        logger.error(traceback.format_exc())
        
        return make_response(jsonify({
            'error': f'Ошибка при выполнении предсказания: {str(e)}'
        }), 500)

@api_bp.route('/predict/batch', methods=['POST'])
@timing_decorator
def predict_batch():
    """
    Пакетное предсказание категорий товаров
    
    Returns:
        JSON-ответ с предсказанными категориями
    """
    try:
        # Получаем данные из запроса
        data = request.get_json()
        
        if not data:
            return make_response(jsonify({
                'error': 'Не удалось получить данные из запроса'
            }), 400)
        
        if 'titles' not in data:
            return make_response(jsonify({
                'error': 'В запросе отсутствует поле "titles"'
            }), 400)
        
        titles = data['titles']
        
        if not isinstance(titles, list):
            return make_response(jsonify({
                'error': 'Поле "titles" должно быть списком'
            }), 400)
        
        # Максимальное количество элементов в батче
        config = get_config()
        max_batch_size = config.get('MAX_BATCH_SIZE', 100)
        
        if len(titles) > max_batch_size:
            return make_response(jsonify({
                'error': f'Превышено максимальное количество элементов в пакетном запросе: {max_batch_size}'
            }), 400)
        
        # Фильтруем невалидные тексты
        valid_titles = []
        invalid_indices = []
        
        for i, title in enumerate(titles):
            if validate_text(title):
                valid_titles.append(title)
            else:
                invalid_indices.append(i)
        
        # Предобработка текстов
        processed_titles = preprocessor.process_batch(valid_titles)
        
        # Векторизация
        titles_features = vectorizer.transform(processed_titles)
        
        # Предсказание
        predictions = model_trainer.predict(titles_features)
        
        # Получаем уверенность модели (если доступно)
        confidences = None
        if hasattr(model_trainer.model, 'predict_proba'):
            probas = model_trainer.model.predict_proba(titles_features)
            confidences = probas.max(axis=1)
        
        # Формируем результат
        results = []
        valid_idx = 0
        
        for i in range(len(titles)):
            if i in invalid_indices:
                # Добавляем невалидный элемент с сообщением об ошибке
                results.append({
                    'title': titles[i],
                    'error': 'Название товара содержит недопустимые символы или слишком короткое'
                })
            else:
                # Добавляем валидный элемент с предсказанием
                result = {
                    'title': titles[i],
                    'category': predictions[valid_idx]
                }
                
                # Добавляем уверенность, если доступна
                if confidences is not None:
                    result['confidence'] = float(confidences[valid_idx])
                
                results.append(result)
                valid_idx += 1
        
        # Формируем ответ
        response = {
            'predictions': results
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Ошибка при выполнении пакетного предсказания: {e}")
        logger.error(traceback.format_exc())
        
        return make_response(jsonify({
            'error': f'Ошибка при выполнении пакетного предсказания: {str(e)}'
        }), 500)

@api_bp.route('/categories', methods=['GET'])
def get_categories():
    """
    Получение списка всех возможных категорий
    
    Returns:
        JSON-ответ со списком категорий
    """
    try:
        if model_trainer is None:
            return make_response(jsonify({
                'error': 'API не инициализировано'
            }), 500)
        
        # Получаем все категории из кодировщика меток
        categories = model_trainer.label_encoder.classes_.tolist()
        
        return jsonify({
            'categories': categories,
            'count': len(categories)
        })
    
    except Exception as e:
        logger.error(f"Ошибка при получении списка категорий: {e}")
        logger.error(traceback.format_exc())
        
        return make_response(jsonify({
            'error': f'Ошибка при получении списка категорий: {str(e)}'
        }), 500)