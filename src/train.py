"""
Скрипт для обучения модели классификации товаров
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Добавляем корневую директорию проекта в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем модули из проекта
from src.preprocess import TextPreprocessor
from src.features import create_feature_pipeline
from src.models import ModelTrainer
from src.utils import save_pickle, save_json

def setup_logging():
    """
    Настраивает логирование для скрипта обучения
    
    Returns:
        Logger: Настроенный логгер
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('train.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """
    Парсит аргументы командной строки
    
    Returns:
        Namespace: Аргументы командной строки
    """
    parser = argparse.ArgumentParser(description='Обучение модели классификации товаров')
    
    parser.add_argument('--data', type=str, default='train_data.csv',
                        help='Путь к файлу с обучающими данными')
    parser.add_argument('--model', type=str, default='logreg',
                        choices=['logreg', 'rf', 'svm', 'gbm', 'ensemble'],
                        help='Тип модели для обучения')
    parser.add_argument('--output', type=str, default='models',
                        help='Директория для сохранения модели')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Доля тестовых данных')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Случайное зерно')
    
    return parser.parse_args()

def main():
    """
    Основная функция для обучения модели
    """
    # Настраиваем логирование
    logger = setup_logging()
    
    # Парсим аргументы командной строки
    args = parse_args()
    
    logger.info(f"Загрузка данных из файла: {args.data}")
    
    try:
        # Загружаем данные
        data = pd.read_csv(args.data, encoding='utf-8')
        
        # Проверяем структуру данных
        if 'Name' not in data.columns or 'Category' not in data.columns:
            if len(data.columns) >= 2:
                # Предполагаем, что первый столбец - название, второй - категория
                data.columns = ['Name', 'Category']
                logger.warning("Столбцы переименованы в 'Name' и 'Category'")
            else:
                logger.error("Некорректная структура данных, необходимы столбцы 'Name' и 'Category'")
                return
        
        # Очищаем данные
        data.dropna(inplace=True)
        
        # Разделяем данные на признаки и целевую переменную
        X = data['Name'].values
        y = data['Category'].values
        
        logger.info(f"Загружено {len(X)} образцов с {len(np.unique(y))} классами")
        
        # Разделяем данные на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        
        logger.info(f"Обучающая выборка: {len(X_train)} образцов, тестовая выборка: {len(X_test)} образцов")
        
        # Создаем предобработчик текста
        preprocessor = TextPreprocessor()
        
        # Предобрабатываем текст
        logger.info("Предобработка текста...")
        X_train_processed = preprocessor.process_batch(X_train)
        X_test_processed = preprocessor.process_batch(X_test)
        
        # Создаем извлекатель признаков
        logger.info("Создание и обучение извлекателя признаков...")
        feature_extractor = create_feature_pipeline()
        
        # Извлекаем признаки
        X_train_features = feature_extractor.fit_transform(X_train_processed)
        X_test_features = feature_extractor.transform(X_test_processed)
        
        logger.info(f"Размерность признакового пространства: {X_train_features.shape[1]}")
        
        # Создаем и обучаем модель
        logger.info(f"Обучение модели типа: {args.model}...")
        model_trainer = ModelTrainer(model_type=args.model)
        model_trainer.train(X_train_features, y_train)
        
        # Оцениваем модель
        logger.info("Оценка модели...")
        evaluation = model_trainer.evaluate(X_test_features, y_test)
        
        logger.info(f"Macro F1: {evaluation['macro_f1']:.4f}")
        logger.info(f"Отчет о классификации:\n{evaluation['report']}")
        
        # Создаем директорию для сохранения модели
        os.makedirs(args.output, exist_ok=True)
        
        # Сохраняем модель и векторизатор
        model_path = os.path.join(args.output, 'model.pkl')
        vectorizer_path = os.path.join(args.output, 'vectorizer.pkl')
        label_encoder_path = os.path.join(args.output, 'label_encoder.pkl')
        
        logger.info(f"Сохранение модели в: {model_path}")
        model_trainer.save(model_path, feature_extractor)
        
        # Сохраняем отчет
        report_path = os.path.join(args.output, 'evaluation_report.json')
        report_data = {
            'macro_f1': float(evaluation['macro_f1']),
            'model_type': args.model,
            'num_features': int(X_train_features.shape[1]),
            'num_samples': int(len(X_train)),
            'num_classes': int(len(np.unique(y_train)))
        }
        
        save_json(report_data, report_path)
        logger.info(f"Отчет сохранен в: {report_path}")
        
        logger.info("Обучение модели завершено успешно")
        
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()