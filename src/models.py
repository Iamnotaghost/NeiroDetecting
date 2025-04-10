"""
Модели машинного обучения для классификации товаров
(Упрощенная версия без CatBoost и LightGBM)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List, Tuple, Optional
import joblib
import os
import logging
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, classification_report

# Импортируем модели только из scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Класс для обучения и оценки моделей"""
    
    def __init__(self, model_type: str = 'logreg', model_params: Optional[Dict] = None):
        """
        Инициализирует тренер моделей
        
        Args:
            model_type: Тип модели ('logreg', 'rf', 'gbm', 'svm', 'ensemble')
            model_params: Параметры модели
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def get_model(self) -> BaseEstimator:
        """Создает экземпляр модели в зависимости от типа"""
        if self.model_type == 'logreg':
            params = {
                'C': self.model_params.get('C', 1.0),
                'solver': self.model_params.get('solver', 'liblinear'),
                'max_iter': self.model_params.get('max_iter', 1000),
                'class_weight': self.model_params.get('class_weight', 'balanced'),
                'random_state': 42
            }
            return LogisticRegression(**params)
            
        elif self.model_type == 'rf':
            params = {
                'n_estimators': self.model_params.get('n_estimators', 200),
                'max_depth': self.model_params.get('max_depth', 20),
                'min_samples_split': self.model_params.get('min_samples_split', 2),
                'min_samples_leaf': self.model_params.get('min_samples_leaf', 2),
                'class_weight': self.model_params.get('class_weight', 'balanced'),
                'random_state': 42,
                'n_jobs': -1
            }
            return RandomForestClassifier(**params)
            
        elif self.model_type == 'svm':
            params = {
                'C': self.model_params.get('C', 1.0),
                'loss': self.model_params.get('loss', 'squared_hinge'),
                'random_state': 42,
                'max_iter': self.model_params.get('max_iter', 1000),
                'class_weight': self.model_params.get('class_weight', 'balanced')
            }
            return LinearSVC(**params)
            
        elif self.model_type == 'gbm':
            # Используем GradientBoostingClassifier вместо LightGBM
            params = {
                'n_estimators': self.model_params.get('n_estimators', 200),
                'learning_rate': self.model_params.get('learning_rate', 0.05),
                'max_depth': self.model_params.get('max_depth', 7),
                'min_samples_split': self.model_params.get('min_samples_split', 2),
                'random_state': 42
            }
            return GradientBoostingClassifier(**params)
            
        elif self.model_type == 'ensemble':
            # Ансамбль из нескольких моделей (реализация ниже)
            return EnsembleClassifier(
                models=[
                    ('logreg', LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)),
                    ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced')),
                    ('gbm', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42))
                ]
            )
            
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Обучает модель
        
        Args:
            X_train: Обучающие данные
            y_train: Целевые метки для обучения
            X_val: Валидационные данные (опционально)
            y_val: Целевые метки для валидации (опционально)
        """
        # Кодируем метки классов
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Создаем модель
        self.model = self.get_model()
        
        # Обучаем модель
        if hasattr(self.model, 'fit'):
            if X_val is not None and y_val is not None:
                y_val_encoded = self.label_encoder.transform(y_val)
                self.model.fit(X_train, y_train_encoded)
            else:
                self.model.fit(X_train, y_train_encoded)
    
    def predict(self, X):
        """
        Выполняет предсказание для данных
        
        Args:
            X: Данные для предсказания
        
        Returns:
            Предсказанные метки классов
        """
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        # Делаем предсказания
        y_pred_encoded = self.model.predict(X)
        
        # Декодируем метки классов
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def evaluate(self, X, y):
        """
        Оценивает модель
        
        Args:
            X: Данные для оценки
            y: Истинные метки классов
            
        Returns:
            Отчет о классификации и macro-F1 метрика
        """
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        # Кодируем истинные метки
        y_encoded = self.label_encoder.transform(y)
        
        # Предсказываем метки
        y_pred_encoded = self.model.predict(X)
        
        # Вычисляем метрики
        macro_f1 = f1_score(y_encoded, y_pred_encoded, average='macro')
        report = classification_report(y_encoded, y_pred_encoded, 
                                       target_names=self.label_encoder.classes_)
        
        return {
            'macro_f1': macro_f1,
            'report': report
        }
    
    def save(self, model_path, vectorizer=None):
        """
        Сохраняет модель и кодировщик меток
        
        Args:
            model_path: Путь для сохранения модели
            vectorizer: Векторизатор для сохранения (опционально)
        """
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Сохраняем модель
        joblib.dump(self.model, model_path)
        
        # Сохраняем кодировщик меток
        label_encoder_path = os.path.join(
            os.path.dirname(model_path), 
            'label_encoder.pkl'
        )
        joblib.dump(self.label_encoder, label_encoder_path)
        
        # Сохраняем векторизатор, если он передан
        if vectorizer is not None:
            vectorizer_path = os.path.join(
                os.path.dirname(model_path), 
                'vectorizer.pkl'
            )
            joblib.dump(vectorizer, vectorizer_path)
    
    @classmethod
    def load(cls, model_path, label_encoder_path=None, vectorizer_path=None):
        """
        Загружает модель и кодировщик меток
        
        Args:
            model_path: Путь к сохраненной модели
            label_encoder_path: Путь к сохраненному кодировщику меток
            vectorizer_path: Путь к сохраненному векторизатору
            
        Returns:
            Экземпляр класса ModelTrainer с загруженной моделью и векторизатор (если указан)
        """
        # Определяем пути, если не указаны
        if label_encoder_path is None:
            label_encoder_path = os.path.join(
                os.path.dirname(model_path), 
                'label_encoder.pkl'
            )
        
        # Создаем экземпляр класса
        trainer = cls()
        
        # Загружаем модель
        trainer.model = joblib.load(model_path)
        
        # Загружаем кодировщик меток
        trainer.label_encoder = joblib.load(label_encoder_path)
        
        # Загружаем векторизатор, если путь указан
        vectorizer = None
        if vectorizer_path is not None:
            vectorizer = joblib.load(vectorizer_path)
        
        return trainer, vectorizer

class EnsembleClassifier(BaseEstimator):
    """Ансамбль моделей с мягким голосованием"""
    
    def __init__(self, models):
        """
        Инициализирует ансамбль моделей
        
        Args:
            models: Список кортежей (имя, модель)
        """
        self.models = models
        self.model_dict = {name: model for name, model in models}
        
    def fit(self, X, y):
        """Обучает все модели в ансамбле"""
        for _, model in self.models:
            model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """
        Предсказывает вероятности классов
        
        Args:
            X: Данные для предсказания
            
        Returns:
            Усредненные вероятности классов
        """
        # Получаем предсказания вероятностей от каждой модели
        probas = []
        for _, model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
            else:
                # Для моделей без predict_proba (например, LinearSVC)
                # используем decision_function
                decision = model.decision_function(X)
                # Преобразуем в вероятности с помощью softmax
                proba = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
            probas.append(proba)
        
        # Усредняем вероятности
        avg_proba = np.mean(probas, axis=0)
        return avg_proba
    
    def predict(self, X):
        """
        Предсказывает метки классов
        
        Args:
            X: Данные для предсказания
            
        Returns:
            Предсказанные метки классов
        """
        # Получаем вероятности классов
        avg_proba = self.predict_proba(X)
        
        # Выбираем класс с наибольшей вероятностью
        return np.argmax(avg_proba, axis=1)

def hyperparameter_search(X, y, model_type, param_grid, cv=5):
    """
    Выполняет поиск гиперпараметров
    
    Args:
        X: Данные для обучения
        y: Целевые метки
        model_type: Тип модели
        param_grid: Сетка параметров
        cv: Количество фолдов для кросс-валидации
    
    Returns:
        Лучшие параметры и результат
    """
    # Создаем пустую модель соответствующего типа
    trainer = ModelTrainer(model_type)
    model = trainer.get_model()
    
    # Создаем стратифицированные фолды
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Создаем объект GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_splitter,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    # Выполняем поиск
    grid_search.fit(X, y)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }