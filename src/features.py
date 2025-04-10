import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Any, Union

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """Извлекает статистические признаки из текста"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """
        Извлекает признаки на основе длины текста
    
        Args:
            X: Список текстов
    
        Returns:
            Матрица признаков [n_samples, n_features]
        """
        # Убедимся, что X - это список или одномерный массив
        if isinstance(X, str):
            X = [X]  # Если передана строка, обернем её в список
    
        # Создаем массив для хранения признаков
        features = np.zeros((len(X), 3))
    
        for i, text in enumerate(X):
            if not text or not isinstance(text, str):
                continue
            
            # Длина текста
            features[i, 0] = len(text)
        
            # Количество слов
            features[i, 1] = len(text.split())
        
            # Средняя длина слова
            words = text.split()
            if words:
                features[i, 2] = sum(len(word) for word in words) / len(words)
        return features
    
class WordPatternExtractor(BaseEstimator, TransformerMixin):
    """Извлекает признаки на основе шаблонов в словах"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """
        Извлекает признаки на основе длины текста

        Args:
            X: Список текстов
    
        Returns:
            Матрица признаков [n_samples, n_features]
        """
        # Убедимся, что X - это список или одномерный массив
        if isinstance(X, str):
            X = [X]  # Если передана строка, обернем её в список
    
        # Создаем массив для хранения признаков
        features = np.zeros((len(X), 3))
    
        for i, text in enumerate(X):
            if not text or not isinstance(text, str):
                continue
            
            # Длина текста
            features[i, 0] = len(text)
        
            # Количество слов
            features[i, 1] = len(text.split())
        
            # Средняя длина слова
            words = text.split()
            if words:
                features[i, 2] = sum(len(word) for word in words) / len(words)
            
        return features

def create_feature_pipeline(min_df=1, max_df=1.0, ngram_range=(1, 3), max_features=10000):
    """
    Создает пайплайн для извлечения признаков
    
    Args:
        min_df: Минимальная частота документа для TF-IDF
        max_df: Максимальная частота документа для TF-IDF
        ngram_range: Диапазон n-грамм
        max_features: Максимальное количество признаков
    
    Returns:
        Пайплайн для извлечения признаков
    """
    tfidf = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True
    )
    
    count_vec = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 1),
        max_features=max_features // 2
    )   
    
    features = FeatureUnion([
        ('tfidf', tfidf),
        ('count_vec', count_vec),
        ('text_length', TextLengthExtractor()),
        ('word_pattern', WordPatternExtractor())
    ])
    
    return features