import re
import nltk
from nltk.corpus import stopwords
import unicodedata
from typing import List, Optional
import os
import logging
from tqdm import tqdm  # Для отображения прогресса

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Проверяем и скачиваем ресурсы NLTK безопасным способом
def download_nltk_resources():
    """Безопасно скачивает необходимые ресурсы NLTK"""
    try:
        nltk.data.find('corpora/stopwords')
        logger.info("NLTK stopwords уже загружены")
    except LookupError:
        try:
            # Пробуем использовать общий каталог
            nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')
            logger.info("NLTK stopwords успешно загружены в общий каталог")
        except PermissionError:
            try:
                # Если нет прав, пробуем использовать домашнюю директорию пользователя
                home_dir = os.path.expanduser("~")
                nltk_dir = os.path.join(home_dir, 'nltk_data')
                os.makedirs(nltk_dir, exist_ok=True)
                nltk.download('stopwords', download_dir=nltk_dir)
                logger.info(f"NLTK stopwords успешно загружены в {nltk_dir}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить NLTK stopwords: {e}")
                logger.warning("Стоп-слова не будут использоваться при обработке текста")
                return False
    return True

# Пытаемся загрузить ресурсы при импорте модуля
nltk_resources_available = download_nltk_resources()

# Простой стеммер для русского языка
class SimpleRussianStemmer:
    """Простой стеммер для русского языка"""
    
    def __init__(self):
        # Распространенные окончания русских слов
        self.common_endings = [
            'ый', 'ая', 'ое', 'ые', 'ого', 'ому', 'ым', 'ом', 'ой', 'ую', 'ых', 'ими', 'ыми',
            'ий', 'яя', 'ее', 'ие', 'его', 'ему', 'им', 'ем', 'ей', 'юю', 'их', 'ями', 'ими',
            'ть', 'ться', 'ет', 'ут', 'ют', 'ит', 'ат', 'ят', 'ешь', 'ете', 'ем', 'им', 'ишь',
            'у', 'ю', 'а', 'я', 'о', 'е', 'ы', 'и', 'ов', 'ев', 'ёв', 'ей'
        ]
        # Сортируем по длине (сначала длинные окончания)
        self.common_endings.sort(key=len, reverse=True)
        
    def stem(self, word):
        """Простая стемминг-функция для русского слова"""
        if not word or len(word) <= 3:
            return word
            
        # Проходим по распространенным окончаниям
        for ending in self.common_endings:
            if word.endswith(ending) and len(word) - len(ending) >= 3:
                return word[:-len(ending)]
                
        return word

class TextPreprocessor:
    def __init__(self, language: str = 'russian', batch_size: int = 1000):
        """
        Инициализирует предобработчик текста
        
        Args:
            language: Язык для стоп-слов ('russian', 'english')
            batch_size: Размер пакета для обработки
        """
        self.language = language
        self.batch_size = batch_size
        
        # Загружаем стоп-слова, если они доступны
        if nltk_resources_available:
            try:
                self.stop_words = set(stopwords.words(language))
                logger.info(f"Загружено {len(self.stop_words)} стоп-слов")
            except Exception as e:
                logger.warning(f"Ошибка при загрузке стоп-слов: {e}")
                self.stop_words = set()
        else:
            self.stop_words = set()
            
        # Инициализация простого стеммера вместо pymorphy2
        logger.info("Инициализация простого стеммера...")
        self.stemmer = SimpleRussianStemmer()
        logger.info("Стеммер инициализирован")
        
    def normalize_text(self, text: str) -> str:
        """
        Нормализует текст (приведение к нижнему регистру, 
        удаление специальных символов и т.д.)
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Замена нескольких пробелов одним
        text = re.sub(r'\s+', ' ', text)
        
        # Удаление специальных символов и чисел
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Нормализация Unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Удаление лишних пробелов
        text = text.strip()
        
        return text
    
    def remove_stop_words(self, text: str) -> str:
        """Удаляет стоп-слова из текста"""
        if not text or not self.stop_words:
            return text
        
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        
        return " ".join(filtered_words)
    
    def stem_text(self, text: str) -> str:
        """Стемминг текста с помощью простого стеммера"""
        if not text:
            return ""
        
        try:
            words = text.split()
            stemmed_words = [self.stemmer.stem(word) for word in words if word]
            
            return " ".join(stemmed_words)
        except Exception as e:
            logger.warning(f"Ошибка при стемминге текста: {e}")
            return text
    
    def process(self, text: str, 
                normalize: bool = True, 
                remove_stops: bool = True, 
                stem: bool = True) -> str:
        """
        Полная обработка текста
        
        Args:
            text: Исходный текст
            normalize: Применять нормализацию
            remove_stops: Удалять стоп-слова
            stem: Применять стемминг
        
        Returns:
            Обработанный текст
        """
        if not text:
            return ""
        
        if normalize:
            text = self.normalize_text(text)
            
        if remove_stops:
            text = self.remove_stop_words(text)
            
        if stem:
            text = self.stem_text(text)
            
        return text
    
    def process_batch(self, texts: List[str], **kwargs) -> List[str]:
        """
        Обработка списка текстов с улучшенным выводом прогресса
        """
        results = []
        total = len(texts)
        
        # Используем tqdm для отображения прогресса
        for i, text in enumerate(tqdm(texts, desc="Обработка текстов")):
            try:
                processed_text = self.process(text, **kwargs)
                results.append(processed_text)
                
                # Дополнительный вывод для отладки каждые 1000 записей
                if (i + 1) % 1000 == 0:
                    logger.info(f"Обработано {i+1}/{total} текстов ({(i+1)/total*100:.2f}%)")
            except Exception as e:
                logger.error(f"Ошибка при обработке текста #{i}: {e}")
                # Добавляем пустую строку вместо проблемного текста
                results.append("")
                
        return results
                
    def process_batch_in_chunks(self, texts: List[str], **kwargs) -> List[str]:
        """
        Обработка списка текстов небольшими пакетами для экономии памяти
        и более стабильной работы
        """
        all_results = []
        total = len(texts)
        
        # Разбиваем на пакеты по batch_size
        for chunk_start in range(0, total, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, total)
            logger.info(f"Обработка пакета {chunk_start//self.batch_size + 1} "
                      f"(записи {chunk_start+1}-{chunk_end} из {total})")
            
            # Получаем текущий пакет текстов
            chunk = texts[chunk_start:chunk_end]
            
            # Обрабатываем пакет
            chunk_results = []
            for i, text in enumerate(tqdm(chunk, desc=f"Пакет {chunk_start//self.batch_size + 1}")):
                try:
                    processed = self.process(text, **kwargs)
                    chunk_results.append(processed)
                except Exception as e:
                    logger.error(f"Ошибка при обработке текста #{chunk_start+i}: {e}")
                    chunk_results.append("")  # Добавляем пустую строку в случае ошибки
            
            # Добавляем результаты этого пакета к общим результатам
            all_results.extend(chunk_results)
            
            # Вывод прогресса
            logger.info(f"Общий прогресс: {chunk_end}/{total} текстов ({chunk_end/total*100:.2f}%)")
        
        return all_results