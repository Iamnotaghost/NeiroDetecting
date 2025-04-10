"""
Инициализационный файл для пакета app
"""

from flask import Flask

# Определяем функцию для создания Flask-приложения
def create_app(config_name=None):
    """
    Фабричная функция для создания Flask-приложения
    
    Args:
        config_name: Имя конфигурации
    
    Returns:
        Flask: Экземпляр Flask-приложения
    """
    from app.config import get_flask_config
    from app.middleware import setup_middlewares
    from app.api import api_bp
    
    # Создаем приложение
    app = Flask(__name__)
    
    # Применяем конфигурацию
    app.config.update(get_flask_config())
    
    # Настраиваем middleware
    setup_middlewares(app)
    
    # Регистрируем Blueprint для API
    app.register_blueprint(api_bp, url_prefix='/api')
    
    return app