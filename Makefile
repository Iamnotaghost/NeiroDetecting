.PHONY: build run stop dev-env train docker-setup logs clean all

# Переменные
IMAGE_NAME = product-classifier
CONTAINER_NAME = product-classifier
DOCKER_COMPOSE = docker-compose

# Общая команда для сборки и запуска
all: build run

# Настройка среды разработки
dev-env:
	@echo "Настройка окружения разработки..."
	bash -c "bash setup.sh"

# Обучение модели
train:
	@echo "Запуск обучения модели..."
	bash -c "python -m src.train --data train_data.csv --model logreg --output models"

# Сборка Docker-образа
build:
	@echo "Сборка Docker-образа..."
	bash -c "$(DOCKER_COMPOSE) build"

# Запуск в Docker
run:
	@echo "Запуск контейнера в Docker..."
	bash -c "$(DOCKER_COMPOSE) up -d"

# Остановка контейнера
stop:
	@echo "Остановка контейнера..."
	bash -c "$(DOCKER_COMPOSE) down"

# Просмотр логов
logs:
	@echo "Просмотр логов контейнера..."
	bash -c "$(DOCKER_COMPOSE) logs -f classifier-api"

# Очистка Docker-ресурсов
clean:
	@echo "Очистка ресурсов Docker..."
	bash -c "$(DOCKER_COMPOSE) down -v"
	bash -c "docker rmi $(IMAGE_NAME) || true"

# Настройка Docker (установка Docker, если его нет)
docker-setup:
	@echo "Проверка наличия Docker..."
	@if ! command -v docker &> /dev/null; then \
		echo "Docker не установлен. Установка Docker..."; \
		bash -c "curl -fsSL https://get.docker.com -o get-docker.sh"; \
		bash -c "sudo sh get-docker.sh"; \
		bash -c "sudo usermod -aG docker $$USER"; \
		echo "Docker установлен. Пожалуйста, перезагрузите систему."; \
	else \
		echo "Docker уже установлен."; \
	fi

	@echo "Проверка наличия Docker Compose..."
	@if ! command -v docker-compose &> /dev/null; then \
		echo "Docker Compose не установлен. Установка Docker Compose..."; \
		bash -c "sudo curl -L \"https://github.com/docker/compose/releases/download/v2.15.1/docker-compose-$$(uname -s)-$$(uname -m)\" -o /usr/local/bin/docker-compose"; \
		bash -c "sudo chmod +x /usr/local/bin/docker-compose"; \
		echo "Docker Compose установлен."; \
	else \
		echo "Docker Compose уже установлен."; \
	fi