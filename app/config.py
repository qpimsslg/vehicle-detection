import os
from pathlib import Path

# Путь для загрузок — будет настраиваться через переменные окружения
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "5"))  # максимальный размер файла в MB
ALLOWED_TYPES = {"image/jpeg","image/jpg", "image/png", "image/webp"}

# Пример добавления других конфигов (например, логирование, security и т.д.)
