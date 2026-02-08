from ultralytics import YOLO
from PIL import Image, ImageDraw
from pathlib import Path

# Загрузка модели YOLO
def load_model(model_path: Path):
    model = YOLO(str(model_path))  # Загружаем модель YOLO
    return model

# Функция для детекции объектов на изображении
def detect_objects(image_path: Path, model_path: Path):
    model = load_model(model_path)  # Загружаем модель YOLO
    results = model(str(image_path))  # Запуск инференса на изображении

    # Метод .plot() возвращает изображение с отрисованными bounding box'ами в формате BGR
    im_bgr = results[0].plot()  # Берем только первое изображение из batch
    im_rgb = im_bgr[..., ::-1]  # Преобразуем BGR в RGB (для отображения)

    result_image = Image.fromarray(im_rgb)  # Преобразуем в формат PIL, чтобы сохранить

    return result_image
