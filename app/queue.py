import queue
import threading
from .detection import detect_objects
from pathlib import Path
from .config import UPLOAD_DIR

# очередь для хранения задач
task_queue = queue.Queue()
# список рабочих потоков
workers = []

# функция для работы с очередью
def worker():
    while True:
        file_path, model_type, image_name = task_queue.get()

        try:
            # выполняем детекцию на изображении
            model_path = Path(f"models/visdrone_yolo26_{model_type}.pt")
            result_image = detect_objects(Path(file_path), model_path)

            result_image_path = UPLOAD_DIR /f"{model_type}_result_{image_name}"
            result_image.save(result_image_path)

            print(f"Processed image saved at: {result_image_path}")
            task_queue.task_done()

        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
            task_queue.task_done()


# запуск рабочих потоков
def start_workers(num_threads=4):
    global workers
    for _ in range(num_threads):
        thread = threading.Thread(target=worker, daemon=True)
        workers.append(thread)
        thread.start()


# функция для добавления задач в очередь
def add_to_queue(file_path, model_type, image_name):
    task_queue.put((file_path, model_type, image_name))
