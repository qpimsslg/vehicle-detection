from typing import Annotated
from fastapi import FastAPI, File, UploadFile
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from starlette.responses import FileResponse
from .config import UPLOAD_DIR, MAX_FILE_SIZE, ALLOWED_TYPES
from .detection import detect_objects
from fastapi.staticfiles import StaticFiles


app = FastAPI()
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/")
async def root():
    return {"message": "Hello World"}


def validate_file_size(file: UploadFile):
    file_size = len(file.file.read())
    if file_size > MAX_FILE_SIZE * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File is too large.")

def save_file(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type.")

    # генерируем уникальное имя
    file_extension = Path(file.filename).suffix.lower()
    save_name = f"{uuid.uuid4().hex}{file_extension}"
    save_path = UPLOAD_DIR / save_name

    # сохраняем файл
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return save_path, save_name

@app.post("/files/")
async def upload_image(file: UploadFile = File(...)):
    validate_file_size(file)
    save_path, _ = save_file(file)
    return {"message": "Файл успешно загружен", "path": str(save_path)}

# эндпоинт для модели "быстрая/неточная"
@app.post("/quick")
async def quick(file: UploadFile = File(...)):
    try:
        # сохраняем файл на сервере
        image_path, image_name = save_file(file)

        # детекция
        model_path = Path("models/visdrone_yolo26n_50epochs.pt")
        result_image = detect_objects(image_path, model_path)

        # сохраняем результат
        result_image_path = UPLOAD_DIR / f"quick_result_{image_name}"
        result_image.save(result_image_path)

        return {
            "message": "Detection completed for quick model",
            "result_image": f"/result_image/{result_image_path.name}",
            "path": str(result_image_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

@app.post("/accurate")
async def accurate(file: UploadFile = File(...)):
    try:
        # сохраняем файл на сервере
        image_path, image_name = save_file(file)

        # детекция
        model_path = Path("models/visdrone_yolo26l_20epochs.pt")
        result_image = detect_objects(image_path, model_path)

        # сохраняем результат
        result_image_path = UPLOAD_DIR / f"accurate_result_{image_name}"
        result_image.save(result_image_path)

        return {"message": "Detection completed for accurate model", "result_image": str(result_image_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


@app.get("/result_image/{filename}")
async def get_result_image(filename: str):
    result_image_path = UPLOAD_DIR / filename

    if not result_image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(result_image_path)