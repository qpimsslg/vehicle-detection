import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pathlib import Path
from starlette.responses import FileResponse
from .config import UPLOAD_DIR, MAX_FILE_SIZE, ALLOWED_TYPES
from fastapi.staticfiles import StaticFiles
from .queue import add_to_queue, start_workers
import gdown

def download_model():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    accurate_path = models_dir / "visdrone_yolo26_accurate.pt"
    quick_path = models_dir / "visdrone_yolo26_quick.pt"

    model_accurate_url = "https://drive.google.com/uc?id=1o8GU2OBRgyJgZYCn8LSmqKtUPJ7n1S4X"
    model_quick_url = "https://drive.google.com/uc?id=1ReB3Elbp0DJSqZ162E3dIy334aFLQC0z"

    if not accurate_path.exists():
        gdown.download(model_accurate_url, str(accurate_path), quiet=False)

    if not quick_path.exists():
        gdown.download(model_quick_url, str(quick_path), quiet=False)


app = FastAPI()
download_model()
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

start_workers(num_threads=4)

def validate_file_size(file: UploadFile):
    file_size = len(file.file.read())
    if file_size > MAX_FILE_SIZE * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File is too large.")
    file.file.seek(0)

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


@app.post("/predict")
async def predict(file: UploadFile = File(...), model_type: str = Form(...)):
    try:
        # сохраняем файл на сервере
        file_path, image_name = save_file(file)

        # добавляем задачу в очередь
        add_to_queue(file_path, model_type, image_name)

        return {
            "message": f"Detection started for {model_type} model",
            "status": "Processing",
            "result_image": f"/result_image/{model_type}_result_{image_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/result_image/{filename}")
async def get_result_image(filename: str):
    result_image_path = UPLOAD_DIR / filename

    if not result_image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(result_image_path)