import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import time
import gdown
from pathlib import Path
import threading
import uuid
from app.detection import detect_objects
from app.config import UPLOAD_DIR

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


download_model()

# адрес FastAPI сервера
API_URL = "https://vehicle-detection-htl3.onrender.com"

def run_detection(image_path, model_path, result_path, done_event):
    try:
        result_image = detect_objects(image_path, model_path)
        result_image.save(result_path)
    except Exception as e:
        st.session_state["detection_error"] = str(e)
    finally:
        done_event.set()

# интерфейс streamlit
st.title("Object Detection with YOLO")

st.markdown("""
    Это приложение использует **YOLO модели** для классификации и детекции транспортных средств на изображениях.

    Для более подробной информации о:
    - Датасете,
    - Модели,
    - Классах детекции,

    а также для инструкций по использованию, пожалуйста, посетите [мой GitHub](https://github.com/qpimsslg/vehicle-detection).
""")


st.markdown("""
    Загрузите изображение и выберите модель для обработки:
""")

model_type = st.selectbox("Choose the model:", ("quick", "accurate"))

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # отображаем исходное изображение
    st.image(uploaded_file, caption="Uploaded Image", width='stretch')

    # загружаем изображение на сервер и обрабатываем
    if st.button("Detect Objects"):
        # сохраняем загруженный файл
        image_name = f"{uuid.uuid4().hex}.jpg"
        image_path = UPLOAD_DIR / image_name
        result_path = UPLOAD_DIR / f"{model_type}_result_{image_name}"

        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        model_path = Path(f"models/visdrone_yolo26_{model_type}.pt")

        # запускаем детекцию в отдельном потоке
        done_event = threading.Event()
        thread = threading.Thread(
            target=run_detection,
            args=(image_path, model_path, result_path, done_event)
        )
        thread.start()

        # ждём результата
        with st.spinner("Processing image, please wait..."):
            done_event.wait(timeout=120)

        if result_path.exists():
            result_image = Image.open(result_path)
            st.success("Detection complete!")
            st.image(result_image, caption="Processed Image with BBoxes")
        else:
            error = st.session_state.get("detection_error", "неизвестная ошибка")
            st.error(f"Detection error: {error}")