import streamlit as st
from PIL import Image
import threading
import uuid
from pathlib import Path
from app.detection import detect_objects

UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def run_detection(image_path, model_path, result_path, done_event):
    try:
        result_image = detect_objects(image_path, model_path)
        result_image.save(result_path)
    except Exception as e:
        st.session_state["detection_error"] = str(e)
    finally:
        done_event.set()

# интерфейс streamlit
st.title("Vehicle Detection with YOLO")

st.markdown("""
Приложение детектирует транспортные средства на аэрофотоснимках 
с помощью моделей YOLOv8, обученных на датасете VisDrone.

**Поддерживаемые классы:** легковые автомобили, микроавтобусы, грузовики, 
автобусы, мотоциклы, трёхколёсные велосипеды.

**Модели:**
- **Quick** — быстрая модель (YOLOv8n)
- **Accurate** — точная модель (YOLOv8s, mAP50 = 0.538)

Подробнее о проекте — в [репозитории на GitHub](https://github.com/qpimsslg/vehicle-detection)
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