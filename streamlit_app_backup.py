import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import time

# адрес FastAPI сервера
API_URL = "https://vehicle-detection-htl3.onrender.com"

# пробуждаем Render если он спит
try:
    requests.get(f"{API_URL}/docs", timeout=60)
except:
    pass

# функция для загрузки изображения через POST запрос на сервер
def upload_image(file, model_type):
    url = f"{API_URL}/predict"

    # указываем MIME-тип при отправке файла
    files = {"file": (file.name, file, file.type)}
    response = requests.post(url, files=files, data={"model_type": model_type})

    if response.status_code != 200:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None  # None, если ошибка

    return response.json()


# функция для получения обработанного изображения
def get_result_image(filename, retries=10, delay=2):
    url = f"{API_URL}/result_image/{filename}"
    for attempt in range(retries):
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        time.sleep(delay)
    return None

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
        # отправка на FastAPI сервер
        st.info("Processing image, please wait...")
        response = upload_image(uploaded_file, model_type)

        # печатаем сообщение сервера
        if response:
            st.success(response["message"])

            # получаем обработанное изображение
            result_image_url = response["result_image"]
            result_image = get_result_image(result_image_url.split("/")[-1], retries=20, delay=5)

            # отображаем результат
            #st.image(result_image, caption="Processed Image with BBoxes", width='stretch')

            if result_image:
                st.success("Detection complete!")
                st.image(result_image, caption="Processed Image with BBoxes", width='stretch')
            else:
                st.error("Processing took too long. Please try again.")