from flask import Flask, jsonify, request
from PIL import Image
import time
import numpy as np # Импортируем библиотеку numpy
import requests
import json
import sys

from tensorflow.keras.models import Model # Импортируем модели keras: Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, concatenate, Activation, MaxPooling2D, Conv2D, BatchNormalization, Flatten, Dense, Dropout, Conv2DTranspose, Concatenate, Reshape
#from tensorflow.keras.optimizers import Adam # Импортируем оптимизатор Adam
from tensorflow.keras.models import model_from_json


# Загружаем нашу обученную нейросеть
json_file = open("model/digitals-model.json", "r")
model_json = json_file.read()
json_file.close()
# Восстанавливаем нашу модель из данных, которые хранятся в файле
model = model_from_json(model_json)
# Загружаем веса модели
model.load_weights('model/digitals-weights.hdf5')
model.summary()

# Создаем наше веб-приложение
app = Flask(__name__)

# Функция проверки формата изображения - альбомный или книжный.
# Возвращаемые значения: True - альбомный, False - книжный
def format_is_album(w: int, h: int):
    return True if w > h else False


# Функция умной обрезки
def smart_trimming(img):
    img_w, img_h = img.size   # Берём размер картинки
    target_size = [28, 28]  # Размер, который мы должны получить в результате

    # Проверка на формат
    if format_is_album(img_w, img_h):
        new_h = target_size[1]                # Делаем высоту основной осью
        new_w = round(new_h / img_h * img_w)  # Считаем ширину
    else:
        new_w = target_size[0]                # Делаем ширину основной осью
        new_h = round(new_w / img_w * img_h)  # Считаем высоту

    # Применяем наши измениния касательно размеров
    img = img.resize((new_w, new_h), Image.ANTIALIAS)

    # Находим центр картинки
    center = [new_w//2, new_h//2]
    # Находим левую верхнюю и правую нижнюю точки для прямоугольной обрезки
    top_left = [center[0] - target_size[0]//2, center[1] - target_size[1]//2]
    bottom_right = [center[0] + target_size[0]//2, center[1] + target_size[1]//2]

    # Обрезаем изображение
    img = img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

    return img




@app.route('/')
def index():
    index = open("static/index.html", "r")
    page = index.read()
    index.close()
    return page



@app.route('/predict_image', methods=['POST'])
def predictImage():
    data = {'status': 'error'}
    image = request.files['image']
    key = request.form.get('key')

    if image:
        print()
        print("Начинаем оработку картинки")
        imageFile = 'static/images/upload_image.jpg'
        image.save(imageFile)

        # Уменьшаем и обрезаем изображение (пока оно у нас в виде объекта Pillow)
        image_crop = Image.open(imageFile).convert('L')
        image_crop = smart_trimming(image_crop)
        image_crop.save('static/images/cropped_image.jpg')

        # Преобразуем объект Pillow в массив
        image_array = np.array(image_crop)

        # Получаем одномерный массив
        data_for_predict = image_array.reshape(1, 28*28)
        # Инвертируем изображение
        data_for_predict = 255-data_for_predict
        # Нормализуем изображение
        data_for_predict = data_for_predict / 255
        
        print(data_for_predict)

        # Подаем данные в нашу модель и получаем ответы
        prediction = model.predict(data_for_predict)
       
        

        # Берем нулевой элемент, т.к. подали только одну картинку
        result = prediction[0]

        predict_digit = np.argmax(result)

        print("Ответ нейронки: ", result)
        print("Ответ: ", predict_digit)

        data['status'] = 'ok'
        data['image_link'] = 'static/images/cropped_image.jpg'
        data['image_source_link'] = 'static/images/upload_image.jpg'
        data['answer'] = str(predict_digit)
    
    print(data)
    return jsonify(data)

app.run(debug=True, host='ai.role.ru', port=1100)
