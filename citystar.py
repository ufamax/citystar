from flask import Flask, jsonify, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import pickle


# функция загрузки обьекта
def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)  # loading


# категориальные данные

# функция преобразования района в формат для подачи в нейронку
def pack_district(d):
    d = str(d)  # конвертирум в строку
    t = district.transform([d])[0]  # берем категорию
    n = len(district.classes_)  # число классов этого столбца
    l = to_categorical(t, n)  # превращаем токен в One-Hot Encoding
    return list(l)  # возвращаем в виде списка в формате One-Hot Encoding


# функция преобразования улицы в формат для подачи в нейронку
def pack_street(d):
    d = str(d)  # конвертирум в строку
    t = street.transform([d])[0]  # берем категорию
    n = len(street.classes_)  # число классов этого столбца
    l = to_categorical(t, n)  # превращаем токен в One-Hot Encoding
    return list(l)  # возвращаем в виде списка в формате One-Hot Encoding


# функция преобразования этажа в формат для подачи в нейронку
def pack_house_floor(d):
    d = str(d)  # конвертирум в строку
    t = house_floor.transform([d])[0]  # берем категорию
    n = len(house_floor.classes_)  # число классов этого столбца
    l = to_categorical(t, n)  # превращаем токен в One-Hot Encoding
    return list(l)  # возвращаем в виде списка в формате One-Hot Encoding


# функция преобразования планировки в формат для подачи в нейронку
def pack_planning(d):
    d = str(d)  # конвертирум в строку
    t = planning.transform([d])[0]  # берем категорию
    n = len(planning.classes_)  # число классов этого столбца
    l = to_categorical(t, n)  # превращаем токен в One-Hot Encoding
    return list(l)  # возвращаем в виде списка в формате One-Hot Encoding


# функция преобразования комнат в формат для подачи в нейронку
def pack_rooms(d):
    d = str(d)  # конвертирум в строку
    t = rooms.transform([d])[0]  # берем категорию
    n = len(rooms.classes_)  # число классов этого столбца
    l = to_categorical(t, n)  # превращаем токен в One-Hot Encoding
    return list(l)  # возвращаем в виде списка в формате One-Hot Encoding


# числовые данные

# функция преобразования area1 в формат для подачи в нейронку
def pack_area1(d):
    try:
        d = float(d)  # конвертирум во float
    except:
        d = 0.0  # при ошибке подаем 0
    r = area1.transform(np.array(d).reshape(-1, 1))[0][0]  # нормализуем по скалеру для этого столбца
    return r  # возвращаем в виде числа float


# функция преобразования area2 в формат для подачи в нейронку
def pack_area2(d):
    try:
        d = float(d)  # конвертирум во float
    except:
        d = 0.0  # при ошибке подаем 0
    r = area2.transform(np.array(d).reshape(-1, 1))[0][0]  # нормализуем по скалеру для этого столбца
    return r  # возвращаем в виде числа float


# функция преобразования price в формат для подачи в нейронку
def pack_price(d):
    try:
        d = float(d)  # конвертирум во float
    except:
        d = 0.0  # при ошибке подаем 0
    r = price.transform(np.array(d).reshape(-1, 1))[0][0]  # нормализуем по скалеру для этого столбца
    return r  # возвращаем в виде числа float


# текстовые данные

# функция преобразования текста в вектор Bag-of-Words
def pack_comment(d):
    d = str(d)  # конвертирум в строку
    t = tokenizer.texts_to_sequences([d])[0]  # превращаем слова в токены
    l = tokenizer.sequences_to_matrix([t])[0]  # превращаем в вектор BoF
    return list(l)  # вовзращаем списком вектор BoF


# функция сборки вектора для подачи в модель
def get_all_parameters(*args):
    out = []  # здесь будут все параметры в ряд
    for c in args:
        if isinstance(c, float) or isinstance(c, int):
            out.append(c)  # добавить append в список
        if isinstance(c, list):
            out.extend(c)  # дополнить extend список
    return out  # вернем список


# функция предикта цены
def get_price(model, district, street, house_floor, planning, rooms, area1, area2, comment):
    # собираем входные данные с помощью get_all_parameters()
    x_data = [get_all_parameters(pack_district(district), \
                                 pack_street(street), \
                                 pack_house_floor(house_floor), \
                                 pack_planning(planning), \
                                 pack_rooms(rooms), \
                                 pack_area1(area1), \
                                 pack_area2(area2), \
                                 pack_comment(comment))]
    x_data = np.array(x_data)  # переводим в numpy
    pred = model.predict(x_data, verbose=0)  # делаем прогноз
    predUnscaled = price.inverse_transform(pred.reshape(-1, 1)).flatten()
    return str(round(predUnscaled[0], 0))  # вернем окргулив в формате str


# загрузим енкодеры
district = load_obj('data/district')
street = load_obj('data/street')
house_floor = load_obj('data/house_floor')
planning = load_obj('data/planning')
rooms = load_obj('data/rooms')
area1 = load_obj('data/area1')
area2 = load_obj('data/area2')
price = load_obj('data/price')
tokenizer = load_obj('data/comment')

# загрузим модель
model = load_model('data/model1.h5', compile=True)

# создаем наше веб-приложение
app = Flask(__name__)


@app.route('/predict_price', methods=['GET'])
def predict_price():
    data = {'status': 'error'}  # по умолчанию ошибка
    district = request.args.get('district', default='', type=str)
    street = request.args.get('street', default='', type=str)
    house_floor = request.args.get('house_floor', default='', type=str)
    planning = request.args.get('planning', default='', type=str)
    rooms = request.args.get('rooms', default='', type=str)
    area1 = request.args.get('area1', default='', type=str)
    area2 = request.args.get('area2', default='', type=str)
    comment = request.args.get('comment', default='', type=str)

    # делаем прогноз
    price = get_price(model, district, street, house_floor, planning, rooms, area1, area2, comment)
    data['price'] = price  # вернем цену
    data['status'] = 'ok'  # говорим что все ок

    print(data)
    return jsonify(data)


app.run(debug=True, host='localhost', port=52111)

# # тестовый локальный запуск
# print('Верный ответ:', 2850)
# print('Прогноз:', get_price(model, 'Ленинский', 'Куйбышева', '1', 'раздельная', '3', '57', '35',
#                             'Трехкомнатная квартира нестандартной планировки (третья комната с отдельным входом) '
#                             'в нормальном состоянии, санузел раздельный, трубы пластиковые, водомеры. Первый '
#                             'этаж, высоко. В центре Ленинского района. Есть все: садики, школы, скверы и парки, '
#                             'магазины, поликлиника. Очень тихое, спокойное место.'))
# print('Верный ответ:', '6990')
# print('Прогноз:',
#       get_price(model, 'Орджоникидзевский', '50-летия Магнитки', '3', 'нестандартная', '3', '98.5', '60',
#                 'Продам трехкомнатную квартиру 100кв.м. Просторная,светлая,уютная- встроенные удобные '
#                 'гардеробные, два больших балкона, кондиционер, водонагреватель, раздельный санузел, '
#                 'видео- звонок. Во дворе дома детский сад и школа. Остается мебель и кухонный гарнитур. В '
#                 'собственности более 5 лет. Долгов и обременений нет.Один собственник.'))
# print('Верный ответ:', '5100')
# print('Прогноз:', get_price(model, 'Ленинский', 'Менделеева', '3', 'старой планировки', '3', '83', '55',
#                             'ПРОДАМ трехкомнатную квартиру старой планировки в хорошем состоянии, раздельная. вся '
#                             'инфраструктура рядом. квартира в хорошем состоянии, пл.окна, водомеры, '
#                             'заменена эл.проводка, стены выровнены, натяжной потолок, м/двери, ламинат. с/у разд. '
#                             'в кафеле, душ.кабина, остается кухонный гарнитур, прихожая, холодильник, '
#                             'два кондиционера, стиральная машина. б/з. или ОБМЕНЯЮ на двухкомнатную квартиру '
#                             'старой планировки в Ленинском районе + ваша доплата. рассмотрю варианты'))
