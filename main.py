import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import keras as k
import tensorflow as tf
import matplotlib.pyplot as plt

# reading the data

data_frame = pd.read_csv('Zodiacs.csv')
input_names = ["Day", "Month"]
output_names = ["Sighns"]

raw_input_data = data_frame[input_names]
raw_output_data = data_frame[output_names]

# Подготавливаем данные для работы, возвращаем их в список

encoders = {"Day": lambda days: {
    1: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    2: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    3: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    4: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    5: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    6: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    7: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    8: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    9: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    10: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    11: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    12: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    13: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    14: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    15: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    16: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    17: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    18: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    19: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    20: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    21: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    22: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    23: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    24: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    25: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    26: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    27: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    28: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    29: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    30: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    31: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}.get(days),
            "Month": lambda months: {"Jan": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     "Feb": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     "Mar": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     "Apr": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                     "May": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                     "Jun": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                     "Jul": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                     "Aug": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                     "Sep": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                     "Oct": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                     "Nov": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                     "Dec": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}.get(months),
            "Sighns": lambda sighn: {"Aries": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     "Taurus": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     "Gemini": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     "Cancer": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                     "Leo": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                     "Virgo": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                     "Libra": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                     "Scorpio": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                     "Sagittarius": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                     "Capricorn": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                     "Aquarius": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                     "Pisces": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}.get(sighn)}


def dataframe_to_dict(df):
    result = dict()
    for column in df.columns:
        values = data_frame[column].values
        result[column] = values
    return result


# Separate data

def make_supervised(df):
    raw_input_data = data_frame[input_names]
    raw_output_data = data_frame[output_names]
    return {"inputs": dataframe_to_dict(raw_input_data),
            "outputs": dataframe_to_dict(raw_output_data)}


# encode the data

def encode(data):
    vectors = []
    for data_name, data_values in data.items():
        encoded = list(map(encoders[data_name], data_values))
        vectors.append(encoded)
    formatted = []
    for vector_raw in list(zip(*vectors)):
        vector = []
        for element in vector_raw:
            for e in element:
                vector.append(e)
        formatted.append(vector)
    return formatted


supervised = make_supervised(data_frame)
encoded_inputs = np.array(encode(supervised["inputs"]), dtype=int)
encoded_outputs = np.array(encode(supervised["outputs"]), dtype=int)

# Make a samples

train_x = encoded_inputs[200:]
train_y = encoded_outputs[200:]
test_x = encoded_inputs[200:]
test_y = encoded_outputs[200:]

# Train model

model = k.Sequential([tf.keras.Input(shape=(None, 32, 43))])
model.add(k.layers.Dense(units=43, activation='relu'))
model.add(k.layers.Dense(units=12, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
#model.load_weights('weights')
fit_results = model.fit(x=train_x, y=train_y, epochs=20, batch_size=2, validation_split=0.2)
model.save_weights('weights')

# visualization

# plt.title("Losses train/validation")
# plt.plot(fit_results.history["loss"], label="Train")
# plt.plot(fit_results.history["val_loss"], label="Validation train")
# plt.legend()
# plt.show()
#
# plt.title("Accuracies train/validation")
# plt.plot(fit_results.history["accuracy"], label="Acc Train")
# plt.plot(fit_results.history["accuracy"], label="Acc Validation train")
# plt.legend()
# plt.show()

answer = 'y'
while answer != 'n':
    print('Enter the day you were born')
    day = int(input())
    print('Enter the number of month you were born')
    month = int(input())

    day_ready = []
    for i in range(0, 31):
        if i == (day - 1):
            day_ready.append(1)
        else:
            day_ready.append(0)
    month_ready = []
    for j in range(0, 12):
        if j == (month - 1):
            month_ready.append(1)
        else:
            month_ready.append(0)
    for k in range(0, 12):
        day_ready.append(month_ready[k])

    data_ready = np.array(day_ready, dtype=int)
    x = np.expand_dims(data_ready, axis=0)
    res = model.predict(x)
    a = np.argmax(res)
    if a == 0:
        zodiac = "Aries"
    elif a == 1:
        zodiac = "Taurus"
    elif a == 2:
        zodiac = "Gemini"
    elif a == 3:
        zodiac = "Cancer"
    elif a == 4:
        zodiac = "Leo"
    elif a == 5:
        zodiac = "Virgo"
    elif a == 6:
        zodiac = "Libra"
    elif a == 7:
        zodiac = "Scorpio"
    elif a == 8:
        zodiac = "Sagittarius"
    elif a == 9:
        zodiac = "Capricorn"
    elif a == 10:
        zodiac = "Aquarius"
    elif a == 11:
        zodiac = "Pisces"
    print(f"Your sighn is ", zodiac)
    print('Want to continue?    y/n')
    answer = str(input())
