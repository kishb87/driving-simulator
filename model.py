import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
import cv2
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Input, Activation, Conv2D, Flatten, MaxPooling2D, Activation, Dropout
from keras.optimizers import Adam
from keras.models import Sequential, model_from_json
import json

def normalize_image(image):
    image = image / 255
    image -= 0.5
    return image

def resize_image(image, plot = False):
    crop_image = image[50:150, :]
    image = cv2.resize(crop_image, (200, 66), interpolation=cv2.INTER_AREA)
    return image

def color_transform(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

def get_image(path, plot = False):
    image = cv2.imread(path)
    image = color_transform(image)
    image = normalize_image(image)
    image = resize_image(image)
    if plot == True:
        plt.imshow(image)
        plt.show()
    return image

def preprocess(i, data):
    index = randint(0, len(data) - 1)
    loc = np.random.randint(3)
    path = None
    shift_ang = None
    if (loc == 0 ):
        path = data.ix[index][0]
        shift_ang = 0
    elif (loc == 1):
        path = data.ix[index][1]
        shift_ang = .25
    elif (loc == 2):
        path = data.ix[index][2]
        shift_ang = -.25
    path = data.ix[index][0]
    x = get_image(path)
    y = data.ix[index][3]
    return x, y

def create_batch(data, batch_size):
    while True:
        for i in range(batch_size):
            x, y = preprocess(i, data)
            image_batch[i] = x
            steering_batch[i] = y
        yield image_batch, steering_batch

if __name__ == '__main__':
    data = pd.read_csv("driving_log.csv")

    batch_size = 64
    steering_batch = np.zeros(batch_size)
    image_batch = np.zeros((batch_size, 66, 200, 3))

    # shuffle dataframe
    data = data.sample(frac=1)

    model = Sequential()

    model.add(Conv2D(24, 5, 5, input_shape=(66, 200, 3)))
    model.add(MaxPooling2D((2,2)))
    model.add((Dropout(0.5)))
    model.add(Activation('relu'))

    model.add(Conv2D(36, 5, 5))
    model.add(MaxPooling2D((2,2)))
    model.add((Dropout(0.5)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, 5, 5))
    model.add(MaxPooling2D((2,2), border_mode='same'))
    model.add((Dropout(0.5)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, 3, 3))
    model.add((Dropout(0.5)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, 3, 3))
    model.add((Dropout(0.5)))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(1164, activation='relu'))
    model.add((Dropout(0.5)))

    model.add(Dense(100, activation='relu'))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(10, activation='relu'))

    model.add(Dense(1))

    gen = create_batch(data, batch_size)
    val = create_batch(data, batch_size)

    model.compile(loss='mse', optimizer=Adam(lr=0.0001),metrics=['mean_squared_error'])
    history = model.fit_generator(create_batch(data, batch_size), samples_per_epoch=20032,
                                  nb_epoch=8,validation_data=val, nb_val_samples=6400, verbose=2)

    model.save(filepath='model.h5')
    model_json = model.to_json()

    with open('model.json', 'w') as f:
        json.dump(model_json, f)
    print("Saved model to filename="+'model.h5'+ ", and model_arch="+'model.json')
