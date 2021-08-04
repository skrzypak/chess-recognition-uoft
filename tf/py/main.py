import os
import cv2
import numpy as np
from random import shuffle

import datetime

import tensorflow as tf
from keras import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from keras.losses import SparseCategoricalCrossentropy

import global_configuration

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator


def create_data():
    # 0 is training, 1 is for validation
    out_data = [[], []]
    dir_paths = [
        CONFIGURATION['TRAIN_DIR'],
        CONFIGURATION['TEST_DIR']
    ]
    prints = [
        'Training data ->',
        'Validating data ->'
    ]

    for inx_path, p in enumerate(dir_paths):
        print("Generating dataset...")
        for curr in os.listdir(p):
            count = 0
            curr_dir = os.path.join(p, curr)
            label_category = CONFIGURATION['PIECES_CATEGORIES'].index(curr)
            for img_name in os.listdir(curr_dir):
                path = os.path.join(curr_dir, img_name)

                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (CONFIGURATION['FIELD_IMG_SIZE'], CONFIGURATION['FIELD_IMG_SIZE']))
                img = cv2.medianBlur(img, 5)

                data = img_to_array(img)
                samples = np.expand_dims(data, 0)
                data_gen = ImageDataGenerator(
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,
                    rotation_range=10,
                    )

                it = data_gen.flow(samples, batch_size=32)
                for i in range(15):
                    batch = it.next()
                    image = batch[0].astype('uint8')
                    out_data[inx_path].append([np.array(image), label_category])
                    count += 1

            print(prints[inx_path] + ' folder: {} count: {}'.format(curr, count))

    print("")
    shuffle(out_data[0])
    np.save('../../assets/npy/training_data.npy', out_data[0])
    shuffle(out_data[1])
    np.save('../../assets/npy/testing_data.npy', out_data[1])
    return out_data[0], out_data[1]


def get_features_labels(data_array):
    x = []
    y = []

    for features, label in data_array:
        x.append(features)
        y.append(label)

    x = np.array(x).reshape(-1, CONFIGURATION['FIELD_IMG_SIZE'], CONFIGURATION['FIELD_IMG_SIZE'], 1)
    y = np.array(y).reshape(-1, 1)

    x = x / 255.0
    return x, y


def create_model(shapes):
    return Sequential([
        Conv2D(filters=32, kernel_size=4, activation='relu', padding='same', input_shape=shapes.shape[1:]),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=4, activation=tf.keras.layers.LeakyReLU(), padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=128, kernel_size=4, activation=tf.keras.layers.LeakyReLU(), padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=32, kernel_size=4, activation=tf.keras.layers.LeakyReLU(), padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer='l2'),
        Dense(256, activation='relu', kernel_regularizer='l2'),
        Dropout(0.3),
        Dense(13, activation='softmax')
    ])


if __name__ == '__main__':

    CONFIGURATION = global_configuration.get_tf()

    # Load Libraries and Data
    if os.path.isfile(
            '../../assets/npy/training_data.npy') and os.path.isfile('../../assets/npy/testing_data.npy'):
        training_data = np.load('../../assets/npy/training_data.npy', allow_pickle=True)
        testing_data = np.load('../../assets/npy/testing_data.npy', allow_pickle=True)
    else:
        training_data, testing_data = create_data()

    print('Training and testing data loaded successfully')

    X, Y = get_features_labels(training_data)
    test_x, test_y = get_features_labels(testing_data)

    print('Configure model')

    #  Build model
    model = create_model(X)
    # model.summary()
    model.compile(loss=SparseCategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    # Init model logs
    log_dir = "../logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    mc = ModelCheckpoint('../models/best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    # Resolve model
    model.fit(X, Y, epochs=11, validation_data=(test_x, test_y), callbacks=[tensorboard_callback, mc],
              verbose=1, use_multiprocessing=True)

    model.save(os.path.join('../models', CONFIGURATION['MODEL_NAME']))

    print("\nEvaluate on testing data")
    result = model.evaluate(test_x, test_y, verbose=1)
    print("test loss, test acc:", result)
