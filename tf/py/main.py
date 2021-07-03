import os
import cv2
import numpy as np
from random import shuffle

from tqdm import tqdm
import tensorflow as tf
import datetime

import global_configuration


def create_train_data():
    init_training_data = []

    for curr in os.listdir(CONFIGURATION['TRAIN_DIR']):
        curr_train_dir = os.path.join(CONFIGURATION['TRAIN_DIR'], curr)
        label_category = CONFIGURATION['PIECES_CATEGORIES'].index(curr)
        for img_train in tqdm(os.listdir(curr_train_dir)):
            path = os.path.join(curr_train_dir, img_train)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_ready = cv2.resize(img, (CONFIGURATION['FIELD_IMG_SIZE'], CONFIGURATION['FIELD_IMG_SIZE']))
            init_training_data.append([np.array(img_ready), label_category])

    shuffle(init_training_data)
    np.save('../../assets/npy/train_data.npy', init_training_data)
    return init_training_data


def create_testing_data():
    init_testing_data = []
    inx = 0

    for curr in os.listdir(CONFIGURATION['TEST_DIR']):
        curr_testing_dir = os.path.join(CONFIGURATION['TEST_DIR'], curr)
        for img_test in tqdm(os.listdir(curr_testing_dir)):
            path = os.path.join(curr_testing_dir, img_test)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_ready = cv2.resize(img, (CONFIGURATION['FIELD_IMG_SIZE'], CONFIGURATION['FIELD_IMG_SIZE']))
            init_testing_data.append([np.array(img_ready), inx])
            inx += 1

    shuffle(init_testing_data)
    np.save('../../assets/npy/testing_data.npy', init_testing_data)
    return init_testing_data


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
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='sigmoid', input_shape=shapes.shape[1:]),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU()),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU()),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU()),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation=tf.keras.layers.ReLU()),
        tf.keras.layers.Dense(13, activation='sigmoid')
    ])


if __name__ == '__main__':

    CONFIGURATION = global_configuration.get_tf()

    # Creating / loading train data
    if os.path.isfile('../../assets/npy/train_data.npy'):
        training_data = np.load('../../assets/npy/train_data.npy', allow_pickle=True)
    else:
        training_data = create_train_data()

    print('Training data loaded successfully')

    if os.path.isfile('../../assets/npy/testing_data.npy'):
        testing_data = np.load('../../assets/npy/testing_data.npy', allow_pickle=True)
    else:
        testing_data = create_testing_data()

    print('Testing data loaded successfully')

    # Data accuracy

    X, Y = get_features_labels(training_data)
    test_x, test_y = get_features_labels(testing_data)

    # CNN
    model = create_model(X)
    model.summary()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy'])

    # Init logs
    log_dir = "../logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Resolve model
    model.fit(X, Y, batch_size=32, epochs=11, validation_split=CONFIGURATION['LR'],
              validation_data=(test_x, test_y), callbacks=[tensorboard_callback])

    model.save(os.path.join('../models', CONFIGURATION['MODEL_NAME']))

    print("\nEvaluate on train data")
    result = model.evaluate(X, Y, batch_size=32, steps=5)
    print("test loss, test acc:", result)
