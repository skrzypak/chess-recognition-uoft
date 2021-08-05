import os
import sys

import cv2
import numpy as np
from keras_preprocessing.image import img_to_array, ImageDataGenerator

import global_configuration


def main(args):

    CONFIGURATION = global_configuration.get()
    dir_path = '../assets/chess_dataset/train'
    dir_out_path = '../assets/chess_dataset/generate'

    for curr_dir_name in os.listdir(dir_path):
        curr_dir = os.path.join(dir_path, curr_dir_name)
        curr_dir_out = os.path.join(dir_out_path, curr_dir_name)

        if not os.path.exists(curr_dir_out):
            os.mkdir(curr_dir_out)

        print(curr_dir_out, 'START...')

        for img_name in os.listdir(curr_dir):
            img_path = os.path.join(curr_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (CONFIGURATION['FIELD_IMG_SIZE'], CONFIGURATION['FIELD_IMG_SIZE']))
            data = img_to_array(img)
            samples = np.expand_dims(data, 0)
            data_gen = ImageDataGenerator(
                zoom_range=0.1,
                width_shift_range=0.15,
                height_shift_range=0.15,
                horizontal_flip=True,
                rotation_range=25,
            )
            it = data_gen.flow(samples, batch_size=32, save_to_dir=curr_dir_out)
            for i in range(50):
                it.next()

        print(curr_dir_out, 'DONE')


if __name__ == '__main__':
    main(sys.argv[1:])
