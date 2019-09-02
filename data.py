## Imports
import os

import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras

## Seeding
seed = 2019
np.random.seed = seed
tf.seed = seed


class DualDataGen(keras.utils.Sequence):
    def __init__(self):
        pass

    def __load__(self, name, path):
        pass

    def __getitem__(self, index):
        pass

    def on_epoch_end(self):
        pass

    def __len__(self):
        pass

class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, database_path='', batch_size=8, image_size=256, channels=3):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels
        self.on_epoch_end()
        self.database_path = database_path

    def __load__(self, name):
        ## Path
        image_path = os.path.join(self.path, name)

        ## Reading Image
        image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if not image.shape == (self.image_size, self.image_size, self.channels):
            image = cv2.resize(image, (self.image_size, self.image_size))
        ## Normalizaing
        image = image / 0xffff

        return image

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size

        files_batch = self.ids[index * self.batch_size: (index + 1) * self.batch_size]

        images = []
        for id_name in files_batch:
            images.append(self.__load__(id_name))
        images = np.array(images)
        return images, images

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

    def n_way_validation_support_set(self, n):
        dirs = [a for a in os.listdir(self.database_path) if os.path.isdir(os.path.join(self.database_path,a))]
        set = {}
        for dir in dirs:
            folder = os.path.join(self.database_path, dir)
            files = os.listdir(folder)
            selected = np.random.permutation(files)[:n]
            set[dir]=[]
            for s in selected:
                image = cv2.imread(os.path.join(folder, s), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR) / 0xffff
                if not image.shape == (self.image_size, self.image_size, self.channels):
                    image = cv2.resize(image, (self.image_size, self.image_size))
                set[dir].append(image)

        return set

    @staticmethod
    def l2_norm(a,b):
        return np.linalg.norm(b-a)
