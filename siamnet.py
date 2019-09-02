import tensorflow as tf
from tensorflow import keras
import numpy as np

kl = keras.layers
kb = keras.backend


class siamnet:
    def __init__(self, batch_size, units=40, filters=(32, 64, 128, 256), image_size=256, channels=3):
        self.batch_size = batch_size
        self.f = filters
        self.units = units
        self.im_size = image_size
        self.channels = channels

    def siam_contrastive_loss(self):
        def loss_f(y_true, y_pred):
            margin = 2.5
            square_pred = kb.square(y_pred)
            margin_square = kb.square(kb.maximum(margin - y_pred, 0))
            return kb.mean(y_true * square_pred + (1-y_true) * margin_square)

        return loss_f

    def make_siamnet(self):
        inputs = kl.Input((self.im_size, self.im_size, self.channels))
        l0 = inputs

        ml1 = self.mid_layer(l0, self.f[0], 2)
        dl1 = self.down_layer(ml1,self.f[0])
        ml2 = self.mid_layer(dl1, self.f[1], 2)
        dl2 = self.down_layer(ml2, self.f[1])
        ml3 = self.mid_layer(dl2, self.f[2], 2)
        dl3 = self.down_layer(ml3, self.f[2])
        ml4 = self.mid_layer(dl3, self.f[3], 2)
        dl4 = self.down_layer(ml4, self.f[3])
        dl5 = self.down_layer(dl4, self.f[3])

        flat = kl.Flatten()(dl5)

        self.outputs = kl.Conv2D(self.channels, (1,1), padding="same")(flat)
        self.outputs = kl.LeakyReLU()(self.outputs)

        base = keras.models.Model(inputs, self.outputs, name='SIAM')

        self.left_input = kl.Input((self.im_size, self.im_size, self.channels))
        self.right_input = kl.Input((self.im_size, self.im_size, self.channels))

        left = keras.models.Model(self.left_input, base)
        right = keras.models.Model(self.right_input, base)

        return left, right

    def down_layer(self, input, filters, kernel_size=(3,3), padding="same", strides=2, activation="relu"):
        c = kl.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(input)
        a = kl.LeakyReLU()(c)
        bn = kl.BatchNormalization()(a)
        return bn

    def mid_layer(self, input, filters, repeat_count):
        if repeat_count==0:
            c = kl.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", strides=1)(input)
            c = kl.LeakyReLU()(c)
        else:
            for i in range(repeat_count):
                if i == 0:
                    c = kl.Conv2D(filters=filters, kernel_size=(3,3), padding="same", strides=1)(input)
                    c = kl.LeakyReLU()(c)
                else:
                    c = kl.Conv2D(filters=filters, kernel_size=(3,3), padding="same", strides=1)(c)
                    c = kl.LeakyReLU()(c)
        bn = kl.BatchNormalization()(c)
        return bn
