import tensorflow as tf
from tensorflow import keras
import numpy as np

kl = keras.layers
kb = keras.backend


class AE:
    def __init__(self, batch_size, units=40, filters=(64, 128, 128, 256), image_size=256, channels=3):
        self.batch_size = batch_size
        self.f = filters
        self.units = units
        self.im_size = image_size
        self.channels = channels

    def ae_loss(self):
        rloss = self.reconstruction_loss
        def loss_f(y_true, y_pred):
            rec_loss = rloss(y_true, y_pred)
            rec_loss = tf.reduce_sum(rec_loss, axis=[1, 2])
            # kl1_loss = kloss(self.mu5, self.sigma5)

            loss = tf.reduce_mean(rec_loss)
            # return tf.reduce_mean(rec_loss)
            return loss
        return loss_f

    def reconstruction_loss(self, target, reconstructions):
        loss = tf.reduce_sum(tf.squared_difference(target, reconstructions), axis=-1)
        # loss = keras.losses.mse(y_true=target,y_pred=reconstructions)
        # loss *= self.im_size**2
        return loss

    def make_ae(self):
        self.inputs = kl.Input((self.im_size, self.im_size, self.channels))
        l0 = self.inputs

        ml1 = self.mid_layer(l0, self.f[0], 2)
        dl1 = self.down_layer(ml1,self.f[0])
        ml2 = self.mid_layer(dl1, self.f[1], 2)
        dl2 = self.down_layer(ml2, self.f[1])
        ml3 = self.mid_layer(dl2, self.f[2], 2)
        dl3 = self.down_layer(ml3, self.f[2])
        # ml4 = self.mid_layer(dl3, self.f[3], 2)
        dl4 = self.down_layer(dl3, self.f[3])
        dl5 = self.down_layer(dl4, self.f[3])

        fl = kl.Flatten()(dl5)
        self.middle = kl.Dense(units=self.units,kernel_initializer='ones')(fl)
        d = kl.Dense((self.im_size >> 5) * (self.im_size >> 5) * self.f[0], kernel_initializer='ones')(self.middle)
        rp = kl.Reshape((self.im_size>>5,self.im_size>>5,self.f[0]))(d)

        ul1 = self.up_layer(rp, self.f[3])
        # uml1 = self.mid_layer(ul1, self.f[3], 2)
        ul2 = self.up_layer(ul1, self.f[2])
        uml2 = self.mid_layer(ul2, self.f[2], 2)
        ul3 = self.up_layer(uml2, self.f[2])
        uml3 = self.mid_layer(ul3, self.f[1], 2)
        ul4 = self.up_layer(uml3, self.f[1])
        ul5 = self.up_layer(ul4, self.f[0])
        uml4 = self.mid_layer(ul5, self.f[0], 3)

        self.outputs = kl.Conv2D(self.channels, (1,1), padding="same", activation="sigmoid")(uml4)

        ae = keras.models.Model(self.inputs, self.outputs, name='VAE')
        encoder = keras.models.Model(self.inputs, [self.middle], name="encoder")
        self.latents = [self.middle]
        return ae, encoder, self.latents, [self.inputs, self.outputs]

    def down_layer(self, input, filters, kernel_size=(3,3), padding="same", strides=2, activation="relu"):
        c = kl.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(input)
        a = kl.LeakyReLU()(c)
        bn = kl.BatchNormalization()(a)
        return bn

    def up_layer(self, sample, filters, kernel_size=(3,3), padding="same", strides=1, activation="relu"):
        u = kl.UpSampling2D(size=(2,2))(sample)
        c = kl.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(u)
        a = kl.LeakyReLU()(c)
        dr = kl.Dropout(rate=0.2)(a)
        return dr

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
