import tensorflow as tf
from tensorflow import keras
import numpy as np

kl = keras.layers
kb = keras.backend


class VAE:
    def __init__(self, batch_size, units=40, filters=(32, 64, 128, 256), image_size=256, channels=3):
        self.batch_size = batch_size
        self.f = filters
        self.units = units
        self.im_size = image_size
        self.channels = channels

    def vae_loss(self):
        kloss = self.kl_loss
        rloss = self.reconstruction_loss

        def loss_f(y_true, y_pred):
            rec_loss = rloss(y_true, y_pred)
            rec_loss = tf.reduce_sum(rec_loss, axis=[1, 2])
            kl1_loss = kloss(self.mu5, self.sigma5)

            loss = tf.reduce_mean(rec_loss + kl1_loss)
            # return tf.reduce_mean(rec_loss)
            return loss
        return loss_f

    def reconstruction_loss(self, target, reconstructions):
        loss = tf.reduce_sum(tf.squared_difference(target, reconstructions), axis=-1)
        # loss = keras.losses.mse(y_true=target,y_pred=reconstructions)
        # loss *= self.im_size**2
        return loss

    def kl_loss(self, mu, sigma):
        loss = tf.reduce_mean(- 0.5 * tf.reduce_sum((1. + sigma - tf.square(mu) - tf.exp(sigma)), axis=-1))
        # loss = 1 + tf.log(sigma)*2 - tf.square(mu) - tf.exp(tf.log(sigma)*2)
        return loss

    def make_vae(self):
        self.inputs = kl.Input((self.im_size, self.im_size, self.channels))
        l0 = self.inputs

        ml1 = self.mid_layer(l0, self.f[0], 2)
        dl1 = self.down_layer(ml1,self.f[0])
        ml2 = self.mid_layer(dl1, self.f[1], 2)
        dl2 = self.down_layer(ml2, self.f[1])
        ml3 = self.mid_layer(dl2, self.f[2], 2)
        dl3 = self.down_layer(ml3, self.f[2])
        ml4 = self.mid_layer(dl3, self.f[3], 2)
        dl4 = self.down_layer(ml4, self.f[3])
        dl5 = self.down_layer(dl4, self.f[3])

        self.mu5, self.sigma5, z5 = self.sampling_layer(dl5, self.units, self.f[3], '5')
        d = kl.Dense((self.im_size>>5)*(self.im_size>>5)*self.f[1],kernel_initializer='zeros')(z5)
        rp = kl.Reshape((self.im_size>>5,self.im_size>>5,self.f[1]))(d)

        ul1 = self.up_layer(rp, self.f[3])
        uml1 = self.mid_layer(ul1, self.f[3], 2)
        ul2 = self.up_layer(uml1, self.f[2])
        uml2 = self.mid_layer(ul2, self.f[2], 2)
        ul3 = self.up_layer(uml2, self.f[2])
        uml3 = self.mid_layer(ul3, self.f[1], 2)
        ul4 = self.up_layer(uml3, self.f[1])
        ul5 = self.up_layer(ul4, self.f[0])
        uml4 = self.mid_layer(ul5, self.f[0], 3)

        self.outputs = kl.Conv2D(self.channels, (1,1), padding="same", activation="sigmoid")(uml4)

        vae = keras.models.Model(self.inputs, self.outputs, name='VAE')
        encoder = keras.models.Model(self.inputs, [self.mu5, self.sigma5], name="encoder")
        self.latents = [self.mu5 , self.sigma5]
        return vae, encoder, self.latents, [self.inputs, self.outputs]

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

    def sample_z(self, args):
        mu, sigma = args
        batch = tf.shape(mu)[0]
        dim = mu.get_shape()[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mu + tf.exp(sigma/2) * epsilon

    def sampling_layer(self, input, units, filters, i):
        u = kl.Conv2D(filters=filters,kernel_size=(1,1), kernel_initializer='ones')(input)
        fl1 = kl.Flatten()(u)
        mu = kl.Dense(units, kernel_initializer='ones', name='mu_'+i)(fl1)
        mu = kl.ReLU(max_value=100.,name='mu_'+i+'_a')(mu)
        v = kl.Conv2D(filters=filters,kernel_size=(1,1), kernel_initializer='zeros')(input)
        fl2 = kl.Flatten()(v)
        sigma = kl.Dense(units, kernel_initializer='zeros', name='sigma_'+i)(fl2)
        sigma = kl.ReLU(100., name='sigma_' + i + '_a')(sigma)
        z = kl.Lambda(self.sample_z, name="sample_z"+i)([mu, sigma])

        return mu, sigma, z
