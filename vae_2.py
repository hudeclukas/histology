import numpy as np
import tensorflow as tf
from tensorflow import keras

kl = keras.layers
kb = keras.backend


class VAE_2:
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
            kl1_loss = kloss(self.mu1, self.sigma1)
            kl2_loss = kloss(self.mu2, self.sigma2)
            kl3_loss = kloss(self.mu3, self.sigma3)
            kl4_loss = kloss(self.mu4, self.sigma4)

            loss = tf.reduce_mean(rec_loss + 0.*(kl1_loss + kl2_loss + kl3_loss + kl4_loss))
            # return tf.reduce_mean(rec_loss)
            return loss
        return loss_f

    def reconstruction_loss(self, target, reconstructions):
        loss = tf.reduce_sum(tf.squared_difference(target, reconstructions), axis=-1)
        # loss = keras.losses.mse(y_true=target,y_pred=reconstructions)
        # loss *= self.im_size**2
        return loss

    def kl_loss(self, mu, sigma):
        eps = 1e-10
        loss = tf.reduce_mean(
           - 0.5 * tf.reduce_sum(
                (2. * tf.log(sigma + eps) - tf.square(sigma+eps) - tf.square(mu+eps) + 1.),
                axis=1))
        # loss = 1 + tf.log(sigma)*2 - tf.square(mu) - tf.exp(tf.log(sigma)*2)
        return loss

    def make_vae2(self):
        self.inputs = kl.Input((self.im_size, self.im_size, self.channels))
        l0 = self.inputs

        dl1 = self.down_layer(l0,self.f[0])
        ml1 = self.mid_down_layer(dl1, self.f[0], 3)
        self.mu1, self.sigma1, z1 = self.sampling_layer(ml1, self.units, self.f[0], '1')
        dl2 = self.down_layer(dl1, self.f[1])
        ml2 = self.mid_down_layer(dl2, self.f[0], 2)
        self.mu2, self.sigma2, z2 = self.sampling_layer(ml2, self.units, self.f[1], '2')
        dl3 = self.down_layer(dl2, self.f[2])
        ml3 = self.mid_down_layer(dl3, self.f[0], 1)
        self.mu3, self.sigma3, z3 = self.sampling_layer(ml3, self.units, self.f[2], '3')
        dl4 = self.down_layer(dl3, self.f[3])
        self.mu4, self.sigma4, z4 = self.sampling_layer(dl4, self.units, self.f[3], '4')

        ul4 = self.up_layer(z4, self.f[3], (self.im_size >> 4, self.im_size >> 4, 1))
        ul3 = self.up_layer(z3, self.f[2], (self.im_size >> 3, self.im_size >> 3, 1), ul4)
        ul2 = self.up_layer(z2, self.f[1], (self.im_size >> 2, self.im_size >> 2, 1), ul3)
        ul1 = self.up_layer(z1, self.f[0], (self.im_size >> 1, self.im_size >> 1, 1), ul2)

        up0 = kl.UpSampling2D(size=(2,2), interpolation='bilinear')(ul1)
        c0 = kl.Conv2D(self.f[0], (3, 3), padding="same", activation="relu")(up0)
        self.outputs = kl.Conv2D(self.channels, (1,1), padding="same", activation="sigmoid")(c0)

        vae = keras.models.Model(self.inputs, self.outputs, name='VAE')
        encoder = keras.models.Model(self.inputs, [self.mu1,self.sigma1,self.mu2,self.sigma2,self.mu3,self.sigma3,self.mu4,self.sigma4], name="encoder")
        self.latents = [[self.mu1, self.sigma1], [self.mu2, self.sigma2], [self.mu3, self.sigma3], [self.mu4 ,self.sigma4]]
        return vae, encoder, self.latents, [self.inputs, self.outputs]

    def down_layer(self, input, filters, kernel_size=(3,3), padding="same", strides=1, activation="relu"):
        c = kl.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(input)
        a = kl.LeakyReLU()(c)
        bn = kl.BatchNormalization()(a)
        p = kl.Conv2D(filters=filters, kernel_size=kernel_size, strides=(2,2), padding=padding)(bn)
        return p

    def up_layer(self, sample, filters, size, from_lower=None, kernel_size=(3,3), padding="same", strides=1, activation="relu"):
        d = kl.Dense(size[0]*size[1], kernel_initializer='zeros')(sample)
        r = kl.Reshape(size)(d)
        if not from_lower == None:
            u = kl.UpSampling2D(size=(2,2))(from_lower)
            cc = kl.Concatenate()([u,r])
            c = kl.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(cc)
            a = kl.ReLU()(c)
            dr = kl.Dropout(rate=0.2)(a)
            return dr
        else:
            return kl.LeakyReLU()(r)

    def mid_up_layer(self, input, filters, interpolation, repeat_count):
        if repeat_count==0:
            u = kl.UpSampling2D(size=(2,2), interpolation=interpolation)(input)
            c = kl.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", strides=2, kernel_initializer='zeros')(u)
            c = kl.LeakyReLU()(c)
        else:
            for i in range(repeat_count):
                if i == 0:
                    u = kl.UpSampling2D(size=(2, 2), interpolation=interpolation)(input)
                    c = kl.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", strides=2)(u)
                    c = kl.LeakyReLU()(c)
                else:
                    u = kl.UpSampling2D(size=(2, 2), interpolation=interpolation)(c)
                    c = kl.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", strides=2)(u)
                    c = kl.LeakyReLU()(c)
        return c

    def mid_down_layer(self, input, filters, repeat_count):
        if repeat_count==0:
            c = kl.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", strides=2, activation="relu")(input)
        else:
            for i in range(repeat_count):
                if i == 0:
                    c = kl.Conv2D(filters=filters, kernel_size=(3,3), padding="same", strides=2, activation="relu")(input)
                else:
                    c = kl.Conv2D(filters=filters, kernel_size=(3,3), padding="same", strides=2, activation="relu")(c)
        bn = kl.BatchNormalization()(c)
        return bn

    def sample_z(self, args):
        mu, sigma = args
        batch = tf.shape(mu)[0]
        dim = mu.get_shape()[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mu + tf.exp(tf.log(sigma)/2) * epsilon

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
