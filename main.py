## Imports
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import plot_model

from model import UNet, VAE_2
from data import DataGen

## Seeding
seed = 2019
np.random.seed = seed
tf.seed = seed

image_size = 256
channels = 3
train_path = "D:/Vision_Images/Histology/ICIAR 2018/selectedpairs/All/"
database_path = "D:/Vision_Images/Histology/ICIAR 2018/selectedpairs/benign/"
model_path = "D:/DL_Models/histology/"
model_name = "VAE2_W.h5"
epochs = 2
batch_size = 8

## Training Ids
train_ids = np.asarray(os.listdir(train_path))

## Validation Data Size
val_data_idxs = np.random.randint(low=0, high=len(os.listdir(train_path)), size=37)
train_data_idxs = np.arange(len(train_ids))
train_data_idxs = np.delete(train_data_idxs, val_data_idxs)

valid_ids = train_ids[val_data_idxs]
train_ids = train_ids[train_data_idxs]

train_gen = DataGen(train_ids, train_path, image_size=image_size, channels=channels, batch_size=batch_size)
valid_gen = DataGen(valid_ids, train_path, database_path=database_path, image_size=image_size, channels=channels, batch_size=batch_size)

train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size

if not os.path.exists(model_path):
    os.makedirs(model_path)

# model, model_middle = UNet(256, 3)

vae = VAE_2(batch_size, units=20)
model_vae, model_encoder, latents, ios = vae.make_vae2()

if os.path.exists(os.path.join(model_path, model_name)):
    model_vae.load_weights(os.path.join(model_path,model_name))


# model_vae.add_loss(vae_loss)
adam = tf.keras.optimizers.Adam(lr=0.0001)
model_vae.compile(optimizer=adam, metrics=["acc"], loss=vae.vae_loss())
model_vae.summary()
plot_model(model_vae,model_path+"modelvae.png")
# model_middle.summary()


## Dataset for prediction
x, y = valid_gen.__getitem__(0)
result = model_vae.predict(x)
rr = np.concatenate((x[0],result[0]),axis=1)
plt.imshow(rr)
plt.show()

if epochs > 0:
    model_vae.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=epochs)

    ## Save the Weights
    model_vae.save_weights(model_path + model_name)

result = model_vae.predict(x)
rr = np.concatenate((x[0],result[0]),axis=1)
plt.imshow(rr)
plt.show()

## Dataset for prediction
# support_set = valid_gen.n_way_validation_support_set(5)
# support_set_middles = {}
# for set_key in support_set.keys():
#     set = np.asarray(support_set[set_key])
#     support_set_middles[set_key] = model_middle.predict(set)
# for i in range(1): #valid_steps
#     x, y = valid_gen.__getitem__(i)
#     middles = model_middle.predict(x)
#     verif = model.predict(y)
#     verif_middle = model_middle.predict(verif)
#     mid = middles[0]
#     dst = valid_gen.l2_norm(mid.flatten(), verif_middle[0].flatten())
#     plt.imshow(x[0])
#     plt.title('one_shot_sample ' + str(dst))
#     plt.show()
#     pos = 1
#     fig = plt.figure(figsize=(13,16))
#     for sp_key in support_set_middles.keys():
#         sp_middles = support_set_middles[sp_key]
#
#         for s in range(len(sp_middles)):
#             fig.add_subplot(6, 5, pos)
#             pos+=1
#             dst = valid_gen.l2_norm(mid.flatten(), sp_middles[s].flatten())
#             plt.title(sp_key + ' ' + str(dst))
#             plt.imshow(support_set[sp_key][s])
#
# plt.show()


