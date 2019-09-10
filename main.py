## Imports
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import plot_model

from ae import AE
from data import DataGen

## Seeding
seed = 2019
np.random.seed = seed
tf.seed = seed

image_size = 256
channels = 3
train_path = "D:/HudecL/histology/All/"
database_path = "D:/HudecL/histology/benign/"
model_path = "D:/HudecL/DL_Models/histology/"
model_name = "AE_W_.h5"
epochs = 0
iterations = 20000
batch_size = 16
learning_rate = 1.5e-5

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

ae = AE(batch_size, units=200)
model_ae, model_encoder, latents, ios = ae.make_ae()

if os.path.exists(os.path.join(model_path, model_name)):
    try:
        model_ae.load_weights(os.path.join(model_path, model_name))
        print('Weights loaded from:')
        print(os.path.join(model_path,model_name))
    except ValueError as e:
        print('{0}'.format(e))
        print('Not loading old weights.')

# model_vae.add_loss(vae_loss)
optimizer = tf.keras.optimizers.Adam(lr=learning_rate) #, clipvalue=1000000.
model_ae.compile(optimizer=optimizer, metrics=["mae"], loss=ae.ae_loss())
model_ae.summary()
plot_model(model_ae, model_path + "model_ae.png", show_shapes=True)
plot_model(model_encoder, model_path+"model_ae_enc.png",show_shapes=True)
# model_middle.summary()

## Dataset for prediction
xv, yv = valid_gen.__getitem__(0)
result = model_ae.predict(xv)
pos = 1
fig = plt.figure(figsize=(10, 20))
# fig.suptitle('Iteration: {:d}'.format(i))
for j in range(len(xv)):
    rr = np.concatenate((xv[j], result[j]), axis=1)
    fig.add_subplot(8, 2, pos)
    pos += 1
    plt.title('Start test')
    plt.imshow(rr)
plt.savefig(os.path.join(model_path,'figure_start.png'))
plt.clf()
plt.cla()
plt.close()

if False:
    for i in range(iterations+1):
        xx,yy = train_gen.__getitem__(i % train_steps)
        loss, mae = model_ae.train_on_batch(xx, yy)
        print(i)
        print("Loss: {:f}".format(loss))
        print("Mean absolute error: {:f}".format(mae))
        if i % 200 == 0:
            result = model_ae.predict(xv)
            pos = 1
            fig = plt.figure(figsize=(10, 11))
            # fig.suptitle('Iteration: {:d}'.format(i))
            for j in range(len(xv)):
                rr = np.concatenate((xv[j], result[j]), axis=1)
                fig.add_subplot(4, 2, pos)
                pos += 1
                plt.title('I: {:d} id: {:d}'.format(i,j))
                plt.imshow(rr)
            plt.show()
        if i > 0 and i % 2000:
            print("Model saved.")
            model_ae.save_weights(model_path + model_name)

if epochs > 0:
    for e in range(epochs // 100 +1):
        model_ae.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=100)

        ## Save the Weights
        model_ae.save_weights(model_path + model_name)

        result = model_ae.predict(xv)
        pos = 1
        fig = plt.figure(figsize=(10, 20))
        # fig.suptitle('Iteration: {:d}'.format(i))
        for j in range(len(xv)):
            rr = np.concatenate((xv[j], result[j]), axis=1)
            fig.add_subplot(8, 2, pos)
            pos += 1
            plt.title('Epoch: {:d} id: {:d}'.format(e*100, j))
            plt.imshow(rr)
        plt.savefig(os.path.join(model_path,'figure_{:d}'.format(e) + '.png'))
        plt.clf()
        plt.cla()
        plt.close()


## Dataset for prediction
support_set = valid_gen.n_way_validation_support_set(5)
support_set_middles = {}
for set_key in support_set.keys():
    set = np.asarray(support_set[set_key])
    support_set_middles[set_key] = model_encoder.predict(set)
for i in range(valid_steps):
    x, y = valid_gen.__getitem__(i)
    middles = model_encoder.predict(x)
    verif = model_ae.predict(y)
    verif_middle = model_encoder.predict(verif)

    samples_path = os.path.join(model_path, 'samples')
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    for j in range(len(x)):
        index = j + i * len(x)
        mid = middles[j]
        dst = valid_gen.l2_norm(mid.flatten(), verif_middle[j].flatten())
        plt.title('one_shot_sample ' + str(dst))
        plt.imshow(x[j])
        plt.savefig(os.path.join(samples_path, '_{:d}_oneshot_fig.png'.format(index)))
        plt.clf()
        plt.cla()

        dsts = {}
        for sp_key in support_set_middles.keys():
            sp_middles = support_set_middles[sp_key]
            dsts[sp_key] = list()
            for s in range(len(sp_middles)):
                dsts[sp_key].append(valid_gen.l2_norm(mid.flatten(), sp_middles[s].flatten()))
            idx = np.argsort(dsts[sp_key])
            support_set[sp_key] = np.array(support_set[sp_key])[idx]
            dsts[sp_key] = np.array(dsts[sp_key])[idx]
        pos = 1
        fig = plt.figure(figsize=(16,21))
        for sp_key in support_set_middles.keys():
            sp_middles = support_set_middles[sp_key]
            for s in range(len(sp_middles)):
                fig.add_subplot(6, 5, pos)
                pos += 1
                plt.title(sp_key + ' ' + str(dsts[sp_key][s]))
                plt.imshow(support_set[sp_key][s])

        plt.savefig(os.path.join(samples_path, '_{:d}_n_way_validation.png'.format(index)))
        plt.clf()
        plt.cla()
        plt.close()
