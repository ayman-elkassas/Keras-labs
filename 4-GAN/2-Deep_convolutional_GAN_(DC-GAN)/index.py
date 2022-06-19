# import lib area
# import os
# import gdown
# import zipfile

# import os and create folder celeba_gan
# os.makedirs('celeba_gan', exist_ok=True)
# url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
# output = "celeba_gan/data.zip"
# # download the data using gdown
# gdown.download(url, output, quiet=False)
# # unzip the data
# with zipfile.ZipFile("celeba_gan/data.zip", 'r') as zip_ref:
#     zip_ref.extractall("celeba_gan")

# preprocessing with batch size of 32 using keras
from cProfile import label
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import tqdm

dataset = keras.preprocessing.image_dataset_from_directory(
    'celeba_gan/img_align_celeba',
    label_mode=None,
    image_size=(64, 64),
    batch_size=32,
    shuffle=True)

discriminator = keras.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(64, kernel_size=4, strides=2, padding='same'),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
    layers.LeakyReLU(alpha=0.2),
    layers.Flatten(),
    # dropout layer
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

print(discriminator.summary())

latent_dim = 128

# build the generator
generator = keras.Sequential([
    layers.Input(shape=(latent_dim,)),
    layers.Dense(8*8*128, use_bias=False),
    layers.Reshape(target_shape=(8, 8, 128)),
    # transpose
    layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same'),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(3, kernel_size=5,
                  padding='same', activation='sigmoid')
])

generator.summary()

opt_discriminator = keras.optimizers.Adam(1e-4)
opt_generator = keras.optimizers.Adam(1e-4)
# loss function
loss_fun = keras.losses.BinaryCrossentropy()

for epoch in range(10):
    for idx, real in enumerate((tqdm(dataset))):
        # batch size of real
        batch_size = real.shape[0]
        # generate latent vector
        random_latent_vector = np.random.normal(size=(batch_size, latent_dim))
        # generate fake data
        fake = generator(random_latent_vector)

        if idx % 100 == 0:
            img = keras.preprocessing.image.array_to_img(fake[0])
            img.save('celeba_gan/fake_img_'+str(idx)+'.png')

        # train the discriminator
        with tf.GradientTape() as disc_tape:
            # discriminator loss
            real_loss = loss_fun(tf.ones((batch_size, 1)), discriminator(real))
            fake_loss = loss_fun(
                tf.zeros((batch_size, 1)), discriminator(fake))

            # average of discriminator loss
            d_loss = 0.5 * (real_loss + fake_loss)

        # compute new weights
        grads = disc_tape.gradient(d_loss, discriminator.trainable_weights)
        opt_discriminator.apply_gradients(
            zip(grads, discriminator.trainable_weights))

        # train the generator
        with tf.GradientTape() as gen_tape:
            fake = generator(random_latent_vector)
            # output of discriminator
            discriminator_output = discriminator(fake)
            # generator loss
            g_loss = loss_fun(tf.ones((batch_size, 1)), discriminator_output)

        grads = gen_tape.gradient(g_loss, generator.trainable_weights)
        opt_generator.apply_gradients(
            zip(grads, generator.trainable_weights))
