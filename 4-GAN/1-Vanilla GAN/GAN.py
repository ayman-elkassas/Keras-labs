# import section
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
# Define input image dimensions
# The image shape is specified in target_size parameter
# larger images take longer to train

img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)


def build_generator():
    noise_shape = (100,)

    model = Sequential()
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    model.summary()

    noise = Input(shape=noise_shape)
    # generated image
    img = model(noise)

    return Model(noise, img)


def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    # the validity of the image (0 or 1) real or fake
    validity = model(img)
    return Model(img, validity)

# Now we can build our GAN model


def train(epochs, batch_size=128, save_interval=50):
    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()

    # convert images to float and scale between -1 and 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    # Add channels dimension as the input to our generator and discriminator has a shape of (28,28,1)
    X_train = np.expand_dims(X_train, axis=3)

    half_batch = int(batch_size / 2)

    for epoch in range(epochs):
        # Train the discriminator
        # Select a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]
        noise = np.random.normal(0, 1, (half_batch, 100))

        # Generate a half batch of fake images
        gen_imgs = generator.predict(noise)

        # Train the discriminator
        # losses of real and fake images
        # np.ones for real images and np.zeros for fake images
        d_loss_real = discriminator.train_on_batch(
            imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(
            gen_imgs, np.zeros((half_batch, 1)))

        # take the average of the two losses
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # train the generator
        # create noise vector as input for generator
        # based on normal distribution with mean=0 and std=1
        noise = np.random.normal(0, 1, (batch_size, 100))

        # the generator wants the discriminator to label the generated images as valid (1)
        valid_y = np.array([1] * batch_size)
        g_loss = combined.train_on_batch(noise, valid_y)

        # print the losses
        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
              (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        # save the losses to plot later
        if epoch % save_interval == 0:
            save_imgs(epoch)


# this function saves the generated images to the /output folder
def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)

    # rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    os.getcwd()
    fig.savefig("./images/%d.png" % epoch)
    plt.close()


optimizer = Adam(lr=0.0002, beta_1=0.5)
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

generator = build_generator()
generator.compile(loss='binary_crossentropy',
                  optimizer=optimizer)

z = Input(shape=(100,))
img = generator(z)
discriminator.trainable = False

valid = discriminator(img)

# here we combine the generator and the discriminator
# we are only training the generator
# the ultimate goal is to make the discriminator to label the generated images as valid (1)
# the combined model is stacked on top of the generator and the discriminator
# noise as input => generates images => passes through the discriminator => outputs a validity score

combined = Model(z, valid)
combined.compile(loss='binary_crossentropy',
                 optimizer=optimizer)

# save interval is the number of epochs after which the model is saved
train(epochs=30000, batch_size=32, save_interval=1000)

generator.save('generator_model_test.h5')
