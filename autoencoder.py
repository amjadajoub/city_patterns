import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow.keras as keras

img_dim = 64  
n_components = 500
epochs = 50
n = 8
sample_n = 85
input_img = Input(shape=(img_dim * img_dim,))
encoded = Dense(n_components, activation='relu')(input_img)
decoded = Dense(img_dim * img_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(n_components,))
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

import numpy as np
X = np.load("X_500.npy")


x_train = X
x_train = x_train.astype('float32')
idx_1 = int(np.random.random_integers(0, sample_n-2, 1))
idx_2 = int(np.random.random_integers(sample_n, 2* sample_n-2, 1))
idx_3 = int(np.random.random_integers(2*sample_n, 3* sample_n-2, 1))
idx_4 = int(np.random.random_integers(3*sample_n, 4* sample_n-2, 1))
x_test = np.concatenate([X[idx_1:idx_1+2, :],
                        X[idx_2:idx_2+2, :],
                        X[idx_3:idx_3+2, :],
                        X[idx_4:idx_4+2, :]], axis=0).astype('float32')

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=256,
                shuffle=True)



encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

encoder.save(f"encoder_{n_components}")
decoder.save(f"decoder_{n_components}")
projected_images = encoder.predict(X)
np.save(f"projected_data_{n_components}.npy", projected_images)

import matplotlib.pyplot as plt


plt.figure(figsize=(4, 4))
counter = 0
for i in range(1,16,2):
    # display original
    ax = plt.subplot(4, 4, i)
    plt.imshow(x_test[counter].reshape(img_dim, img_dim))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(decoded_imgs[counter].reshape(img_dim, img_dim))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    counter = counter + 1
plt.show()

