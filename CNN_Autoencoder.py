from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, UpSampling2D
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.models import Model
import numpy as np
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPOCHS = 100
INIT_LR = 0.01

# Load the MNIST data and normalise it
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
print (x_train.shape)
print (x_test.shape)

# Total features in a single image, 28x28
input_img = Input(shape=(28,28,1))
autoencoder = Sequential()


# Encoder

autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28,28,1)))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same'))

# Flatten encoding for visualization
autoencoder.add(Flatten())
autoencoder.add(Reshape((4, 4, 8)))

# Decoder
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(16, (3, 3), activation='relu'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

autoencoder.summary()

# Encoder model
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten_1').output)
encoder.summary()

# Defining the optimizer
#opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# Load the parameter & opt to the model
autoencoder.compile(loss="binary_crossentropy", optimizer="adam")

# Start training
autoencoder.fit(x_train, x_train,
                 epochs=EPOCHS,
                 batch_size=256,
                 validation_data=(x_test, x_test))
autoencoder.save("/output/auto.model")
#autoencoder.load_weights("auto.model")
encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
#plt.show()
plt.savefig('/output/foo.png')
