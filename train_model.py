from keras.datasets import mnist
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))

X_train.shape  # (60000, 28, 28, 1)

X_train = X_train / 255.
X_test = X_test / 255.

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

model = tf.keras.Sequential([
    layers.Conv2D(filters=32,
                  kernel_size=(5, 5),
                  activation='relu',
                  input_shape=(28, 28, 1)),

    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Dropout(0.2),

    layers.Flatten(),

    layers.Dense(units=128,
                 activation='relu'),

    layers.Dense(units=10,
                 activation='softmax')
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.fit(X_train,
          y_train,
          epochs=10,
          batch_size=200,
          shuffle=True)

model.evaluate(X_test, y_test)

model.save("my_model.h5")
