#!/bin/env python

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
 
train_dir = "data/dss-data-cat/train"
test_dir = "data/dss-data-cat/test"

mnist_conv_base = models.load_model('mnist-base.h5')

"""
Base mode for transfer learning is configured below. 
1. Layers are shaved off.
2. Layers are set to be none trainable
"""

print('TOTAL LAYERS: ', len(mnist_conv_base.layers))
mnist_conv_base.trainable = False
mnist_conv_base.summary()
n = 6
for i in range(n):
    mnist_conv_base.layers.pop()

mnist_conv_base.add(layers.Flatten())
mnist_conv_base.summary()

train_data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=180
        )

test_data_gen = ImageDataGenerator(
        rescale=1./255
        )

train_generator = train_data_gen.flow_from_directory(
        train_dir,
        target_size=(28, 28),
        batch_size=20,
        color_mode="grayscale"
        )

test_generator = test_data_gen.flow_from_directory(
        test_dir,
        target_size=(28, 28),
        batch_size=20,
        color_mode="grayscale"
        )

model = models.Sequential()
model.add(mnist_conv_base)
#model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(9, activation="softmax"))

model.summary()

"""
model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.Adadelta(),
              metrics=['acc'])

    Total amount of data in train: 8168

    eight: 856
    five: 845
    four: 970
    nine: 912
    one: 741
    seven: 960
    six: 1197
    three: 843
    two: 844

    Total amount of data in test: 91

    eight: 10
    five: 10
    four: 10
    nine: 11
    one: 10
    seven: 10
    six: 10
    three: 10
    two: 10
"""
history = model.fit_generator(
        train_generator,
        steps_per_epoch=409,
        epochs=10,
        validation_data=test_generator,
        validation_steps=5)


