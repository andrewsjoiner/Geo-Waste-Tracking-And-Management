import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation,Conv2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
import time
from keras.models import model_from_json

epochs = 2

train_data_path = 'C:/Users/Hitesh/Desktop/HexaHive Train'
validation_data_path = 'C:/Users/Hitesh/Desktop/HexaHive Validatation'

"""
Parameters
"""
img_width, img_height = 120,120
batch_size = 32
samples_per_epoch = 500
validation_steps = 300
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 5
lr = 0.0004

# model = Sequential()
# #model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ="same", input_shape=(img_width, img_height, 3)))

# model.add(Conv2D(128,(3,3), input_shape = (64,64,3), activation='relu'))

# model.add(Conv2D(64,(3,3), activation='relu'))

# model.add(Conv2D(32,(3,3), activation='relu'))

# model.add(Conv2D(16,(3,3), activation='relu'))


# model.add(Activation("relu"))

# model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

# #model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))

# model.add(Activation("relu"))

# model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

model = Sequential()

model.add(Conv2D(128,(3,3), input_shape = (120,120,3), activation='relu'))

model.add(Conv2D(64,(3,3), activation='relu'))

model.add(Conv2D(32,(3,3), activation='relu'))

model.add(Conv2D(8,(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

# model.add(Activation("relu"))
# #model.add(Dropout(0.5))
# model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')
print("Class Index are:")
print("Indices:", train_generator.class_indices)

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),shuffle=True,
    batch_size=batch_size,
    class_mode='categorical')

"""
Tensorboard log
"""
# log_dir = './tf-log/'
# tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
# cbks = [tb_cb]

model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator, steps_per_epoch = 250,
    validation_steps=validation_steps)

model.save('HexaHive.h5')
print("Model Saved")

model.save_weights('HexaHiveweight.h5')
print("Weights Saved")

model_json = model.to_json()
with open("HexaHiveJson.json", "w") as json_file:
    json_file.write(model_json)
print("json File Saved")