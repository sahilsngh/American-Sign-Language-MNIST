# Created by Sahil Chauhan
import os
import time
import numpy as np
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

def train_model():

	model = Sequential() #model.add

	model.add(Conv2D(32, (3,3), input_shape=(28, 28, 1)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(32, (3,3)))
	model.add(Activation("relu"))
	
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Activation("relu"))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation("relu"))

	model.add(Dense(24))
	model.add(Activation("softmax"))

	# Compile Model. 
	model.compile(
	    optimizer='adam',
	    loss='sparse_categorical_crossentropy',
	    metrics=['accuracy']
	)
	return model

# Download the dataset from 'https://www.kaggle.com/datamunge/sign-language-mnist/download'
# and put it under "dataset/" directory. 
# *(create one if you don't have one in your working directory)*

path = f'{os.getcwd()}/dataset/'.replace('\\', '/')
train_path = path + "sign_minst_train.csv"
test_path = path + "sign_minst_test.csv"

data_train=pd.read_csv(train_path)
data_test=pd.read_csv(test_path)

data_train.info()
data_test.info()

# Data splitting and PreProcessing 

training_images = data_train.iloc[:,1:].values
training_labels = data_train.iloc[:,0].values

testing_images = data_test.iloc[:,1:].values
testing_labels = data_test.iloc[:,0].values

training_images = training_images.reshape(-1,28,28,1)
testing_images = testing_images.reshape(-1,28,28,1)


print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    rescale=1 / 255
)
    
# Keep These
print(f"output should be (27455, 28, 28, 1) : {training_images.shape}")
print(f"output should be (7172, 28, 28, 1) : {testing_images.shape}")
    

# To start Tensorboard open cmd in working directory and type 
# tensorboard --logdir=path_to_your_logs

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs', histogram_freq=0, write_graph=True,
    write_images=False, update_freq='epoch', profile_batch=2,
    embeddings_freq=0, embeddings_metadata=None)
print('Your Tensorboard will be running at |tensorboard --logdir=logs|')

# Creating a Training_model Instance
model = train_model()

# Display the model's architecture
model.summary()

# Training the model
history = model.fit(train_datagen.flow(training_images, training_labels, batch_size=32),
                              steps_per_epoch=len(training_images) / 32,
                              epochs=20,
                              callbacks=[tensorboard_callback],
                              validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=32),
                              validation_steps=len(testing_images) / 32)

model.evaluate(testing_images, testing_labels, verbose=0)

model.save("asl_24.h5")


# If you don't want to use Tensorboard use matplot instead

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(len(acc))
# plt.plot(epochs, acc, 'r', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend(loc=0)
# plt.figure()
# plt.plot(epochs, loss, 'r', label='Training Loss')
# plt.plot(epochs, val_loss, 'b', label='Validation Loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

