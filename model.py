import os
import sys
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

from preprocess import *

########################################################################################################################################
# Turn tuning on or off
TUNING = False

########################################################################################################################################

images,labels = readDataIn()

# Check Input Shape
print (images.shape)
print (labels.shape)

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(images,labels, test_size=0.10,random_state=534)




########################################################################################################################################
# Variables

batch_size = 128
nb_epoch = 5

########################################################################################################################################

if TUNING:

	print ("Fine tuning mode")

	with open('./model.json', 'r') as jfile:
	    model = model_from_json(json.load(jfile))

	model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

	model.load_weights('./model.h5')

	model.fit_generator(batch_generator(X_train, y_train, batch_size=batch_size),
	                        samples_per_epoch=X_train.shape[0]*2,
	                        nb_epoch=nb_epoch,
	                        nb_val_samples = X_val.shape[0]*2,
	                        validation_data=batch_generator(X_val, y_val, batch_size=batch_size))

	try:
		os.remove("./model.h5")
	except:
		pass

	model.save_weights("./model.h5")

	print("Saved weights to disk")

########################################################################################################################################

else:

	model = Sequential()

	model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(32, 64, 3)))

	model.add(Convolution2D(32, 3, 3, border_mode='same'))
	model.add(LeakyReLU())

	model.add(Convolution2D(32, 3, 3))
	model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.6))

	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(LeakyReLU())

	model.add(Convolution2D(64, 3, 3))
	model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.6))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(LeakyReLU())
	model.add(Dropout(0.6))

	model.add(Dense(1))
	model.summary()

	model.compile(loss='mean_squared_error', optimizer=Adam())

	model.fit_generator(batch_generator(X_train, y_train, batch_size=batch_size),
	                        samples_per_epoch=X_train.shape[0]*2,
	                        nb_epoch=nb_epoch,
	                        nb_val_samples = X_val.shape[0]*2,
	                        validation_data=batch_generator(X_val, y_val, batch_size=batch_size))

########################################################################################################################################

	model_json = model.to_json()

	try:
		os.remove("./model.json")
	except:
		pass


	with open("./model.json", "w") as json_file:
	    json.dump(model_json, json_file)

	try:
		os.remove("./model.h5")
	except:
		pass

	model.save_weights("./model.h5")

	print ("Saved to disk")

########################################################################################################################################