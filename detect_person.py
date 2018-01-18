import pickle
import pandas as pd 
import numpy as np 
import csv
import os
import datetime
import sys
import gc
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Convolution2D, MaxPooling2D 
from keras.layers import GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalMaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

#MODE = 'test'

def find_checkpoint_file(folder):
	checkpoint_file = [f for f in os.listdir(folder) if 'checkpoint' in f]
	if len(checkpoint_file) == 0:
		return []
	modified_time = [os.path.getmtime(f) for f in checkpoint_file]
	return checkpoint_file[np.argmax(modified_time)]

# pickle_file will be closed at this point, preventing your from accessing it any further
X_train = []
y_train = [] 

with open("train.csv", "r") as input_file:
	read_file = csv.reader(input_file, delimiter='\t')
	for line in read_file:
		poo = []
		poo.append(line[0])
		X_train.append(line[:-1])
		y_train.append(line[-1:])

print(X_train[0:2])
print(y_train[0])
print(type(X_train))
X_train = np.asarray(X_train)

X_test = []
y_test = []
with open("test.csv", "r") as input_file:
	read_file = csv.reader(input_file, delimiter='\t')
	for line in read_file:
		#poo = [] 
		#poo.append(line[0])
		X_test.append(line[:-1])
		y_test.append(line[-1:])

#print(X_test[0])
#print(y_test[0])
X_test = np.asarray(X_test)

# normalize the data from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train/255.0
X_test = X_test/255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


#print(X_train.shape)
#print(X_test.shape)
############# 1D Convolutional network ###############
#import pdb;pdb.set_trace()
X_tr = np.reshape(X_train, (X_train.shape[0], 80, 80, 1))
X_te = np.reshape(X_test, (X_test.shape[0],80, 80, 1))

ep = 40	

model = Sequential()
    
model.add(Convolution2D(64, 5, 5, border_mode='valid',
                            input_shape=(80, 80, 1)))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))
      
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th')) 
model.add(Convolution2D(64, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th')) 
model.add(Convolution2D(64, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
     
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(Convolution2D(128, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(Convolution2D(128, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
     
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
     
model.add(Flatten())
model.add(Dense(1024))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(Dropout(0.2))
     
      
model.add(Dense(num_classes))
      
      
model.add(Activation('softmax'))

ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
                  optimizer=ada,
                  metrics=['accuracy'])

X_t, X_v, y_t, y_v = train_test_split(X_tr, y_train, 
						test_size=0.2, random_state=42)

# Image augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False) 

#train_gen = datagen.flow(X_t, y_t, batch_size=32)
datagen.fit(X_t)

test_datagen = ImageDataGenerator()
test_gen = test_datagen.flow(X_v, y_v, batch_size=32)

if MODE == 'train':
	early_stop = EarlyStopping(patience=5)
	model.fit_generator(datagen.flow(X_t, y_t, batch_size=32), samples_per_epoch=X_t.shape[0], epochs=ep, 
			validation_data = test_gen, validation_steps=X_v.shape[0]/8,
			callbacks=[early_stop, 
			ModelCheckpoint('checkpoint_best_epoch.hdf5',
			save_best_only=True,verbose=1)])


else:
	saved_weights = find_checkpoint_file('.')
	if len(saved_weights) == 0:
		print("Network hasn't been trained yet!")
		sys.exit()
	else:
		model.load_weights(saved_weights)
		scores = model.evaluate(X_te, y_test, verbose=0)
		print("Loaded saved weights.")
		print("Testing CNN model.. Keep patience !!")
		print("############################")
		print("CNN accuracy: %.2f%%" % (scores[1]*100))
		printz("############################")