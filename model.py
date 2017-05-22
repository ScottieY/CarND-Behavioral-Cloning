import csv
import cv2
import numpy as np

import os

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                names = ['./data/IMG/'+batch_sample[i].split('/')[-1] for i in range(3)]
                cap_images = [cv2.imread(name) for name in names]
                angle = float(batch_sample[3])
                images.extend(cap_images)
                angles.extend([angle, angle + .2, angle - .2])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
input_shape=(160,320,3)
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')
