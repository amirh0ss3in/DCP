import numpy as np

from keras.utils import to_categorical
import matplotlib.pyplot as plt


import keras
from keras.utils import to_categorical
import matplotlib.pyplot as plt
p=np.load("C:/Users/Amirhossein/Desktop/DCP_Negative_b.npy")
n=np.load("C:/Users/Amirhossein/Desktop/DCP_positive_b.npy")
#213
yp=list()
for i in range(len(p)):
    yp.append(1)

yn=list()
for i in range(len(n)):
    yn.append(0)

x=list()
for i in p:
    x.append(i)
for i in n:
    x.append(i)

y=list()
for i in yp:
    y.append(i)
for i in yn:
    y.append(i)

y=np.array(y)

x=np.array(x).astype(np.float32)
x/=255.0
X=list()
for i in x:
    #i=np.dot(i, [0.299, 0.587, 0.114])
    s=np.reshape(i,[25, 256, 256,3])
    X.append(s)
X=np.array(X).astype(np.float32)

x_train=np.array(X[:,12])
y_train=np.array(y[:])

y_train=to_categorical(y_train)

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

data = x_train
labels = y_train
batch_size=2
(trainX, testX, trainY, testY) = train_test_split(
                                data,labels, 
                                test_size=0.2, 
                                random_state=42) 

train_datagen = keras.preprocessing.image.ImageDataGenerator(
          zoom_range = 0.6,
          rotation_range=60,
          shear_range=0.2,
          width_shift_range = 0.2, 
          height_shift_range = 0.2,
          horizontal_flip = True,
          zca_whitening=True,
          fill_mode ='nearest') 

val_datagen = keras.preprocessing.image.ImageDataGenerator()


train_generator = train_datagen.flow(
        trainX, 
        trainY,
        batch_size=batch_size,
        shuffle=True)

validation_generator = val_datagen.flow(
                testX,
                testY,
                batch_size=batch_size) 


from keras.models import Sequential , model_from_json
from keras.layers import LeakyReLU, Dense, Conv2D , Flatten , MaxPooling2D , Dropout, LSTM , ConvLSTM2D , BatchNormalization ,Conv3D,MaxPooling3D, Input , ZeroPadding2D ,Convolution2D ,ZeroPadding3D
from keras import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

#print(x_test.shape)
#for i in range(4):
#    plt.imshow(x_test[i,12], interpolation='nearest')
#    plt.show()
# example of training a gan on mnist
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot

from keras.optimizers import Adam



input_shape = (256, 256, 3)

#Instantiate an empty model
model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#model.add(Dropout(0.5))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

model.fit_generator(
        train_generator,
        steps_per_epoch=len(trainX) // batch_size,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=len(testX) // batch_size)

#model.fit(trainX, trainY , shuffle=True,validation_data=(testX ,testY),batch_size=4 ,epochs=10)

