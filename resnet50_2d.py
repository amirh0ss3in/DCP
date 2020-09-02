from keras.models import Model
from keras.layers import Dense,Flatten
from keras.applications import resnet50 
from keras import backend as K

from keras.utils import to_categorical
import keras
import matplotlib.pyplot as plt

import numpy as np

trainX=np.load("C:/Users/Amirhossein/Desktop/trainX.npy")
testX=np.load("C:/Users/Amirhossein/Desktop/testX.npy")
trainY=np.load("C:/Users/Amirhossein/Desktop/trainY.npy")
testY=np.load("C:/Users/Amirhossein/Desktop/testY.npy")


from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split


""" 
train_datagen = keras.preprocessing.image.ImageDataGenerator(
          zoom_range = 0.6,
          rotation_range=60,
          shear_range=0.2,
          width_shift_range = 0.2, 
          height_shift_range = 0.2,
          horizontal_flip = True,
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
                batch_size=batch_size)  """


model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(256,256,3))
#model.summary(line_length=150)

flatten = Flatten()
new_layer2 = Dense(2, activation='softmax', name='my_dense_2')

inp2 = model.input
out2 = new_layer2(flatten(model.output))

model = Model(inp2, out2)
model.summary(line_length=150)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

model.fit(trainX,trainY , validation_data=(testX,testY ), batch_size=16, epochs=20 )

""" model.fit_generator(
        train_generator,
        steps_per_epoch=len(trainX) // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=len(testX) // batch_size)

 """
