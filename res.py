from keras.models import Model
from keras.layers import Dense,Flatten
from keras.applications import resnet50 
from keras import backend as K
import numpy as np

from keras.utils import to_categorical
import matplotlib.pyplot as plt


import keras
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


model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(256,256,3))
#model.summary(line_length=150)

flatten = Flatten()
new_layer2 = Dense(2, activation='softmax', name='my_dense_2')

inp2 = model.input
out2 = new_layer2(flatten(model.output))

model = Model(inp2, out2)
model.summary(line_length=150)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

model.fit_generator(
        train_generator,
        steps_per_epoch=len(trainX) // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=len(testX) // batch_size)


