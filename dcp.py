import numpy as np

from keras.utils import to_categorical

p=np.load("DCP_positive.npy")
n=np.load("DCP_negative.npy")

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
    i=np.dot(i, [0.299, 0.587, 0.114])
    s=np.reshape(i,[25, 256, 256,1])
    X.append(s)
X=np.array(X).astype(np.float32)

x_train=np.array(X[15:80])
y_train=np.array(y[15:80])

x_test=list()
for i in X[0:15]:
    x_test.append(i)
for i in X[80:]:
    x_test.append(i)
x_test=np.array(x_test)

y_test=list()
for i in y[0:15]:
    y_test.append(i)
for i in y[80:]:
    y_test.append(i)
y_test=np.array(y_test)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


from keras.models import Sequential , model_from_json
from keras.layers import Dense, Conv2D , Flatten , MaxPooling2D , Dropout, LSTM , ConvLSTM2D , BatchNormalization ,Conv3D,MaxPooling3D, Input
from keras import Model





## input layer
input_layer = Input((25,256,256,1))

## convolutional layers
conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)

## add max pooling to obtain the most imformatic features
pooling_layer1 = MaxPooling3D(pool_size=(2, 2, 2))(conv_layer2)

conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
pooling_layer2 = MaxPooling3D(pool_size=(2, 2, 2))(conv_layer4)

## perform batch normalization on the convolution outputs before feeding it to MLP architecture
pooling_layer2 = BatchNormalization()(pooling_layer2)
flatten_layer = Flatten()(pooling_layer2)

## create an MLP architecture with dense layers : 4096 -> 512 -> 10
## add dropouts to avoid overfitting / perform regularization
dense_layer1 = Dense(units=8, activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=4, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=2, activation='softmax')(dense_layer2)

## define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_test,y_test), batch_size=2, epochs=100)
