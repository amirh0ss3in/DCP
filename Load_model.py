from numpy import expand_dims
from keras.models import load_model
import numpy as np
from keras.utils import to_categorical
number_of_slice=12
 
p=np.load("C:/Users/Amirhossein/Desktop/DCP_positive_b.npy")
n=np.load("C:/Users/Amirhossein/Desktop/DCP_negative_b.npy")
p=p[:,number_of_slice]
n=n[:,number_of_slice]
 
data=list()
for i in n:
    data.append(i)
for i in p:
    data.append(i)
 
import cv2
xx=[]
for i in data:    
    res = cv2.resize(i , dsize=(56, 56), interpolation=cv2.INTER_CUBIC)
    xx.append(res)
xx=np.array(xx)
 
data2=[]
for i in xx:
    i=np.dot(i, [0.299, 0.587, 0.114])
    s=np.reshape(i,[ 56, 56,1])
    data2.append(s)
data2=np.array(data2).astype(np.float32)
 
yn=list()
for i in range(len(n)):
    yn.append(0)
 
yp=list()
for i in range(len(p)):
    yp.append(1)
 
 
y=list()
for i in yn:
    y.append(i)
 
for i in yp:
    y.append(i)
 
ytr=np.array(y)

Xd=((data2 - 127.5) / 127.5).astype('float32')


from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from sklearn.metrics import classification_report
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda 
from keras.layers import Activation
from matplotlib import pyplot
from keras import backend
 
# custom activation function
def custom_activation(output):
    logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result
 
# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(56,56,1), n_classes=2):
    # image input
    in_image = Input(shape=in_shape)
    # downsample
    fe = Conv2D(256, (5,5), strides=(2,2), padding='same')(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(256, (5,5), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(256, (5,5), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.4)(fe)
    # output layer nodes
    fe = Dense(n_classes)(fe)

    # supervised output
    c_out_layer = Activation('softmax')(fe)
    # define and compile supervised discriminator model
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    # unsupervised output
    d_out_layer = Lambda(custom_activation)(fe)
    # define and compile unsupervised discriminator model
    d_model = Model(in_image, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    return d_model, c_model
 
# define the standalone generator model
def define_generator(latent_dim):
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = 256 * 14 * 14
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((14, 14, 256))(gen)
    # upsample to 14x14
    gen = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 28x28
    gen = Conv2DTranspose(256, (8,8), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = Conv2D(1, (14,14), activation='tanh', padding='same')(gen)
    # define model
    model = Model(in_lat, out_layer)
    return model

d_model, c_model=define_discriminator()


c_model.load_weights("c_model.h5")

_, acc = c_model.evaluate(Xd, ytr, verbose=0)
print('Classifier Accuracy: %.3f%%' % (acc * 100))

y_pred = c_model.predict(Xd, batch_size=2, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(ytr, y_pred_bool))