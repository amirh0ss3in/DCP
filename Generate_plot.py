
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



# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect image output from generator as input to discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and outputting a classification
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


latent_dim = 150

g_model=define_generator(latent_dim)
g_model.load_weights("g_model.h5")


d_model, c_model=define_discriminator()
d_model.load_weights("c_model.h5")


gan_model = define_gan(g_model, d_model)


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    z_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict(z_input)
    # create class labels
    y = zeros((n_samples, 1))
    return images, y

def summarize_performance( g_model, latent_dim, n_samples=1000):
    # prepare fake examples
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(100):
        # define subplot
        pyplot.subplot(10, 10, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename1 = 'generated_plot.png'
    pyplot.savefig(filename1,dpi = 2400)
    pyplot.close()
summarize_performance(g_model, latent_dim)