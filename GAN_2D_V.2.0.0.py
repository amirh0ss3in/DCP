from keras.models import Model
from keras.layers import Dense,Flatten
from keras.applications import vgg16 
from keras import backend as K


from keras.utils import to_categorical



import keras


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
black = (0,0,0)
white = (255,255,255)
threshold = (160,160,160)


number_of_slice=12

p=np.load("/content/drive/My Drive/DCP/DCP_negative_b.npy")
n=np.load("/content/drive/My Drive/DCP/DCP_negative_b.npy")
p=p[:,number_of_slice]
n=n[:,number_of_slice]

data=list()
for i in n:
    data.append(i)
for i in p:
    data.append(i)

#n=0 , p=1

#y=list()
#for i in range(len(n)):
#    y.append(0)
#for i in range(len(p)):
#    y.append(1)
#y=np.array(y)


enahnced_data=list()
for i in data:
    # Open input image in grayscale mode and get its pixels.
    img = Image.fromarray(i).convert("LA")
    pixels = img.getdata()
    newPixels = [] 
    for pixel in pixels:
        if pixel < threshold:
            newPixels.append(black)
        else:
            newPixels.append(white)
    newImg = Image.new("RGB",img.size)
    newImg.putdata(newPixels)
    newImg=np.array(newImg)
    enahnced_data.append(newImg)
enahnced_data=np.array(enahnced_data)


#y_train=to_categorical(y)


#print(enahnced_data.shape)
#print(len(y))
#plt.imshow(l[0], interpolation='nearest')
#plt.show()




from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split



x=enahnced_data/255.0 
X=list()
for i in x:
    i=np.dot(i, [0.299, 0.587, 0.114])
    s=np.reshape(i,[256, 256,1])
    X.append(s)
X=np.array(X).astype(np.float32)

x_train=np.array(X)

from keras.layers import GaussianNoise
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

# define the standalone discriminator model
def define_discriminator(in_shape=(256,256,1)):
	model = Sequential()
	model.add(Input(shape=in_shape))
	model.add(GaussianNoise(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.5))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.5))
	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
a=len(x_train[60:120])
# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = a * 64 * 64
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((64, 64, a)))
	# upsample to 14x14
	model.add(Conv2DTranspose(a, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization())
	#upsample to 28x28
	model.add(Conv2DTranspose(a, (64,64), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization())
	model.add(Conv2D(1, (64,64), activation='tanh', padding='same'))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.00002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model


# load and prepare mnist training images
def load_real_samples():
	X= x_train[60:120]
	return X

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	#filename = 'generator_model_%03d.h5' % (epoch + 1)
	#g_model.save(filename)

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=4):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# create training set for the discriminator
			X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
			# update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
		# evaluate the model performance, sometimes
		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = x_train[60:120]
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)

 
