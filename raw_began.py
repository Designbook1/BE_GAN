import numpy as np
import keras.backend as k
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Reshape, Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.preprocessing import image
import os
import time
import numpy as np
from keras.utils import generic_utils

def l1Loss(y_true, y_pred):
    return k.mean(k.abs(y_true - y_pred))


def decoder():
    '''
    The decoder model is used as both half of the discriminator and as the generator.
    '''
    init_dim = 8 #Starting size from the paper
    
    mod_input = Input(shape=(100,))
    x = Dense(128*init_dim**2)(mod_input)
    x = Reshape((init_dim, init_dim, 128))(x)
    
    x = Conv2D(128, (3, 3), activation = 'elu', padding="same")(x)
    x = Conv2D(128, (3, 3), activation = 'elu', padding="same")(x)
    
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(128, (3, 3), activation = 'elu', padding="same")(x)
    x = Conv2D(128, (3, 3), activation = 'elu', padding="same")(x)
    
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(128, (3, 3), activation = 'elu', padding="same")(x)
    x = Conv2D(128, (3, 3), activation = 'elu', padding="same")(x)
        
    x = Conv2D(3, (3, 3), activation = 'elu', padding="same")(x)
    
    return Model(mod_input,x)

def encoder():
    '''
    The encoder model is the inverse of the decoder used in the autoencoder.
    '''
    
    mod_input = Input(shape=(32, 32, 3))
    x = Conv2D(3, (3, 3), activation = 'elu', padding="same")(mod_input)
    
    x = Conv2D(128, (3, 3), activation = 'elu', padding="same")(x)
    x = Conv2D(128, (3, 3), activation = 'elu', padding="same", strides=(2,2))(x)
    
    x = Conv2D(128, (3, 3), activation = 'elu', padding="same")(x)
    x = Conv2D(128, (3, 3), activation = 'elu', padding="same")(x)
    
    x = Flatten()(x)
    x = Dense(100)(x)
    
    return Model(mod_input,x)

def autoencoder():
    '''
    The autoencoder is used as the discriminator
    '''
    mod_input = Input(shape=(32, 32, 3))
    x = encoder()(mod_input)
    x = decoder()(x)
    
    return Model(mod_input, x)

def gan(generator, discriminator):
    '''
    Combined generator and discriminator
    Keyword arguments:
    generator -- The instantiated generator model
    discriminator -- The instantiated discriminator model
    '''
    mod_input = generator.input
    x = generator(mod_input)
    x = discriminator(x)

    return Model(mod_input, x)

#Training parameters
epochs = 30
batches_per_epoch = 150
batch_size = 16
gamma = .5 #between 0 and 1

adam = Adam(lr=0.00005) #lr: between 0.0001 and 0.00005

#Build models
generator = decoder()
discriminator = autoencoder()
gan = gan(generator,discriminator)
generator.compile(loss=l1Loss, optimizer=adam)
discriminator.compile(loss=l1Loss, optimizer=adam)
gan.compile(loss=l1Loss, optimizer=adam)

kLambda = 0.001
epsilon = k.epsilon()
k_t = k.epsilon()


def dataRescale(x):
    return x*2/255 - 1
#Load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
dataGenerator = image.ImageDataGenerator(preprocessing_function = dataRescale)
batchIterator = dataGenerator.flow(X_train, batch_size = batch_size)

for e in range(epochs):
    
		progbar = generic_utils.Progbar(batches_per_epoch*batch_size)
		start = time.time()
			
		for b in range(batches_per_epoch):
			zD = np.random.uniform(-1,1,(batch_size, 100))
			zG = np.random.uniform(-1,1,(batch_size*2, 100)) 
				            
			#Train D
			real = batchIterator.next()
			d_loss_real = discriminator.train_on_batch(real, real)
				
			gen = generator.predict(zD)
			weights = -k_t*np.ones(batch_size)
			d_loss_gen = discriminator.train_on_batch(gen, gen, weights)
				
			d_loss = d_loss_real + d_loss_gen
				
			#Train G
			discriminator.trainable = False
			target = generator.predict(zG)
			g_loss = gan.train_on_batch(zG, target)
			discriminator.trainable = True
				
			#Update k
			k_t = k_t + kLambda*(gamma*d_loss_real - g_loss)
			k_t = min(max(k_t, epsilon), 1)
				
			#Report Results
			m_global = d_loss + np.abs(gamma*d_loss_real - g_loss)
			progbar.add(batch_size, values=[("M", m_global),("Loss_D", d_loss),("Loss_G", g_loss),("k", k_t)])
	
			print('\nEpoch {}/{}, Time: {}'.format(e + 1, epochs, time.time() - start))
