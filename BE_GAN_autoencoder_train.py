#!/usr/bin/python 
# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, merge
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda
from keras.layers.advanced_activations import  ELU
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
from PIL import Image
import argparse
import math
import os
from scipy import ndimage
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from keras import optimizers
import random
import matplotlib.pyplot as plt 
from scipy.misc import imsave
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as k
from keras.utils import generic_utils
import setGPU
import time

# Parameter initialization
img_w = 224
img_h = 224
channels = 3
batch_size = 16
weight_decay = 5e-4
classes = 332
gamma = 0.5
kLambda = 0.001
epsilon = k.epsilon()
k_t = k.epsilon()
Current_Refined = np.zeros([batch_size, img_w, img_h, channels], dtype=np.float32)

# Read training data
# GT file list
GT_file = open("/home/zhaojian/Keras/GAN_APPLE/lists/IJBA_split1_finetune.txt", "r")
GT_lines = GT_file.readlines()
GT_file.close()
N_GT = len(GT_lines)
GT_file_list = []
for i in range(N_GT):
    GT_file_list.append(GT_lines[i].split()[0])
# Syn file list + Syn label
Syn_file = open("/home/zhaojian/Keras/GAN_APPLE/lists/IJBA_split1_syn_rand.txt", "r")
Syn_lines = Syn_file.readlines()
Syn_file.close()
N_Syn = len(Syn_lines)
Syn_file_list = []
Syn_label = np.zeros([N_Syn, 1], dtype=np.int)
for i in range(N_Syn):
    Syn_file_list.append(Syn_lines[i].split()[0])
    Syn_label[i] = int(Syn_lines[i].split()[1])
Syn_label = np_utils.to_categorical(Syn_label)

test = cv2.resize(cv2.imread(Syn_file_list[0]), (img_w, img_h)).astype(np.float32)

# Load the fine-tuned VGGFACE model structure and best weights
# Load model structure
json_file = open('/home/zhaojian/Keras/GAN_APPLE/models/VGGFACE/keras_vggface_IJBA.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
VGGFACE = model_from_json(loaded_model_json)
# load best weights
VGGFACE.load_weights("/home/zhaojian/Keras/GAN_APPLE/models/VGGFACE/weights-improvement-14-0.84.hdf5")
VGGFACE.trainable = False # freeze VGGFACE to tune GN's parameters

# 01-Generator network definition
def Generator_Net():    
    GN_inputs = Input(shape=(img_w, img_h, channels))
    
    GN_conv_0 = Conv2D(3, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(GN_inputs)
    GN_conv_0 = ELU()(GN_conv_0)
    
    GN_conv_1 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(GN_conv_0)
    GN_conv_1 = ELU()(GN_conv_1)
    GN_conv_2 = Conv2D(128, (3, 3), padding='same', strides = (2, 2), kernel_regularizer=l2(weight_decay))(GN_conv_1)
    GN_conv_2 = ELU()(GN_conv_2)   

    GN_conv_3 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(GN_conv_2)
    GN_conv_3 = ELU()(GN_conv_3)
    GN_conv_4 = Conv2D(256, (3, 3), padding='same', strides = (2, 2), kernel_regularizer=l2(weight_decay))(GN_conv_3)
    GN_conv_4 = ELU()(GN_conv_4)
    
    GN_conv_5 = Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(GN_conv_4)
    GN_conv_5 = ELU()(GN_conv_5)
    GN_conv_6 = Conv2D(384, (3, 3), padding='same', strides = (2, 2), kernel_regularizer=l2(weight_decay))(GN_conv_5)
    GN_conv_6 = ELU()(GN_conv_6)
    
    GN_conv_7 = Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(GN_conv_6)
    GN_conv_7 = ELU()(GN_conv_7)
    GN_conv_8 = Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(GN_conv_7)
    GN_conv_8 = ELU()(GN_conv_8)
    
    GN_conv_9 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(GN_conv_8)
    GN_conv_9 = ELU()(GN_conv_9)
    GN_conv_10 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', input_shape=(28, 28, 3), kernel_regularizer=l2(weight_decay))(GN_conv_9)
    GN_conv_10 = ELU()(GN_conv_10)
    
    GN_conv_11 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(GN_conv_10)
    GN_conv_11 = ELU()(GN_conv_11)
    GN_conv_12 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', input_shape=(56, 56, 3), kernel_regularizer=l2(weight_decay))(GN_conv_11)
    GN_conv_12 = ELU()(GN_conv_12)
    
    GN_conv_13 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(GN_conv_12)
    GN_conv_13 = ELU()(GN_conv_13)
    GN_conv_14 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', input_shape=(128, 128, 3), kernel_regularizer=l2(weight_decay))(GN_conv_13)
    GN_conv_14 = ELU()(GN_conv_14)
    
    GN_conv_15 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(GN_conv_14)
    GN_conv_15 = ELU()(GN_conv_15)
    GN_conv_16 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(GN_conv_15)
    GN_conv_16 = ELU()(GN_conv_16)
    
    GN_outputs = Conv2D(3, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(GN_conv_16)
    GN_outputs = ELU()(GN_outputs)
    
    GN = Model(outputs=GN_outputs, inputs=GN_inputs) 
    
    return GN

# 02-Discriminator network definition
def Discriminator_Net():  
    DN_inputs = Input(shape=(img_w, img_h, channels))
    
    DN_conv_0 = Conv2D(3, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(DN_inputs)
    DN_conv_0 = ELU()(DN_conv_0)
    
    DN_conv_1 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(DN_conv_0)
    DN_conv_1 = ELU()(DN_conv_1)
    DN_conv_2 = Conv2D(128, (3, 3), padding='same', strides = (2, 2), kernel_regularizer=l2(weight_decay))(DN_conv_1)
    DN_conv_2 = ELU()(DN_conv_2)   

    DN_conv_3 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(DN_conv_2)
    DN_conv_3 = ELU()(DN_conv_3)
    DN_conv_4 = Conv2D(256, (3, 3), padding='same', strides = (2, 2), kernel_regularizer=l2(weight_decay))(DN_conv_3)
    DN_conv_4 = ELU()(DN_conv_4)
    
    DN_conv_5 = Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(DN_conv_4)
    DN_conv_5 = ELU()(DN_conv_5)
    DN_conv_6 = Conv2D(384, (3, 3), padding='same', strides = (2, 2), kernel_regularizer=l2(weight_decay))(DN_conv_5)
    DN_conv_6 = ELU()(DN_conv_6)
    
    DN_conv_7 = Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(DN_conv_6)
    DN_conv_7 = ELU()(DN_conv_7)
    DN_conv_8 = Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(DN_conv_7)
    DN_conv_8 = ELU()(DN_conv_8)
    
    DN_conv_9 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(DN_conv_8)
    DN_conv_9 = ELU()(DN_conv_9)
    DN_conv_10 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', input_shape=(28, 28, 3), kernel_regularizer=l2(weight_decay))(DN_conv_9)
    DN_conv_10 = ELU()(DN_conv_10)
    
    DN_conv_11 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(DN_conv_10)
    DN_conv_11 = ELU()(DN_conv_11)
    DN_conv_12 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', input_shape=(56, 56, 3), kernel_regularizer=l2(weight_decay))(DN_conv_11)
    DN_conv_12 = ELU()(DN_conv_12)
    
    DN_conv_13 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(DN_conv_12)
    DN_conv_13 = ELU()(DN_conv_13)
    DN_conv_14 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', input_shape=(128, 128, 3), kernel_regularizer=l2(weight_decay))(DN_conv_13)
    DN_conv_14 = ELU()(DN_conv_14)
    
    DN_conv_15 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(DN_conv_14)
    DN_conv_15 = ELU()(DN_conv_15)
    DN_conv_16 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(DN_conv_15)
    DN_conv_16 = ELU()(DN_conv_16)
    
    DN_outputs = Conv2D(3, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(DN_conv_16)
    DN_outputs = ELU()(DN_outputs)
    
    DN = Model(outputs=DN_outputs, inputs=DN_inputs) 
    
    return DN

# 03-Tune Gererator network's parameters according to Discriminator network's output
def Generator_on_Discriminator(GN, DN):
    GoD = Sequential()
    GoD.add(GN)
    DN.trainable = False # freeze DN to tune GN's parameters
    GoD.add(DN)
    return GoD

# 04-Tune Gererator network's parameters according to VGGFACE network's output: Generator+VGGFACE->multi-class cross-entropy loss
def Generator_on_VGGFACE(GN, VGGFACE):
    GoV = Sequential()
    GoV.add(GN)
    VGGFACE.trainable = False # Freeze VGGFACE to tune GN's parameters
    GoV.add(VGGFACE)
    return GoV

# Initialize three models
DN = Discriminator_Net()
GN = Generator_Net()
GoD = Generator_on_Discriminator(Generator_Net(), Discriminator_Net())
GoV = Generator_on_VGGFACE(Discriminator_Net(), VGGFACE)

# Set basic parameters for training, first train with lr=0.0002
lr = 0.00005
nb_epoch = 150
steps_per_epoch = N_Syn/batch_size

optim = Adam(lr=lr)

DN.compile(loss='mean_absolute_error', optimizer=optim)
GN.compile(loss='mean_absolute_error', optimizer=optim)
GoD.compile(loss='mean_absolute_error', optimizer=optim)
GoV.compile(loss='categorical_crossentropy', optimizer=optim)

# Training process
print("Start training process... *Basic parameters: lr->%f, nb_epoch->%d, steps_per_epoch->%d." % (lr, nb_epoch, steps_per_epoch))
for e in range(nb_epoch):
    progbar = generic_utils.Progbar(steps_per_epoch*batch_size)
    start = time.time()
    for s in range(steps_per_epoch):
        GT_Real_X_batch = np.zeros((batch_size, img_w, img_h, channels), dtype=np.float32)
        GT_Real_img_file = random.sample(GT_file_list, batch_size)
        for k in range(batch_size):
            GT_Real_img = cv2.resize(cv2.imread(GT_Real_img_file[k]), (img_w, img_h)).astype(np.float32)
            GT_Real_X_batch[k, ...] = GT_Real_img
    	GoV_X_batch = np.zeros((batch_size, img_w, img_h, channels), dtype=np.float32)
        GoV_Y_batch  = np.zeros((batch_size, classes), dtype=np.int)
        Input_Syn_img_file = Syn_file_list[s*batch_size:(s+1)*batch_size]
        GoV_Y_batch = Syn_label[s*batch_size:(s+1)*batch_size, ...]
        for k in range(batch_size):
            Input_Syn_img = cv2.resize(cv2.imread(Input_Syn_img_file[k]), (img_w, img_h)).astype(np.float32)
            GoV_X_batch[k, ...] = Input_Syn_img
    	# Train DN
	DN_X_batch = GT_Real_X_batch
	DN_Y_batch = GT_Real_X_batch
	DN_loss_real = DN.train_on_batch(DN_X_batch, DN_Y_batch)			
	Current_Refined = GN.predict(GoV_X_batch)
	DN_X_batch = Current_Refined
	DN_Y_batch = Current_Refined
	weights = -k_t*np.ones(batch_size)
	DN_loss_refine = DN.train_on_batch(DN_X_batch, DN_Y_batch,  weights)	
	DN_loss = DN_loss_real + DN_loss_refine
	# Train GN
	DN.trainable = False
	GN_X_batch = GoV_X_batch
	GN_Y_batch = GoV_X_batch
	GN_loss_l1 = GN.train_on_batch(GN_X_batch, GN_Y_batch)
	GN_loss_iden = GoV.train_on_batch(GoV_X_batch, GoV_Y_batch)
	GoD_X_batch = GoV_X_batch
	GoD_Y_batch = GN.predict(GoV_X_batch)
        GN_loss_refine = GoD.train_on_batch(GoD_X_batch, GoD_Y_batch) 
	DN.trainable = True
	#Update k_t
	k_t = k_t + kLambda*(gamma*DN_loss_real - GN_loss_refine)
	k_t = min(max(k_t, epsilon), 1)
	#Report Results
	M_global = DN_loss + np.abs(gamma*DN_loss_real - GN_loss_refine)
	progbar.add(batch_size, values=[("M_global", M_global),("DN_loss", DN_loss),("GN_loss_l1", GN_loss_l1),("GN_loss_iden", GN_loss_iden),("GN_loss_refine", GN_loss_refine),("k_t", k_t)])  
    print('\nEpoch {}/{}, Time: {}'.format(e, nb_epoch, time.time() - start))
    # Save GN and DN model every epoch
    GN.save_weights(('/home/zhaojian/Keras/GAN_APPLE/models/Generator/BE_GAN_autoencoder/'+'lr_'+str(lr)+'epoch_'+str(e)+'_GN'), True)
    DN.save_weights(('/home/zhaojian/Keras/GAN_APPLE/models/Discriminator/BE_GAN_autoencoder/'+'lr_'+str(lr)+'epoch_'+str(e)+'_DN'), True)
    # Save the first refined image as internal result every epoch
    refined = GN.predict(test.reshape(1,img_w,img_h,channels)).reshape(img_w,img_h,channels)[:,:,np.array([2,1,0])]
    refined = Image.fromarray(refined.astype(np.uint8))
    imsave(('/home/zhaojian/Keras/GAN_APPLE/internal_results/BE_GAN_autoencoder/'+'lr_'+str(lr)+'epoch_'+str(e)+'_'+'28798_00390.png'), refined)

