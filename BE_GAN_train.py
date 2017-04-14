#!/usr/bin/python 
# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, merge
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Deconvolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda
from keras.layers.advanced_activations import  ELU, LeakyReLU
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
batch_size = 4
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

# Generator network definition
def Generator_Net():
    # Input
    inputs = Input(shape=(img_w, img_h, channels))
    conv_0 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(inputs)
    conv_0 = LeakyReLU(alpha=0.3)(conv_0)
    conv_0 = BatchNormalization()(conv_0)
    
    # Resnet block1
    conv_1_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_0)
    conv_1_1 = LeakyReLU(alpha=0.3)(conv_1_1)
    conv_1_1 = BatchNormalization()(conv_1_1)
    conv_1_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_1_1)
    res_b1 = merge([conv_0, conv_1_2], mode='sum')
    res_b1 = LeakyReLU(alpha=0.3)(res_b1)
    res_b1 = BatchNormalization()(res_b1)

    # Resnet block2
    conv_2_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b1)
    conv_2_1 = LeakyReLU(alpha=0.3)(conv_2_1)
    conv_2_1 = BatchNormalization()(conv_2_1)
    conv_2_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_2_1)
    res_b2 = merge([res_b1, conv_2_2], mode='sum')
    res_b2 = LeakyReLU(alpha=0.3)(res_b2)
    res_b2 = BatchNormalization()(res_b2)

    # Resnet block3
    conv_3_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b2)
    conv_3_1 = LeakyReLU(alpha=0.3)(conv_3_1)
    conv_3_1 = BatchNormalization()(conv_3_1)
    conv_3_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_3_1)
    res_b3 = merge([res_b2, conv_3_2], mode='sum')
    res_b3 = LeakyReLU(alpha=0.3)(res_b3)
    res_b3 = BatchNormalization()(res_b3)

    # Resnet block4
    conv_4_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b3)
    conv_4_1 = LeakyReLU(alpha=0.3)(conv_4_1)
    conv_4_1 = BatchNormalization()(conv_4_1)
    conv_4_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_4_1)
    res_b4 = merge([res_b3, conv_4_2], mode='sum')
    res_b4 = LeakyReLU(alpha=0.3)(res_b4)
    res_b4 = BatchNormalization()(res_b4)
    
    # Resnet block5
    conv_5_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b4)
    conv_5_1 = LeakyReLU(alpha=0.3)(conv_5_1)
    conv_5_1 = BatchNormalization()(conv_5_1)
    conv_5_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_5_1)
    res_b5 = merge([res_b4, conv_5_2], mode='sum')
    res_b5 = LeakyReLU(alpha=0.3)(res_b5)
    res_b5 = BatchNormalization()(res_b5)
    
    # Resnet block6
    conv_6_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b5)
    conv_6_1 = LeakyReLU(alpha=0.3)(conv_6_1)
    conv_6_1 = BatchNormalization()(conv_6_1)
    conv_6_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_6_1)
    res_b6 = merge([res_b5, conv_6_2], mode='sum')
    res_b6 = LeakyReLU(alpha=0.3)(res_b6)
    res_b6 = BatchNormalization()(res_b6)

    # Resnet block7
    conv_7_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b6)
    conv_7_1 = LeakyReLU(alpha=0.3)(conv_7_1)
    conv_7_1 = BatchNormalization()(conv_7_1)
    conv_7_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_7_1)
    res_b7 = merge([res_b6, conv_7_2], mode='sum')
    res_b7 = LeakyReLU(alpha=0.3)(res_b7)
    res_b7 = BatchNormalization()(res_b7)

    # Resnet block8
    conv_8_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b7)
    conv_8_1 = LeakyReLU(alpha=0.3)(conv_8_1)
    conv_8_1 = BatchNormalization()(conv_8_1)
    conv_8_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_8_1)
    res_b8 = merge([res_b7, conv_8_2], mode='sum')
    res_b8 = LeakyReLU(alpha=0.3)(res_b8)
    res_b8 = BatchNormalization()(res_b8)

    # Resnet block9
    conv_9_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b8)
    conv_9_1 = LeakyReLU(alpha=0.3)(conv_9_1)
    conv_9_1 = BatchNormalization()(conv_9_1)
    conv_9_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_9_1)
    res_b9 = merge([res_b8, conv_9_2], mode='sum')
    res_b9 = LeakyReLU(alpha=0.3)(res_b9)
    res_b9 = BatchNormalization()(res_b9)

    # Resnet block10
    conv_10_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b9)
    conv_10_1 = LeakyReLU(alpha=0.3)(conv_10_1)
    conv_10_1 = BatchNormalization()(conv_10_1)
    conv_10_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_10_1)
    res_b10 = merge([res_b9, conv_10_2], mode='sum')
    res_b10 = LeakyReLU(alpha=0.3)(res_b10)
    res_b10 = BatchNormalization()(res_b10)

    # Output
    outputs = Conv2D(3, (1, 1), padding='same', activation='sigmoid', kernel_regularizer=l2(weight_decay))(res_b10)
    outputs = Lambda(lambda x: x * 255)(outputs)
    GN = Model(outputs=outputs, inputs=inputs)         

    return GN  

# 02-Discriminator network definition
def Discriminator_Net():
    # Input
    inputs = Input(shape=(img_w, img_h, channels))
    conv_0 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(inputs)
    conv_0 = LeakyReLU(alpha=0.3)(conv_0)
    conv_0 = BatchNormalization()(conv_0)
    
    # Resnet block1
    conv_1_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_0)
    conv_1_1 = LeakyReLU(alpha=0.3)(conv_1_1)
    conv_1_1 = BatchNormalization()(conv_1_1)
    conv_1_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_1_1)
    res_b1 = merge([conv_0, conv_1_2], mode='sum')
    res_b1 = LeakyReLU(alpha=0.3)(res_b1)
    res_b1 = BatchNormalization()(res_b1)

    # Resnet block2
    conv_2_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b1)
    conv_2_1 = LeakyReLU(alpha=0.3)(conv_2_1)
    conv_2_1 = BatchNormalization()(conv_2_1)
    conv_2_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_2_1)
    res_b2 = merge([res_b1, conv_2_2], mode='sum')
    res_b2 = LeakyReLU(alpha=0.3)(res_b2)
    res_b2 = BatchNormalization()(res_b2)

    # Resnet block3
    conv_3_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b2)
    conv_3_1 = LeakyReLU(alpha=0.3)(conv_3_1)
    conv_3_1 = BatchNormalization()(conv_3_1)
    conv_3_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_3_1)
    res_b3 = merge([res_b2, conv_3_2], mode='sum')
    res_b3 = LeakyReLU(alpha=0.3)(res_b3)
    res_b3 = BatchNormalization()(res_b3)

    # Resnet block4
    conv_4_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b3)
    conv_4_1 = LeakyReLU(alpha=0.3)(conv_4_1)
    conv_4_1 = BatchNormalization()(conv_4_1)
    conv_4_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_4_1)
    res_b4 = merge([res_b3, conv_4_2], mode='sum')
    res_b4 = LeakyReLU(alpha=0.3)(res_b4)
    res_b4 = BatchNormalization()(res_b4)
    
    # Resnet block5
    conv_5_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b4)
    conv_5_1 = LeakyReLU(alpha=0.3)(conv_5_1)
    conv_5_1 = BatchNormalization()(conv_5_1)
    conv_5_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_5_1)
    res_b5 = merge([res_b4, conv_5_2], mode='sum')
    res_b5 = LeakyReLU(alpha=0.3)(res_b5)
    res_b5 = BatchNormalization()(res_b5)
    
    # Resnet block6
    conv_6_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b5)
    conv_6_1 = LeakyReLU(alpha=0.3)(conv_6_1)
    conv_6_1 = BatchNormalization()(conv_6_1)
    conv_6_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_6_1)
    res_b6 = merge([res_b5, conv_6_2], mode='sum')
    res_b6 = LeakyReLU(alpha=0.3)(res_b6)
    res_b6 = BatchNormalization()(res_b6)

    # Resnet block7
    conv_7_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b6)
    conv_7_1 = LeakyReLU(alpha=0.3)(conv_7_1)
    conv_7_1 = BatchNormalization()(conv_7_1)
    conv_7_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_7_1)
    res_b7 = merge([res_b6, conv_7_2], mode='sum')
    res_b7 = LeakyReLU(alpha=0.3)(res_b7)
    res_b7 = BatchNormalization()(res_b7)

    # Resnet block8
    conv_8_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b7)
    conv_8_1 = LeakyReLU(alpha=0.3)(conv_8_1)
    conv_8_1 = BatchNormalization()(conv_8_1)
    conv_8_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_8_1)
    res_b8 = merge([res_b7, conv_8_2], mode='sum')
    res_b8 = LeakyReLU(alpha=0.3)(res_b8)
    res_b8 = BatchNormalization()(res_b8)

    # Resnet block9
    conv_9_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b8)
    conv_9_1 = LeakyReLU(alpha=0.3)(conv_9_1)
    conv_9_1 = BatchNormalization()(conv_9_1)
    conv_9_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_9_1)
    res_b9 = merge([res_b8, conv_9_2], mode='sum')
    res_b9 = LeakyReLU(alpha=0.3)(res_b9)
    res_b9 = BatchNormalization()(res_b9)

    # Resnet block10
    conv_10_1 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(res_b9)
    conv_10_1 = LeakyReLU(alpha=0.3)(conv_10_1)
    conv_10_1 = BatchNormalization()(conv_10_1)
    conv_10_2 = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay))(conv_10_1)
    res_b10 = merge([res_b9, conv_10_2], mode='sum')
    res_b10 = LeakyReLU(alpha=0.3)(res_b10)
    res_b10 = BatchNormalization()(res_b10)

    # Output
    outputs = Conv2D(3, (1, 1), padding='same', activation='sigmoid', kernel_regularizer=l2(weight_decay))(res_b4)
    outputs = Lambda(lambda x: x * 255)(outputs)
    DN = Model(outputs=outputs, inputs=inputs) 

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
    GN.save_weights(('/home/zhaojian/Keras/GAN_APPLE/models/Generator/BE_GAN/'+'lr_'+str(lr)+'epoch_'+str(e)+'_GN'), True)
    DN.save_weights(('/home/zhaojian/Keras/GAN_APPLE/models/Discriminator/BE_GAN/'+'lr_'+str(lr)+'epoch_'+str(e)+'_DN'), True)
    # Save the first refined image as internal result every epoch
    refined = GN.predict(test.reshape(1,img_w,img_h,channels)).reshape(img_w,img_h,channels)[:,:,np.array([2,1,0])]
    refined = Image.fromarray(refined.astype(np.uint8))
    imsave(('/home/zhaojian/Keras/GAN_APPLE/internal_results/BE_GAN/'+'lr_'+str(lr)+'epoch_'+str(e)+'_'+'36216_00510.png'), refined)
