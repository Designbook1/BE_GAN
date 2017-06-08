# -*- coding: utf-8 -*-
"""MS-LearnNet for MS-Celeb-1M low-shot novel set challenge.
"""
from __future__ import print_function
from __future__ import absolute_import

from collections import defaultdict
from keras.utils.generic_utils import Progbar
import cPickle as pickle
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense, Reshape
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers.merge import Concatenate
from keras.layers import Lambda
from keras.models import Sequential, Model
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.regularizers import l2
import tensorflow as tf
import keras
from keras.utils import np_utils
from keras.models import model_from_json
from keras.utils import generic_utils
from keras.optimizers import SGD
#import setGPU
import time
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Inititialize parameters
batch_size = 16
img_w = 224
img_h = 224
img_c = 3
nb_epoch = 50
weight_decay = 1e-4
lr = 0.01
# Read training data
tr_Pair1_file = open("/media/zhaojian/6TB/data/MS-Celeb-1M/Challenge2/tr_pair1.txt", "r")
tr_Pair1_lines = tr_Pair1_file.readlines()
tr_Pair1_file.close()
tr_N_Pair = len(tr_Pair1_lines)
tr_Pair1_file_list = []
tr_Pair_label = np.zeros([tr_N_Pair, 1], dtype=np.int)
for i in range(tr_N_Pair):
    tr_Pair1_file_list.append(tr_Pair1_lines[i].split()[0])
    tr_Pair_label[i] = int(tr_Pair1_lines[i].split()[1])

tr_Pair2_file = open("/media/zhaojian/6TB/data/MS-Celeb-1M/Challenge2/tr_pair2.txt", "r")
tr_Pair2_lines = tr_Pair2_file.readlines()
tr_Pair2_file.close()
tr_Pair2_file_list = []
for i in range(tr_N_Pair):
    tr_Pair2_file_list.append(tr_Pair2_lines[i].split()[0])

# Read testing data
te_Pair1_file = open("/media/zhaojian/6TB/data/MS-Celeb-1M/Challenge2/te_pair1.txt", "r")
te_Pair1_lines = te_Pair1_file.readlines()
te_Pair1_file.close()
te_N_Pair = len(te_Pair1_lines)
te_Pair1_file_list = []
te_Pair_label = np.zeros([te_N_Pair, 1], dtype=np.int)
te_Pair1 = np.zeros((te_N_Pair,img_w,img_h,img_c), dtype=np.float32)
for i in range(te_N_Pair):
    te_Pair1_file_list.append(te_Pair1_lines[i].split()[0])
    te_Pair_label[i] = int(te_Pair1_lines[i].split()[1])
for j in range(te_N_Pair):
    x = image.load_img(te_Pair1_file_list[i], target_size=(224,224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    te_Pair1[j, ...] = x

te_Pair2_file = open("/media/zhaojian/6TB/data/MS-Celeb-1M/Challenge2/te_pair2.txt", "r")
te_Pair2_lines = te_Pair2_file.readlines()
te_Pair2_file.close()
te_Pair2_file_list = []
te_Pair2 = np.zeros((te_N_Pair,img_w,img_h,img_c), dtype=np.float32)
for i in range(te_N_Pair):
    te_Pair2_file_list.append(te_Pair2_lines[i].split()[0])
for j in range(te_N_Pair):
    x = image.load_img(te_Pair2_file_list[i], target_size=(224,224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    te_Pair2[j, ...] = x

nb_batch = int(tr_N_Pair / batch_size)

# Define resnet50 as backbone feature extractor
def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), kernel_regularizer=l2(weight_decay), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', kernel_regularizer=l2(weight_decay), name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), kernel_regularizer=l2(weight_decay), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, kernel_regularizer=l2(weight_decay),
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', kernel_regularizer=l2(weight_decay),
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), kernel_regularizer=l2(weight_decay), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_regularizer=l2(weight_decay),
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50():

    img_input = Input(shape=(224,224,3))

    bn_axis = 3

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), kernel_regularizer=l2(weight_decay), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    
    x = Conv2D(2048, (3, 3), activation = 'relu', padding = 'same', kernel_regularizer=l2(weight_decay))(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    # Create model.
    model = Model(inputs=img_input, outputs=x)

    return model

# Define our network architecture
# Feature extractor for two branch
FeaExtractor = ResNet50()
# 1st branch input
Net1_input = Input(batch_shape=(batch_size,img_w,img_h,img_c))
# 1st branch extracted features (batch_size,7,7,2048)
Net1_out = FeaExtractor(Net1_input)
# 2nd branch input
Net2_input = Input(batch_shape=(batch_size,img_w,img_h,img_c))
# 2nd branch extracted features (batch_size,7,7,2048)
Net2_out = FeaExtractor(Net2_input)
# Concat features from two branches
fea = Concatenate(axis=1)([Net1_out, Net2_out])
# Final classifier, genuine pair with label 1, imposter pair with label 0
fea = Dense(4096, activation = 'relu')(fea)
fea = Dense(4096, activation = 'relu')(fea)
mod_out = Dense(1, activation = 'sigmoid')(fea)
# Build model
model = Model(inputs = [Net1_input, Net2_input], outputs = mod_out)
# Model summary
model.summary()

# Define optimizer
optim = SGD(lr=lr, momentum=0.9)
# Compile model
model.compile(loss='binary_crossentropy', optimizer=optim)

# Archiv model architecture to json
model_json = model.to_json()
with open("/media/zhaojian/6TB/project/ms-celeb-1m/c2/models/ms-siamese.json", "w") as json_file:
    json_file.write(model_json)

# Start training
for e in range(nb_epoch):
    
    progbar = generic_utils.Progbar(nb_batch*batch_size)
    start = time.time()
   
    for b in range(nb_batch):
        
        # Get a batch of training pairs
        tr_X1_batch = np.zeros((batch_size,img_w,img_h,img_c), dtype=np.float32)
        tr_X1_file = tr_Pair1_file_list[b * batch_size:(b + 1) * batch_size]
        tr_label_batch = tr_Pair_label[b * batch_size:(b + 1) * batch_size]
	for i in range(batch_size):
            x = image.load_img(tr_X1_file[i], target_size=(224,224))
            x = image.img_to_array(x)
	    x = np.expand_dims(x, axis=0)
	    x = preprocess_input(x)
            tr_X1_batch[i, ...] = x
	tr_X2_batch = np.zeros((batch_size,img_w,img_h,img_c), dtype=np.float32)
        tr_X2_file = tr_Pair2_file_list[b * batch_size:(b + 1) * batch_size]
        for j in range(batch_size):
            x = image.load_img(tr_X2_file[i], target_size=(224,224))
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            tr_X2_batch[i, ...] = x
	
	tr_X_batch = [tr_X1_batch,tr_X2_batch]
	tr_Y_batch = tr_label_batch

	train_loss=model.train_on_batch(tr_X_batch, tr_Y_batch)
        # Report results
        progbar.add(batch_size, values=[("train_loss", train_loss)])
            
    # Evaluate the testing loss here
    print('\nTesting for epoch {}:'.format(e + 1))
    test_loss = model.evaluate([te_Pair1,te_Pair2],te_Pair_label,batch_size=batch_size,verbose=False)
    print('\nTesting loss {}:'.format(test_loss))    
  
    print('\nEpoch {}/{}, Time: {}'.format(e, nb_epoch, time.time() - start))

    # Save model weights every epoch
    model.save_weights('/media/zhaojian/6TB/project/ms-celeb-1m/c2/models/ms-siamese_model_epoch_{0:03d}.hdf5'.format(e), True)
