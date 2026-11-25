# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 11:32:01 2025

@author: Vimal
"""

from numpy import zeros, ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from tensorflow.keras import Input
import tensorflow.keras.backend as K
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation 
from keras.layers import Concatenate, Dropout, BatchNormalization, Dense, Lambda
from sklearn.model_selection import train_test_split
import numpy as np
import os
import glob
from datetime import datetime
from zipfile import ZipFile
import pandas as pd
from pandas import ExcelWriter
from natsort import natsorted
from time import time

#------- Functions for Collecting Data -------
def create_dataset(directory):
    filepath = glob.glob(os.path.join(directory, '*.dat'))
    data_list = [np.loadtxt(file)[:, :, np.newaxis] for file in filepath]
    data = np.stack(data_list)
    
    split_data = list()
    for k in range(len(data)):
        split_arrays = []
        block_size = 32
        for i in range(0, 128, block_size):
            for j in range(0, 128, block_size):
                block = data[k][i:i+block_size, j:j+block_size]
                split_arrays.append(block)
        split_data.extend(split_arrays)
    return np.stack(split_data)

'''
Slicing and Stiching Strategy is include in the previous function.
Block size must be adjusted depending on the data size.
'''

#----- Generator - U-Net Configuration containing Encoder and Deconder -----

# Function to create a conditioned pressure layer
def broadcast_p_for_disc(x):
    x = K.expand_dims(x, 1)
    x = K.expand_dims(x, 1)
    multiples = [1, 4, 4, 1]
    x = K.tile(x, multiples)
    return x

def broadcast_p_for_gen(x):
    x = K.expand_dims(x, 1)
    x = K.expand_dims(x, 1)
    return x

# Encoder Block
def encoder(layer_in, n_filters, batchnorm=True):
    init = RandomNormal(stddev=0.02, seed=42) # Random Weight Initialization
    g = Conv2D(n_filters, (3,3), strides=(2,2), padding='same',
               kernel_initializer=init)(layer_in) # Conv2D Downsampling Layer
    if batchnorm:
        # Adding Batch Normalization Layer if batchnorm=True
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g) # Activation Layer is Unconditional
    return g # This function defines the Encoder Block

# Decoder Block
def decoder(layer_in, skip_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02, seed=22) # Random Weight Initialization
    g = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same',
                        kernel_initializer=init)(layer_in)
                        # Conv2DTranspose Upsampling Layer
    if dropout:
        # Adding Dropout Layer if dropout=True
        g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, skip_in]) # Skipping Layers for Upsampling
    g = Activation('relu')(g)
    return g # This function defines the Decoder Block

# Generator Network

def generator(image_shape=(32,32,1)):
    init = RandomNormal(stddev=0.02, seed=0) # Random Weight Initialization
    in_image = Input(shape=image_shape)
    in_p = Input(shape=(1,), name='Pressure_Scalar_Input')
    
    # Configuration of U-Net
    # Downsampling Part using Encoders
    
    e1 = encoder(in_image, 32, batchnorm=False)
    e2 = encoder(e1, 64)
    e3 = encoder(e2, 128)
    e4 = encoder(e3, 256) 
    # Encoder Blocks with Input Layer connection and Number of Filters
    
    # Bottleneck Layer without Batch Normalization and ReLu Activation
    b = Conv2D(256, (3,3), strides=(2,2), padding='same',
               kernel_initializer=init)(e4)
    b = Activation('relu')(b)
    
    # Concatenating Pressure to the Bottleneck Layer
    p_features = Dense(256, kernel_initializer=init)(in_p)
    p_map = Lambda(broadcast_p_for_gen)(p_features)
    
    b_concatenated = Concatenate()([b, p_map])
    
    # Upsampling Part with Decoders
    
    d1 = decoder(b_concatenated, e4, 256)
    d2 = decoder(d1, e3, 128)
    d3 = decoder(d2, e2, 64, dropout=False)
    d4 = decoder(d3, e1, 32, dropout=False)
    # Decoder Blocks with Input Layer, Skip Connections and Number of Filters
    
    g = Conv2DTranspose(image_shape[2], (3,3), strides=(2,2), padding='same',
                        kernel_initializer=init)(d4)
    out_image = Activation('tanh')(g)
    model = Model([in_image, in_p], out_image)
    # Model(a, b) compiles all the layers from layer 'a' to 'b' and generates
    # a model
    return model

# ------- Discriminator - pix2pix Network-------

def discriminator(image_shape):
    init = RandomNormal(mean=0, stddev=0.02, seed=4)
    
    # Input Pressure Scalar but Concatenating only at the 3rd Convolutional Output
    in_p = Input(shape=(1,), name='Pressure_Scalar_Input')
    
    # Pairing the Source and Target Image in the Discriminator Input
    source_image = Input(shape=image_shape)
    target_image = Input(shape=image_shape)
    paired = Concatenate()([source_image, target_image])
    
    # Configuration of the Network
    d = Conv2D(32, (3,3), strides=(2,2), padding='same', 
               kernel_initializer=init)(paired)
    d = LeakyReLU(alpha=0.2)(d) # Activation Layer 
    
    d = Conv2D(64, (3,3), strides=(2,2), padding='same', 
               kernel_initializer=init)(d)
    d = BatchNormalization()(d) # Batch Normalisation is Recommended
    d = LeakyReLU(alpha=0.2)(d) 
    
    d = Conv2D(128, (3,3), strides=(2,2), padding='same', 
               kernel_initializer=init)(d)
    d = BatchNormalization()(d) # Batch Normalisation is Recommended
    d = LeakyReLU(alpha=0.2)(d)
    
    # Concatenating Pressure into the Convolutional Output
    p_features = Dense(128, kernel_initializer=init)(in_p)
    p_map = Lambda(broadcast_p_for_disc)(p_features)
    d_conditioned = Concatenate()([d, p_map])
    
    # Continuing with the Convolution
    d_conditioned = Conv2D(256, (3,3), strides=(2,2), padding='same', 
               kernel_initializer=init)(d_conditioned)
    d_conditioned = BatchNormalization()(d_conditioned) # Batch Normalisation is Recommended
    d_conditioned = LeakyReLU(alpha=0.2)(d_conditioned)
    
    # PatchGAN (Local Discriminator) of the pix2pix Network
    d_conditioned = Conv2D(1, (3,3), padding='same', kernel_initializer=init)(d_conditioned)
    patch_out = Activation('sigmoid')(d_conditioned)
    
    model = Model([source_image, target_image, in_p], patch_out)
    
    # Defining Training Options
    opt = Adam(learning_rate=0.0002, beta_1=0.99)
    model.compile(loss='binary_crossentropy', optimizer=opt, 
                  loss_weights=[0.5])
    return model

#------- GAN Model and its related Functions -------

# Loss Functions

# Here a L2 Loss Function is defined. Based on the preferences, we can 
# include or exclude other Loss Functions. The defined Loss Functions must
# be added to the 'model.compile' function in the GAN Model

def l2_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))
# K is the Keras backend Operator 

# GAN Architecture

def gan(g_model, d_model, image_shape):
    # The following command makes the Weights in the Discriminator Untrainable
    # only during the GAN (not inside the Discriminator itself)
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    
    input_source = Input(shape=image_shape) # Defining Source Image
    input_p = Input(shape=(1,), name='GAN_Pressure_Input')
    gen_out = g_model([input_source, input_p]) # Using the Source Image as input to 
    # Generator
    # gen_out is the Output Image generator by the U-Net
    disc_out = d_model([input_source, gen_out, input_p]) # Using the Source Image and 
    # Generator Output to Discriminator
    model = Model([input_source, input_p], [disc_out, gen_out])
    # Creating the Model with the layers starting from input_source
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae', l2_loss],
                  optimizer=opt, loss_weights=[1, 100, 10])
    # Compiling Model with three Loss Functions with different Weights
    return model

#------- Functions related to Training -------

# Function to Summarize the Performance of GAN
# This function saves the GAN model periodically

def save_model(step, g_model):
    filename = 'model_%0.8d.h5' % (step+1)
    g_model.save(filename)
    print('>>> Saved: %s' % filename)

# Functions to Generate Training Samples 
# Generating Real Samples

def generate_real_samples(dataset, n_samples, patch_shape):
    trainA, trainB, p_train = dataset
    idx = randint(0, trainA.shape[0], n_samples)
    X1, X2, X3 = trainA[idx], trainB[idx], p_train[idx]
    y = ones((n_samples, patch_shape, patch_shape, 1))
    # In this Function, the Real Image Samples are generated directly by 
    # taking them from the dataset
    return [X1, X2, X3], y

# Generating Fake Samples

def generate_fake_samples(g_model, samples, p_batch, patch_shape):
    X = g_model.predict([samples, p_batch])
    y = zeros((len(X), patch_shape, patch_shape, 1))
    # In this Function, the Fake Image Samples are generated by feeding the 
    # samples to the Generator Model
    return X, y

# Train the Model

def train(d_model, g_model, gan_model, train_dataset, val_dataset, n_epochs, n_batch):
    n_patch = d_model.output_shape[1]
    trainA, trainB, p_train = train_dataset
    valA, valB, p_val = val_dataset
    n_samples_val = len(valA)
    batch_per_epoch = int(len(trainA)/n_batch)
    n_steps = batch_per_epoch * n_epochs
    # Number of Steps depends on the Batch Size, Epoch, and the Total Number
    # of Samples in the Dataset
    
    # Creating Lists to Store Performance Metrics
    d1, d2, g = list(), list(), list()
    g, g_l1, g_l2 = list(), list(), list()
    d_val, g_val = list(), list()
    
    for i in range(n_steps):
        # Generating Real and Fake Image Samples
        [X_realA, X_realB, p_batch], y_real = generate_real_samples(train_dataset, 
                                                           n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, p_batch, n_patch)
        
        # Updating Discriminator Model Weights
        d_loss1 = d_model.train_on_batch([X_realA, X_realB, p_batch], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB, p_batch], y_fake)
        
        # Updating Generator Model Weights
        g_loss, gloss1, gloss2 = gan_model.train_on_batch(
            [X_realA, p_batch], [y_real, X_realB])
        
        d1.append(d_loss1)
        d2.append(d_loss2)
        g.append(g_loss)
        g_l1.append(gloss1)
        g_l2.append(gloss2)
        
        # Printing the Progress of Training
        print('>>> %d, d1[%.3f] d2[%.3f] g[%.3f] g_l1[%.3f] g_l2[%.3f]' 
              % (i+1, d_loss1, d_loss2, g_loss, gloss1, gloss2))
        
        # Summarising Performance Periodically to Save the GAN Model
        if (i+1) % (batch_per_epoch*10) == 0:
            save_model(i, g_model)
        
        if (i+1) % batch_per_epoch == 0:
        
            # Passing the Validation Dataset and using the same Functions as in 
            # the Training Process
            
            [X_realA_val, X_realB_val, p_val], y_real_val = generate_real_samples(
                val_dataset, n_samples_val, n_patch)
            X_fakeB_val, y_fake_val = generate_fake_samples(
                g_model, X_realA_val, p_val, n_patch)
            
            # Evaluation Discriminator Loss for Validation Dataset
            
            d_loss1_val = d_model.evaluate([X_realA_val, X_realB_val, p_val], 
                                           y_real_val, verbose=0)
            d_loss2_val = d_model.evaluate([X_realA_val, X_fakeB_val, p_val], 
                                           y_fake_val, verbose=0)
            
            d_loss_val = d_loss1_val + d_loss2_val
            
            # Evaluation GAN Loss for Validation Dataset
            
            g_loss_val, _, _ = gan_model.evaluate(
                [X_realA_val, p_val], [y_real_val, X_realB_val], verbose=0)
                      
            d_val.append(d_loss_val)
            g_val.append(g_loss_val)
            
    return d1, d2, g, g_l1, g_l2, d_val, g_val
    
#------- Main Program -------

p_values = [0.04, 0.08, 0.12, 0.16]

BASE_DIR = r''

X_list, Y_list, P_list = [], [], []

for idx, p_val in enumerate(p_values, start=1):
    
    in_dir = os.path.join(BASE_DIR, f'pres_{str(idx)}', 'In Files')
    out_dir = os.path.join(BASE_DIR, f'pres_{str(idx)}', 'Out Files')
    
    
    input_batch = create_dataset(in_dir)
    output_batch = create_dataset(out_dir)
    
    
    p_batch = np.ones((input_batch.shape[0], 1)) * p_val
        
    X_list.append(input_batch)
    Y_list.append(output_batch)
    P_list.append(p_batch)

# Concatenate all data into the final arrays
input_data = np.concatenate(X_list, axis=0)
output_data = np.concatenate(Y_list, axis=0)
p_data = np.concatenate(P_list, axis=0)


I, _, O, _, P, *_ = train_test_split(input_data, output_data, p_data, train_size=0.175, random_state=42, stratify=p_data)


# P_values is now shape (16000, 1)

# Splitting the Dataset into Train, Test, and Validation

X_train, X_val, y_train, y_val, p_train, p_val = train_test_split(
    I, O, P, test_size=0.3, random_state=42, stratify=P)

train_dataset = [X_train, y_train, p_train]
val_dataset = [X_val, y_val, p_val]

# Train the GAN

start = datetime.now()

import tensorflow as tf

gpu = tf.config.list_physical_devices('GPU')

if gpu:
    try:
        for GpU in gpu:
            tf.config.experimental.set_memory_growth(GpU, True)
        print(f'Using GPU: {gpu}')
    except RuntimeError as e:
        print(e)

image_shape = (32, 32, 1)

d_model = discriminator(image_shape)
g_model = generator(image_shape)

gan_model = gan(g_model, d_model, image_shape)


d1, d2, g, g_l1, g_l2, d_val, g_val = train(
    d_model, g_model, gan_model, train_dataset, val_dataset, 
    n_epochs=25, n_batch=10)

g_model.save('Final_Model.h5')

stop = datetime.now()

d1Loss, d2Loss, gLoss, dValLoss, gValLoss = map(np.stack, [d1, d2, g, d_val, g_val])

traLosses = {
        'Iterations': range(len(d1Loss)),
        'Discriminator Loss': d1Loss+d2Loss,
        'Generator Loss': gLoss}

trainingLosses = pd.DataFrame(traLosses)
trainingLosses.to_csv('Training Losses.csv', index=False)

valLosses = {
        'Iterations': range(len(dValLoss)),
        'Discriminator Loss': dValLoss,
        'Generator Loss': gValLoss}

validationLosses = pd.DataFrame(valLosses)
validationLosses.to_csv('Validation Losses.csv', index=False)


#------- Predicting the Results -------


def collect_data(directory):
    filepath = glob.glob(os.path.join(directory, '*.dat'))
    filepath = natsorted(filepath)
    dataList = [np.loadtxt(file) for file in filepath]
    data = np.stack(dataList)
    return data

def extract_data(data):
    splitArrays = []
    blockSize = 32
    for i in range(0, 128, blockSize):
        for j in range(0, 128, blockSize):
            block = data[i:i+blockSize, j:j+blockSize]
            splitArrays.append(block)
    return splitArrays

def reconstruct_data(data):
    block = []
    for i in range(4):
        rowBlock = []
        for j in range(4):
            rowBlock.append(data[i*4+j])
        block.append(rowBlock)
    recons = np.block(block)
    return recons

def predict_data(model,dataIn,p):
    truePrediction = []
    for i in range(len(dataIn)):
        data = dataIn[i].reshape(1,32,32,1)
        tempPred = model.predict([data,p])
        truePrediction.append(tempPred)
    truePrediction = np.stack(truePrediction)
    truePrediction = truePrediction.reshape(16,32,32)
    reconsPred = reconstruct_data(truePrediction)
    return reconsPred


pValues = [0.04,0.08,0.12,0.16]
pressure = ['pres_1', 'pres_2', 'pres_3', 'pres_4']

Dir = r'G:\Contact Mechanics\Training w Varying Training Size'

inData = collect_data(os.path.join(Dir, 'Test Data', 'In Files'))


model = g_model

for pres, pVal in zip(pressure, pValues):
    
    predDir = os.path.join(Dir, 'Predicted Results', pres )
    p = np.ones((1,1))*pVal
    for i in range(len(inData)):
        data = extract_data(inData[i])
        start_time = time()
        dataPred = predict_data(model, data, p)
        end_time = time()
        elapsed_time = end_time - start_time
        print(f'Elapsed Time: {elapsed_time:.2f} seconds')
        filename = f"s{1000+i+1:03d}_{pres}_pred.dat"
        np.savetxt(os.path.join(predDir, filename), dataPred, fmt='%d')

