# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 12:21:16 2025

@author: Vimalathithan Kannan
"""

# Importing Necessary Models and Classes

from numpy import zeros, ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from tensorflow.keras import Input
import tensorflow.keras.backend as K
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation 
from keras.layers import Concatenate, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import cv2
from natsort import natsorted

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

# Encoder Block
def encoder(layer_in, n_filters, batchnorm=True):
    init = RandomNormal(stddev=0.02) # Random Weight Initialization
    g = Conv2D(n_filters, (3,3), strides=(2,2), padding='same',
               kernel_initializer=init)(layer_in) # Conv2D Downsampling Layer
    if batchnorm:
        # Adding Batch Normalization Layer if batchnorm=True
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g) # Activation Layer is Unconditional
    return g # This function defines the Encoder Block

# Decoder Block
def decoder(layer_in, skip_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02) # Random Weight Initialization
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
    init = RandomNormal(stddev=0.02) # Random Weight Initialization
    in_image = Input(shape=image_shape)
    
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
    
    # Upsampling Part with Decoders
    
    d1 = decoder(b, e4, 256)
    d2 = decoder(d1, e3, 128)
    d3 = decoder(d2, e2, 64, dropout=False)
    d4 = decoder(d3, e1, 32, dropout=False)
    # Decoder Blocks with Input Layer, Skip Connections and Number of Filters
    
    g = Conv2DTranspose(image_shape[2], (3,3), strides=(2,2), padding='same',
                        kernel_initializer=init)(d4)
    out_image = Activation('tanh')(g)
    model = Model(in_image, out_image)
    # Model(a, b) compiles all the layers from layer 'a' to 'b' and generates
    # a model
    return model

# ------- Discriminator - pix2pix Network-------

def discriminator(image_shape):
    init = RandomNormal(mean=0, stddev=0.02, seed=4)
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
    
    d = Conv2D(256, (3,3), strides=(2,2), padding='same', 
               kernel_initializer=init)(d)
    d = BatchNormalization()(d) # Batch Normalisation is Recommended
    d = LeakyReLU(alpha=0.2)(d)
    
    # PatchGAN (Local Discriminator) of the pix2pix Network
    d = Conv2D(1, (3,3), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    
    model = Model([source_image, target_image], patch_out)
    
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
    gen_out = g_model(input_source) # Using the Source Image as input to 
    # Generator
    # gen_out is the Output Image generator by the U-Net
    disc_out = d_model([input_source, gen_out]) # Using the Source Image and 
    # Generator Output to Discriminator
    model = Model(input_source, [disc_out, gen_out])
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
    filename = 'model_%0.6d.h5' % (step+1)
    g_model.save(filename)
    print('>>> Saved: %s' % filename)

# Functions to Generate Training Samples 

# Generating Real Samples

def generate_real_samples(dataset, n_samples, patch_shape):
    trainA, trainB = dataset
    idx = randint(0, trainA.shape[0], n_samples)
    X1, X2 = trainA[idx], trainB[idx]
    y = ones((n_samples, patch_shape, patch_shape, 1))
    # In this Function, the Real Image Samples are generated directly by 
    # taking them from the dataset
    return [X1, X2], y

# Generating Fake Samples

def generate_fake_samples(g_model, samples, patch_shape):
    X = g_model.predict(samples)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    # In this Function, the Fake Image Samples are generated by feeding the 
    # samples to the Generator Model
    return X, y

# Train the Model

def train(d_model, g_model, gan_model, train_dataset, val_dataset, n_epochs, n_batch):
    n_patch = d_model.output_shape[1]
    trainA, trainB = train_dataset
    valA, valB = val_dataset
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
        [X_realA, X_realB], y_real = generate_real_samples(train_dataset, 
                                                           n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        
        # Updating Discriminator Model Weights
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        
        # Updating Generator Model Weights
        g_loss, gloss1, gloss2 = gan_model.train_on_batch(
            X_realA, [y_real, X_realB])
        
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
          
          [X_realA_val, X_realB_val], y_real_val = generate_real_samples(
              val_dataset, n_samples_val, n_patch)
          X_fakeB_val, y_fake_val = generate_fake_samples(
              g_model, X_realA_val, n_patch)
          
          # Evaluation Discriminator Loss for Validation Dataset
          
          d_loss1_val = d_model.evaluate([X_realA_val, X_realB_val], 
                                         y_real_val, verbose=0)
          d_loss2_val = d_model.evaluate([X_realA_val, X_fakeB_val], 
                                         y_fake_val, verbose=0)
          
          d_loss_val = d_loss1_val + d_loss2_val
          
          # Evaluation GAN Loss for Validation Dataset
          
          g_loss_val, _, _ = gan_model.evaluate(
              X_realA_val, [y_real_val, X_realB_val], verbose=0)
                    
          d_val.append(d_loss_val)
          g_val.append(g_loss_val)
        
    return d1, d2, g, g_l1, g_l2, d_val, g_val

#------- Main Program -------

input_data = create_dataset('')
output_data = create_dataset('')

X_train, X_val, y_train, y_val = train_test_split(
    input_data, output_data, test_size=0.3, random_state=42)

train_dataset = [X_train, y_train]
val_dataset = [X_val, y_val]

start = datetime.now()

image_shape = (32, 32, 1)

d_model = discriminator(image_shape)
g_model = generator(image_shape)

gan_model = gan(g_model, d_model, image_shape)

d1, d2, g, g_l1, g_l2, d_val, g_val = train(
    d_model, g_model, gan_model, train_dataset, val_dataset, 
    n_epochs=50, n_batch=20)

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

#------- Save Predicted Files -------

predictionModel = g_model

def extract_data(data):
    split_arrays = []
    block_size = 32
    for i in range(0, 128, block_size):
        for j in range(0, 128, block_size):
            block = data[i:i+block_size, j:j+block_size]
            split_arrays.append(block)
    return split_arrays

def reconstruct_data(data):
    block = []
    for i in range(4):
        row_block = []
        for j in range(4):
            row_block.append(data[i*4+j])
        block.append(row_block)
    recons = np.block(block)
    return recons

def predict_data(data_in):
    true_prediction = []
    for i in range(len(data_in)):
        data = data_in[i].reshape(1, 32, 32, 1)
        temp_red = predictionModel.predict(data)
        true_prediction.append(temp_red)
    true_prediction = np.stack(true_prediction)
    true_prediction = true_prediction.reshape(16, 32, 32)
    Recons_Pred = reconstruct_data(true_prediction)
    return Recons_Pred

directory = r''
filepath = glob.glob(os.path.join(directory, '*.dat'))
filepath = natsorted(filepath)

prediction_directory = r''

start_index = 1001
for idx, file in enumerate(filepath, start=start_index):
    data = np.loadtxt(file)
    data_in = extract_data(data)
    data_out = predict_data(data_in)
    
    
    filename = f"s{idx:04d}_p1_pred.dat"
    savepath = os.path.join(prediction_directory, filename)
    np.savetxt(savepath, data_out, fmt='%d')
    
#------- Evaluate Model -------

def collect_data(directory):
    filepath = glob.glob(os.path.join(directory, '*.dat'))
    data_list = [np.loadtxt(file) for file in filepath]
    data = np.stack(data_list)
    return data

Out_Data = collect_data('')
Pred_Data = collect_data('')

Ham_Dist = []
for j in range(len(Pred_Data)):
    dist = np.sum((Out_Data[j] != Pred_Data[j]))
    Ham_Dist.append(dist)
    
Ham_Dist = np.stack(Ham_Dist)/16384*100

Ham_Dist = pd.DataFrame(Ham_Dist)
Ham_Dist.to_csv('Prediction Results.csv')
