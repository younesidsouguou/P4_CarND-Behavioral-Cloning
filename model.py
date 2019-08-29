# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:21:50 2019

@author: YOUNES IDSOUGUOU
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#### HYPERPARAMETERS

READ_BATCH_SIZE = 1024
LR_ADJUST = 0.2

TRAINING_BATCH_SIZE = 64
EPOCHS = 7
LEARNING_RATE = 5e-3

#### DATA PREPROCESSING

df = pd.read_csv('../../data/P4/driving_log.csv')

X = []
y = []

for k in range(df.shape[0]):
    if k % READ_BATCH_SIZE == 0:
        print('READ BATCH SIZE %d / %d'%(1+k//READ_BATCH_SIZE,
                                df.shape[0]//READ_BATCH_SIZE+1))
        
    X.append(plt.imread('../../data/P4/'+df.iloc[k]['center']))
    y.append(df.iloc[k]['steering'])
    
    X.append(plt.imread('../../data/P4/'+df.iloc[k]['left'].strip()))
    y.append(df.iloc[k]['steering']+LR_ADJUST)
    
    X.append(plt.imread('../../data/P4/'+df.iloc[k]['right'].strip()))
    y.append(df.iloc[k]['steering']-LR_ADJUST)
    
X = np.array(X)
y = np.array(y)

#### DATA AUGMENTATION

X = np.vstack((X, np.flip(X, axis=2)))
y = np.hstack((y, -y))

#### MODEL DESIGN

from keras.models import Model
from keras.layers import Lambda, Dense, Flatten, Cropping2D, Conv2D, Input, Dropout
from keras import optimizers

def DriveNet():
    
    input_c = Input(shape=(160,320,3))
    
    x = Lambda(lambda x: x / 255.0 -0.5, input_shape=(160,320,3))(input_c)
    x = Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3))(x)
    
    x = Conv2D(24,(5,5), strides=(2,2), activation="relu")(x)
    x = Conv2D(36,(5,5), strides=(2,2), activation="relu")(x)
    x = Conv2D(48,(5,5), strides=(2,2), activation="relu")(x)
    x = Conv2D(64,(3,3), activation="relu")(x)
    x = Conv2D(64,(3,3), activation="relu")(x)
    
    x  = Flatten()(x)
    x = Dropout(0.5)(x)
    
    fcl = Dense(100)(x)
    fcl = Dropout(0.5)(fcl)
    
    fcl = Dense(50)(fcl)
    fcl = Dense(10)(fcl)
    
    output = Dense(1)(fcl)
    
    model = Model(inputs=input_c, outputs=output)
    model.summary()
    
    return model

model = DriveNet()

#### MODEL TRAINING

adam = optimizers.Adam(lr=LEARNING_RATE)
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, batch_size=TRAINING_BATCH_SIZE, epochs=EPOCHS,
          validation_split=0.2, shuffle=True)

model.save('model.h5')

