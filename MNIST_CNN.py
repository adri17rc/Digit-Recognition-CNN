#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:55:49 2022

@author: adrianromero
"""

#MDigit recognition. MNIST data set. CNN

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.layers import Conv2D,MaxPooling2D,Flatten, Dense, BatchNormalization

(trainX, trainY), (testX, testy) = mnist.load_data()

#Train set consits in 60000 images, whereas test set has 10000. 

#Plot the first 9 images (numbers)

for i in range(9):
    plt.subplot(330 + 1 + i) #330, shape of the grid
    plt.imshow(trainX[i],cmap=plt.get_cmap('binary'))
    
plt.show()

#Reshape dataset to have a single channel. We know images' size is 28x28 
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

#We shall translate the numbers in train_y to binary.
trainY= keras.utils.to_categorical(trainY)
testy2=keras.utils.to_categorical(testy)

#Pixel values now range from 0 to 255; they should be renormalized

train_norm=trainX.astype('float32')
test_norm=testX.astype('float32')
train_norm=train_norm/255.0
test_norm=test_norm/255.0


#Model. Kernel of size 3x3, non-stride, MaxPoolinglayer. No padding

model= keras.Sequential()
#Other kernel_initializer: he_normal, glorotuniform,ones,etc.
model.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform'))
model.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax')) #Output layer must return probabilities
#Output layer: 10 possible outputs

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

results=model.fit(trainX,trainY,epochs=10,batch_size=32,validation_data=(testX,testy2))

#Plots for loss and acuracy 

#Cross entropy loss 
plt.plot(results.history['loss'], label='train')
plt.plot(results.history['val_loss'], label='test')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Model Loss')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

#Accuracy
plt.plot(results.history['accuracy'], label='train')
plt.plot(results.history['val_accuracy'], label='test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model accuracy')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

#Predict y values (numbers) from x_test. Compare them with y_test. 

prediction=model.predict(testX)  #Prediction in categorical. 
counter=0
fail=0
for i in range(len(testX)):
    if np.argmax(prediction[i])==testy[i]:
        counter+=1
    else:
        fail+=1
 
print('Success %=',counter/100)
print('Fail %=',fail/100)



