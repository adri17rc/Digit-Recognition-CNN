# Digit-Recognition-CNN
Digit recognition by using a convolutional neural network 

A convolutional neural network is proposed to identify handwritten numbers. It is based on the well-known MNIST dataset, which contains 60000 handwritten numbers (train set) in pictures of size 28x28 and another 10000 to test the model. In the attached picture 9 numbers are shown, as an example. 

![MNIST_Pic](https://user-images.githubusercontent.com/96789733/152503439-91d18088-4921-4cf2-aee7-90c9a20be453.png)

-DATA PROCESSING

Raw data was imported from keras datasets, to avoid previous download of heavy files. It is split into train and test sets, and then resized to a 28x28 pixels image in greyscale (index 1, where 3 stands for RGB images). After resizing them, they are scaled to values between 0 and 1. The labels of this images, that is, the numbers, are translated to categorical values, to be able to properly train the model.

-MODEL 

A sequential convolutional neural network was implemented. It consisted in a 5 hidden layers network; data is processed by a Conv2D layer, with a 3x3 kernel, and afterwards a MaxPoolingLayer is applied. BatchNormalization was also added to speed up the running of the simulation, as it takes into account data previously analized. Then two more Conv2D layers were applied, ending with a fully connected Dense layer of 16 nodes, and a final fully connected layer of 10 nodes, one for each number from 0 to 9. Eventually, 45768 parameters were trained. The output, thanks to the softmax activation function, returns the proababilites of each number. It was compiled with the optimizer 'adam', the loss function 'categorical_crossentropy' and metric 'accuracy'.

-RESULTS 
The previous model was trained during 10 epochs with a batch-size of 32, yielding a minimum loss of 0.0075 and val_loss of 0.0346, and an accuracy of 0.9977. Taking the x variable (pictures) test set, predictions were made after the model was completely trained, and then compared to the real values. The NN was able to predict the real values with a 98.92% accuracy, resulting in a fail rate of 1.08%. It is a very slow value for such a simple model, so we can conclude it succeded on its task.

![Model_Loss_MNIST](https://user-images.githubusercontent.com/96789733/152527754-712c1aab-d4ea-45cb-bec8-04f676178c27.png)![Model_Accuracy_MNIST](https://user-images.githubusercontent.com/96789733/152527759-182a2542-7347-46fa-a41d-ac495548db95.png)

The failed numbers might have a very ilegible writing, confusing the model in the prediction. 
