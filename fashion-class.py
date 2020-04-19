import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

fashion_train_df = pd.read_csv('./datasets/fashion-mnist_train.csv', sep = ',')
fashion_test_df = pd.read_csv('./datasets/fashion-mnist_test.csv', sep = ',')

print(fashion_train_df.head())

print(fashion_train_df.shape)

training = np.array(fashion_train_df, dtype="float32")
testing = np.array(fashion_test_df, dtype="float32")

i = random.randint(1,60000)
plt.imshow(training[i,1:].reshape(28,28))

label = training[i,0]
print(label)

# Lets view more images in a grid format
# Define the dimensions of the plot grid

W_grid = 15
L_grid = 15

# fig, axes = plt.subplots(L_grid,W_grid)

fig, axes = plt.subplot(L_grid, W_grid, figsize = (17,17))
axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array
n_training = len(training) # get the length of the training dataset

for i in np.arange(0, W_grid * L_grid):
    index = np.random.randint(0,n_training)
    axes[i].imshow(training[index,1:].reshape(28,28))
    axes[i].set_title(training[index,0], fontsize = 8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)


X_train = training[:,1:]/255
y_train = training[:,0]

X_test = testing[:,1:]/255
y_test = testing[:,0]

from sklearn.model_selection import train_test_split
X_train, X_validate,y_train, y_validate = train_test_split(X_train,y_train, test_size=0.2, random_state=12345)

X_train = X_train.reshape(X_train.shape[0] * (28,28,1))
X_test = X_test.reshape(X_test.shape[0]*(28,28,1))
X_validate = X_validate.reshape(X_validate.shape[0] * (28,28,1))

X_train.shape
X_test.shape
X_validate.shape

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

cnn_model - Sequential()
cnn_model.add(Conv2D(32,3,3, input_shape= (28,28,1), activation = 'relu'))

cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(output_dim=32, activation='relu'))
cnn_model.add(Dense(output_dim=10, activation='sigmoid'))
cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
epochs = 50

cnn_model.fit(X_train, y_train, batch_size=512, nb_epoch=epochs, verbose=1, validation_data = (X_validate,y_validate))

evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))
