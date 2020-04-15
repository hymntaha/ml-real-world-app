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
