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
