import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import time

dataset = pd.read_csv('new_appdata10.csv')

# Data Preprocessing

response = dataset["enrolled"]
dataset = dataset.drop(columns = 'enrolled')
