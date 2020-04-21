import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from dateutil import parser

dataset = pd.read_csv('./datasets/appdata10.csv')

dataset.head()
dataset.descibe()

dataset["hour"] = dateset.hour.str.slice(1,3).astype(int)
