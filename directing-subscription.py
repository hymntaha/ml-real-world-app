import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from dateutil import parser

dataset = pd.read_csv('./datasets/appdata10.csv')

dataset.head()
dataset.descibe()

dataset["hour"] = dateset.hour.str.slice(1,3).astype(int)


dataset2 = dataset.copy().drop(columns = ['user','screen_list','enrolled_date','first_open','enrolled'])
dataset2.head()

plt.subtitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset2.shape[1]+1):
    plt.subplots(3,3,i)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i-1])

    vals = np.size(dataset2.iloc[:, i-1].unique())
    plt.hist(dataset2.iloc[:, i-1], bins = vals, color = '#3F5D7D' )


#correlation with response
dataset2.corrwith(dataset.enrolled).plot.bar(figsize = (20,10),title= 'Correlation with Response Variable', fontsize=15, rot=45, grid = True)
#correlation matrix
sn.set(style='white', font_scale=2)
#Compute the correlation matrix
corr = dataset2.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#Set up the matplotlib figure
f,ax = plt.subplots_(figsize=(18,15))
f.suptitle("Correlation Matrix", fontsize = 40)

#Generat ea custom diverging colormap
cmap = sn.diverging_palette(220,10,as_cmap=True)
