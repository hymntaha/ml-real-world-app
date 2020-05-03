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

#Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": 0.5})

#feature engineering
dataset.dtypes
dataset["first_open"] = [parser.parse(row_data) for row_data in dataset['first_open']]
dataset["enrolled_date"] = [parser.parse(row_data) if isinstance(row_data) else row_data for row_data in dataset['enrolled_date']]

dataset["difference"] = (dateset.enrolled_date - dataset.first_open).astype('timedelta64[h]')

plt.hist(dataset['difference'].dropna(), color='#3F5D7D', range = [0, 100])
plt.title('Distribution of Time-Since-Enrolled')
plt.show

dataset.loc[dataset.difference > 48, 'enrolled'] = 0
dataset = dataset.drop(columns=['difference','enrolled_date', 'first_open'])

# formatiing the screen_list field
top_screens = pd.read_csv('top_screen.csv').top_screens.values

dataset["screen_list"] = dataset.screen_list.astype(str) + ','
for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
    dataset["screen_list"] = dataset.screen_list.str.replace(sc+",", "")

dataset["Other"] = dataset.screen_list.str.count(",")
dataset = dataset.drop(columns=["screen_list"])

#funnels
savings_screens = ["Saving1","Saving2","Saving2Amount","Saving4","Saving5","Saving6","Saving7","Saving8","Saving9","Saving10",]
dataset['SavingsCount'] = dataset[savings_screens].sum(axis = 1)
