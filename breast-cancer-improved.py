import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
# print(cancer)

cancer.keys()
print(cancer['DESCR'])
print(cancer['target'])
cancer['data'].shape
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'],['target']))
print(df_cancer.head())
print(df_cancer.tail())

sns.pairplot(df_cancer, vars = ['mean radius','mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
