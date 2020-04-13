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

sns.pairplot(df_cancer,hue = 'target', vars = ['mean radius','mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
sns.countplot(df_cancer['target'])
sns.scatterplot(x='mean area', y='mean smoothness', hue = 'target', data = df_cancer)
plt.figure(figsize= (20,10))
sns.heatmap(df_cancer.corr(), annot=True)
X = df_cancer.drop(['target'], axis = 1)
y = df_cancer['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=5)
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)
