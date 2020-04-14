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
min_train  = X_train.min()
range_train = (X_train-min_train).max()
X_train_scaled = (X_train-min_train)/range_train
sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue=y_train)
sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue=y_train)
min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test
svc_model.fit(X_train_scaled, y_train)


y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test,y_predict)
sns.heatmap(cm, annot=True)
print(classification_report(y_test,y_predict))
plt.show()

#Improving the model
param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001], 'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit= True, verbose=4)
grid.fit(X_train_scaled, y_train)
grid.best_params_
grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test, grid_predictions))
