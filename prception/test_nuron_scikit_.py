import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler # класс для нормлизации данных
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split# случайное разделение на тестовую и трениров. выборки
import My_class_percept_grad as mcp
import matplotlib.pyplot as plt




iris=datasets.load_iris()

X=iris.data[:, [2,3]]
y=iris.target

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=0)
														#test_size:	   random_state: 
									   #размер тестовой выборки=30%	   случайные числа генерируются np.random

#_________Масштобирование_данных____________________									   
sc = StandardScaler() #создание объекта
X_train_std = sc.fit_transform(X_train)# Нахождение средней, стандартного отклонения  и нормализации одновременно
X_test_std = sc.transform(X_test)# просто нахождение средней и стандартного отклонения
#___________________________________________________

#_______Обучение_персептрона________________________
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

pr=ppn.predict(X_test_std)
answer_assessment=accuracy_score(y_test,pr)#оценка качества предсказания
print('\nanswer_assessment=',answer_assessment)
#___________________________________________________

#_______Отрисовка_и_выввод_данных___________________
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined=np.hstack((y_train, y_test))
mcp.plot_decision_regions(X=X_combined_std, y=y_combined, classifer=ppn, test_idx=range(105-150))
plt.xlabel('длина лепестка [стандартизованная]')
plt.ylabel('ширина лепестка [стандартизованная]')
plt.legend(loc='upper left')
plt.show ()

#___________________________________________________
