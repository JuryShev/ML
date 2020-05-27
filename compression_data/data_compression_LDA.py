"""Линейный дискриминантный анализ (linear discriminant analysis, LDA)

LDA - пытается найти подпространство признаков, которое оптимизирует разделимость классов.

Данный метод состоит из следующих этапов:

1 стандартизировать d-мерный набор данных
2 Для каждого класса вычислить «d-мерный вектор средних.
3 Создать матрицу разброса между классами SB и матрицу разброса внутри классов Sw
4 Вычислить собственные векторы и соответствующие собственные значения матрицы SW(-1) SB.
5 Выбрать k собственных векторов, которые соответствуют k самым большим собственным значениям
  для построения d x k-матрицы преобразования W; собственные векторы являются столбцами этой матрицы.
6 Спроецировать образцы на новое подпространство признаков при помощи матрицы преобразования W.

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df_wine=pd.read_csv(url)


X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
#iloc - используется для доступа по числовому значению (начиная от 0)
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=0)
                                                    #процентное содержание
                                                    #для метки test

#_________number 1__________________________
scaler = StandardScaler() #создание объекта
X_train_std=scaler.fit_transform(X_train)
X_test_std=scaler.transform(X_test)
#___________________________________________
#_________number 2__________________________
np.set_printoptions(precision=4)
mean_vecs=[]

for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label],axis=0))

""" np.mean(X_train_std[y_train==label]- ищет среднее значение для каждого параматра 
    относительно индекса label. Результат-вектор для каждого класса с 13 значениями"""
#___________________________________________

#_________number 3_____________________________________________________________________________________

d=13
S_W=np.zeros((d,d))
for label in range(1,4):
    class_scater=np.cov(X_train_std[y_train==label].T)
    S_W+=class_scater #Матрица разброса внутри класса
""" Для вычисления матрицы разброса внутрикласса спользуем функцию для нахождения
    матрицы ковариации. Так как ковариационная матрица - это нормализованная версия матрицы разброса"""

mean_overall=np.mean(X_train_std, axis=0)
mean_overall=mean_overall.reshape(d,1)
S_B=np.zeros((d,d))
for i, mean_vec in enumerate(mean_vecs):
    N_i=X_train[y_train==i+1,:].shape[0]
    mean_vec=mean_vec.reshape(d,1)
    S_B+=N_i*(mean_vec-mean_overall).dot((mean_vec-mean_overall).T)# Матрица разброса между классами
#________________________________________________________________________________________________________

#_____________________number 4___________________________________________________________________________
eign_vals, eign_vecs=np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

eign_pairs=[(np.abs(eign_vals[i]), eign_vecs[:,i]) for i in range(len(eign_vals))]
"""Полученная конструкция: list(абослютное значение 1    вектор 1
                                    ...                   ...
                                абсолютное значение i    вектор i )"""
#___________________________________________
#_______number 5____________________________
eign_pairs_s=sorted(eign_pairs, key=lambda k: k[0], reverse=True)
# lamda- анонимная функция, в данном случае k[0]-сортировка по нулеыойму столбцу
#print(eign_pairs)
#print('\neign_pairs_s',eign_pairs_s)

w=np.hstack((eign_pairs_s[0][1][:,np.newaxis].real,
          eign_pairs_s[1][1][:,np.newaxis].real))
#w-проекционна матрица
#real-преобразование в вещественные числа

print('\nw=',w)
#___________________________________________

#_____________number 6______________________
X_train_lda=X_train_std.dot(w)
X_test_lda=X_test_std.dot(w)
#___________________________________________
for e_v in eign_pairs_s:
    print(e_v[0])


def plot_decision_regions(X, y, classifer, test_idx=None, resolution=0.2):

	#to configure the token generator & palette
	markers=('s', 'x', 'o', '^', 'v')
	colors =('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	#unique-возвращает уникальные значения из входного массива или столбца или индекса DataFrame
	#to bring the surface of the solution
	x1_min, x1_max=X[:,0].min()-1, X[:,0].max()+1
	x2_min, x2_max=X[:,1].min()-1, X[:,1].max()+1
	xx1, xx2=np.meshgrid(np.arange(x1_min, x1_max, resolution),
		np.arange(x2_min, x2_max, resolution))
	#mesgrid- создание координатной сетки на основе векторов

	Z=classifer.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	# ravel (не создает новый массив а использует существующий.
	# если изменить выходные данные может измениться основной массив)
	Z=Z.reshape(xx1.shape)
	#reshape-придаеет новую форму массиву по ворме входящего массива
	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

	# contourf рисуйте контурные линии и заполненные контуры
		#alpha - Значение смешивания альфа, между 0 (прозрачным) и 1 (непрозрачным).
		#cmap - цветовая карта
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	# to show all

	for idx, cl, in enumerate(np.unique(y)):
	#enmerate- создает кортеж из входных данных

		plt.scatter(x=X[y==cl,0], y=X[y==cl,1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
		#scatter-позволяет отобразить три численных параметра (на осях + размер точки), а так же указать класс объекта.
		# select test samples

LR=LogisticRegression()
LR.fit(X_train_lda,y_train)
plot_decision_regions(X_test_lda, y_test, classifer=LR)


plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper right')
plt.show()
#____________________________________________
