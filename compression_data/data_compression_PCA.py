'''Анализ главных компонент-(principal component analysis, РСА)'''

'''это метод линейного преобразования, относящийся к типу обучения без учителя, 
который широко используется в самых разных областях, чаще всего для снижения размерности

Данный метод состоит из следующих этапов:
1 стандартизировать d-мерный набор данных
2 построить ковариационную матрицу
3 разложить ковариационную матрицу на ее собственные векторы и собственные значения (числа);
4 выбрать k собственных векторов, которые соответствуют k самым большим собственным значениям,
  где k - размерность нового подпространства признаков (k << d)
5 создать проекционную матрицу W из «верхних» k собственных векторов;
6 преобразовать «(-мерный входной набор данных X, 
  используя проекционную матрицу W для получения нового ^-мерного подпространства признаков.
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df_wine=pd.read_csv(url)
#df_wine.columns=['метка класса','алкоголь','яблочная кислота','зола','щелочность золы','магний',
#'всего фенолов','фланоиды','фенол нефланоидные','проантоцианты','интенсивность цвета','оттенок','OD280','пролин']
# df_wine.to_csv(r'D:\book\programming\PY\pandas\ML\ML_rahka\data_csv\wine.csv')

#df_wine=pd.read_csv(r'D:\book\programming\PY\pandas\ML\ML_rahka\data_csv\wine.csv')
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

#ВЫЧИСЛЕНИЕ МАТРИЦЫ КОВАРИАЦИИ
#размерность матрицы 13х13
cov_mat=np.cov(X_train_std.T)
#___________________________________________

#_________number 3__________________________
#ВЫЧИСЛЕНИЕ СОБСТВЕННЫХ ВЕТОРОВ И ЗНАЧЕНИЙ
eign_vals, eign_vecs=np.linalg.eig(cov_mat)
tot=sum(eign_vals)
var_exp=[(i/tot) for i in sorted(eign_vals, reverse=True)]
cum_var_exp=np.cumsum(var_exp)
#___________________________________________

#_________number 4__________________________
eign_pairs=[(np.abs(eign_vals[i]), eign_vecs[:,i]) for i in range(len(eign_vals))]

"""Полученная конструкция: list(абослютное значение 1    вектор 1
                                    ...                   ...
                                абсолютное значение i    вектор i )"""

########Пример альтернативной записи#########
# b=np.array([[1,2,3,4,5],[6,7,8,9,10]])    #
# a=np.array([11,12,13,14,15])              #
# c=[]                                      #
# #c=[(a[v],b[:,v]) for v in range(len(a))] #
# for v in range(len(a)):                   #
#                                           #
#     c.append((a[v],b[:,v]))               #
#############################################

eign_pairs.sort(reverse=True)#Сортировка собственных пар
print(eign_pairs[:3])

#_______number 5____________________________
w=np.hstack((eign_pairs[0][1][:,np.newaxis],
          eign_pairs[1][1][:,np.newaxis]))
#w-проекционна матрица
print('\nw=',w)
#___________________________________________

#_____________number 6______________________
X_train_pca=X_train_std.dot(w)
#___________________________________________

#___________Отрисовка_______________________

# color=['r','b','g']
# markers=['s', 'x', 'o']

# for l,c,m in zip(np.unique(y_train), color, markers):
#     plt.scatter(X_train_pca[y_train==l,0], X_train_pca[y_train==l,1],c=c, label=1, marker=m)

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
LR.fit(X_train_pca,y_train)
plot_decision_regions(X_train_pca, y_train, classifer=LR)


plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper right')
plt.show()
#____________________________________________

