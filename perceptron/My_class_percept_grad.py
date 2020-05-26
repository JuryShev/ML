import numpy as np 
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

class Perceptron_ADALINE (object):

	'''
Parametrs:

eta		Type:flout
		Name:Learning pace
		Value range:[0-1]

n_iter	Type:int
		Name:Number of passes in the sample

Attributes:

w_		Type:array
		Name:Weighting coefficients

errors_	Type:List
		Name:The number of classification errors/epoch


'''

	def __init__(self, eta=0.01, n_iter=10):

		self.eta=eta
		self.n_iter=n_iter

	def fit(self, X, y,):

		'''
		Parametrs:

		X	Type:array
			Form:[n_samples, n_features]
				n_samples: number of samples
				n_features: number of signs

		y	Type:array
			Form:[n_samples]
				n_samples:target values

		Return:
		
		self:object
		'''

		self.w_=np.zeros(1+X.shape[1])
		self.cost_=[]

		for i in range(self.n_iter):
			output=self.net_input(X) #проход чере функцию чистого хода
			#_______Grad_function___________________
			errors=y-output
			self.w_[1:] += self.eta*X.T.dot(errors)
			self.w_[0] += self.eta*errors.sum()
			#_______________________________________
			cost=(errors**2).sum()/2
			self.cost_.append(cost)

		return self

	def net_input(self,X):
		# Calculate net input (Функция чистого входа)
		return np.dot(X, self.w_[1:]) + self.w_[0] 

	def activation(self, X):
		# Calculate function_activ 
		return self.net_input(X)

	
	def predict (self, X):
		# Return class label after single jump
		return np.where(self.activation(X) >= 0.0, 1, -1)

	def data_std(self, X):
		X_std=X
		X_std[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()
		X_std[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()
		return X_std


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
		if test_idx:
			X_test, y_test=X[test_idx, :], y[test_idx]
			plt.scatter(X_test[:, 0], X_test[:,1], c='',
			 alpha=1.0, linewidths=1, marker='o', s=55, label='тестовый набор')




