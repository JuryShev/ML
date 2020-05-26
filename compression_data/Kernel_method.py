''' Анализ главных компонент ядра


1 Вычислить матрицу ядра (матрицу подобия) k,
2 Центрировать матрицу ядра k
3 Взять верхние k собственных векторов центрированной матрицы ядра,
  основываясь на соответствующих им собственных значениях, которые ранжированы по убыванию их величины (длины).
'''

from scipy.spatial.distance import pdist, squareform, sqeuclidean

# pdist- вычисление расстояние между попарными точками(Евкл. расст)
#Формат расстояния матрицы(преобразует вектор расстояния в матричный вид)

from scipy import exp
from scipy.linalg import eigh

# eigh-ищет мобственные значения и вектора включая матрицы с комплексными числами

import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def rbf_kernel_pca(X, gamma, n_components):

    '''
    Реализация ядерного РСА с РБФ в качестве ядра.

    Параметры :     X:(NumPy ndarray)
                    форма=[n_samples, k_features]
                    (входные данные)

                    gamma: float
                    (настроечные параметры ядра)

                    n_components: int
                    (число возвращаемых компонент)

    Возвращает:     X_pc: (NumPy ndarray)
                    форма=[n_samples, k_features]
                    (спроецированный набор данных)
    '''

    #______________number 1_______________________________________________

    # Вычислить попарно евклидово расстоняние в наборе данных размера MxN
    sq_dist=pdist(X, 'sqeuclidean')
    # Попарно конвертровать расстояние в квадратную матрицу
    mat_sq_dist = squareform(sq_dist)
    # вычислить семитричную матрицу ядра
    K=exp(-gamma*mat_sq_dist)

    #_____________________________________________________________________

    # ______________number 2_______________________________________________
    N=K.shape[0]
    one_n=np.ones((N,N))/N
    #one_n-матрица NxN с размерностью ядра K и значениями 1/N
    K=K-one_n.dot(K)-K.dot(one_n)+one_n.dot(K).dot(one_n)
    # _____________________________________________________________________

    # ______________number 3_______________________________________________
    # Извлечь собственные пары из центрированной матрицы ядра
    # функция numpy.eigh возвращает их в отсортированном порядке,
    eign_vals, eign_vecs = eigh(K)

    # _____________________________________________________________________

    # ______________number 4_______________________________________________

    X_pc=np.column_stack((eign_vecs[:, -i] for i in range(1, n_components+1)))

    return X_pc
    # _____________________________________________________________________


def rbf_kernel_pcan(X, gamma, n_components):

    '''
    Реализация ядерного РСА с РБФ в качестве ядра.

    Параметры :     X:(NumPy ndarray)
                    форма=[n_samples, k_features]
                    (входные данные)

                    gamma: float
                    (настроечные параметры ядра)

                    n_components: int
                    (число возвращаемых компонент)

    Возвращает:     alphas: (NumPy ndarray)
                    (верхние собственные веетора)

                    gamma: (NumPy ndarray)
                    (верхние собственные значения)
    '''

    #______________number 1_______________________________________________

    # Вычислить попарно евклидово расстоняние в наборе данных размера MxN
    sq_dist=pdist(X, 'sqeuclidean')
    # Попарно конвертровать расстояние в квадратную матрицу
    mat_sq_dist = squareform(sq_dist)
    # вычислить семитричную матрицу ядра
    K=exp(-gamma*mat_sq_dist)

    #_____________________________________________________________________

    # ______________number 2_______________________________________________
    N=K.shape[0]
    one_n=np.ones((N,N))/N
    #one_n-матрица NxN с размерностью ядра K и значениями 1/N
    K=K-one_n.dot(K)-K.dot(one_n)+one_n.dot(K).dot(one_n)
    # _____________________________________________________________________

    # ______________number 3_______________________________________________
    # Извлечь собственные пары из центрированной матрицы ядра
    # функция numpy.eigh возвращает их в отсортированном порядке,
    eign_vals, eign_vecs = eigh(K)


    # _____________________________________________________________________

    # ______________number 4_______________________________________________

    alphas=np.column_stack((eign_vecs[:, -i] for i in range(1, n_components+1)))
    lambdas = [eign_vals[-i] for i in range(1, n_components + 1)]

    return alphas, lambdas
    # _____________________________________________________________________
def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist=np.array([np.sum(x_new-row)**2 for row in X])
    k=np.exp(-gamma*pair_dist)
    return k.dot(alphas/lambdas)

#___________________________exmple dist 105&103_____________________________
#def cal_pdist1(X):                                                         #
    # Y = X                                                                 #
    # XX = np.einsum('ij,ij->i', X, X)[np.newaxis, :]                       #
    #переумножается мтрица на саму себя и берется главная диагональ         #
    # #ij, ij->i                                                            #
    # YY = XX.T                                                             #
    # distances = -2 * np.dot(X, Y.T)                                       #
    # distances += XX                                                       #
    # distances += YY                                                       #
    # return (distances)                                                    #
#____________________________________________________________________________

#def project_x(x_new, X, gamma,alphas, labdas):


X, y = make_moons(n_samples=100, random_state=123)
scikit_pca=PCA(n_components=2)
#X_spca=scikit_pca.fit_transform(X)
X_spca=rbf_kernel_pca(X, gamma=15, n_components=2)
alphas, lambdas=rbf_kernel_pcan(X, gamma=15, n_components=2)
x_new=X[25]
x_repr=project_x(x_new,X,gamma=15,alphas=alphas,lambdas=lambdas)

fig, ax=plt.subplots(nrows=1, ncols=2, figsize=[7,3])

ax[0].scatter(X_spca[y==0,0], X_spca[y==0,1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1,0], X_spca[y==1,1], color='blue', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==0,0], np.zeros((50,1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1,0], np.zeros((50,1))-0.02, color='blue', marker='^', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()