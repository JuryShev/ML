import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from class_mlp2 import  MostLayPers

def load_mnist(path, kind='train'):
    """Загрузить данные MNIST из пути path"""

    labels_path=os.path.join(path,
                             '%s-labels.idx1-ubyte'
                             %kind)
    images_path = os.path.join(path,
                             '%s-images.idx3-ubyte'
                             %kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n=struct.unpack('II',
                               lbpath.read(8))
        labels=np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols=struct.unpack(">IIII", imgpath.read(16))

        images=np.fromfile(imgpath,
                           dtype=np.uint8).reshape(len(labels),784)
    return images, labels

X_train0, y_train0=load_mnist('D:\Book\programming\PY\pandas\ML\ML_rahka\Chapter 12\DATA\MNIST', kind='train')
print('Тренировка-строки: %d, столбцы %d'%(X_train0.shape[0], X_train0.shape[1]))

X_test0, y_test0=load_mnist('D:\Book\programming\PY\pandas\ML\ML_rahka\Chapter 12\DATA\MNIST', kind='t10k')
print('Тренировка-строки: %d, столбцы %d'%(X_test0.shape[0], X_test0.shape[1]))


#
# for i in range(16):
#     ax= plt.subplot(4,4 ,i+1)
#     im=ax.pcolor(np.random.normal(size=100).reshape([10,10]))
#     plt.tight_layout()
#     plt.title(i)
# plt.savefig("pcolor_4x4.png")

# A = np.random.rand(5, 5)
#
# fig, axs = plt.subplots(1, 3, figsize=(10, 3))
# for ax, interp in zip(axs, ['nearest', 'bilinear', 'bicubic']):
#     ax.imshow(A, interpolation=interp)
#     ax.set_title(interp.capitalize())
#     ax.grid(True)


#__________Save_DATA_to_CSV_______________

# np.savetxt('train_img_csv', X_train, fmt='%i',delimiter=',')
# np.savetxt('train_labels_csv', y_train, fmt='%i',delimiter=',')
# np.savetxt('test_img_csv', X_test, fmt='%i',delimiter=',')
# np.savetxt('test_labels_csv', y_test, fmt='%i',delimiter=',')


X_train=pd.read_csv('train_img_csv',delimiter=',', header=None)
y_train=pd.read_csv('train_labels_csv',delimiter=',', header=None)
X_test=pd.read_csv('test_img_csv',delimiter=',', header=None)
y_test=pd.read_csv('test_labels_csv',delimiter=',', header=None)

X_train=np.uint8(X_train.values)
y_train=np.uint8(y_train.values.T)
X_test=np.uint8(X_test.values)
y_test=np.uint8(y_test.values.T)

#_________________________________________
plt.figure()



for i in range(10) :
    wq = plt.subplot(2, 5, i + 1)
    img0=X_train0[y_train0==i][0].reshape(28,28)
    wq.imshow(img0,interpolation='nearest')

plt.tight_layout()
plt.show()

#___________Train_Neuroll_network__________________

Nn_numb=MostLayPers(n_output=10,
                    n_features=X_train.shape[1],
                    n_hiden=100,
                    l2=0.0001,#0.1(10.53)
                    l1=0.001,#0.1(10.53)
                    epochs=1000,#1000(10.53%)
                    eta=0.0005,#0.0001(10.53)
                    alpha=0.0015,#0.01(10.53%)
                    decrease_const=0,
                    shuffle=True,
                    minibatches=50,
                    random_state=1)
Nn_numb.fit (X_train, y_train, print_progress=True, visual_plot=True)#, visual_plot=True
y_pred=Nn_numb.predict (X_test)
acc=np.sum(y_test==y_pred,axis=1)/y_test.shape[1]

y_pred_train=Nn_numb.predict (X_train)
acc_train=np.sum(y_train==y_pred_train,axis=1)/y_train.shape[1]
print('\n acc=',acc*100)
print('\nacc_train=',acc_train*100)
#__________________________________________________





