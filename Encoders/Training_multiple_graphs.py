import tensorflow as tf
import numpy as np
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def shuffle_batch_encoder(X, n_batches):
    rnd_idx = np.random.permutation(len(X))
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch = X[batch_idx]
        yield X_batch

def plot_image(image0, image1):
    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plt.imshow(image0, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(image1, cmap="gray", interpolation="nearest")
    plt.show()

def train_autoencoder(X_input, n_neurons,
                       learning_rate=0.001,l2_reg=0.0001,
                       n_epochs=3,batch_size = 30, seed=48,
                       hidden_activation=tf.nn.elu,
                       output_activation=tf.nn.elu):
    graph = tf.Graph()

    with graph.as_default():

        tf.set_random_seed(seed)
        he_init = tf.contrib.layers.variance_scaling_initializer()  # initializer Xavier for wight in range sqrt(6/n_outputs+n_inputs)
        l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        X = tf.placeholder(tf.float32, [None, X_input.shape[1] ])

        my_dense_layer = partial(tf.layers.dense,
                                 kernel_initializer=he_init,
                                 kernel_regularizer=l2_regularizer)

        hidden=my_dense_layer(X, n_neurons, activation=hidden_activation, name="hidden")
        output = my_dense_layer(hidden,X_input.shape[1], activation=output_activation, name="output")

        reconstruction_loss = tf.reduce_mean(tf.square(output - X))  # MSE

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # Graph.Keys.REGULARISATION_LOSSES
        # хронит потери L1, L2 регуляризации
        loss = tf.add_n([reconstruction_loss] + reg_losses)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        init.run()
        for epoch in range(n_epochs):
            j = 0
            n_batches = len(X_train) // batch_size
            for X_batch in shuffle_batch_encoder(X_input, n_batches):
                sess.run(training_op, feed_dict={X: X_batch})
                j = j + 1
                if j == n_batches or j == 0:
                    loss_train = loss.eval(feed_dict={X: X_batch})
                    print("\r{}".format(epoch), "Train MSE:", loss_train)
                    j = 0
        params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        hidden_val = hidden.eval(feed_dict={X: X_input})
        """НУЖНО ОПИСАТЬ tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)] """
    return hidden_val, params["hidden/kernel:0"], params["hidden/bias:0"], params["output/kernel:0"], params[
            "output/bias:0"]







(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
training = tf.placeholder_with_default(False, shape=(), name='training')# необходим чтобы передать  tf.layers.batch_normalization
                                                                        # среднее смещение определять по сему набору данных или по пакету
                                                                        # False-весь набор данных, True-пакет
hidden_output, W1, bias_1, W4, bias_4 = train_autoencoder(X_train, n_neurons=300, n_epochs=10, batch_size=30)

_,W2, bias_2, W3, bias_3 = train_autoencoder(hidden_output, n_neurons=150, n_epochs=10, batch_size=30)

reset_graph()
n_inputs=28*28
X_for_test=tf.placeholder(tf.float32, [None, n_inputs])
activation=tf.nn.elu
hidden1=activation(tf.matmul(X_for_test, W1)+bias_1)
hidden2=activation(tf.matmul(hidden1, W2)+bias_2)
hidden3=activation(tf.matmul(hidden2, W3)+bias_3)
outputs=tf.matmul(hidden3, W4)+bias_4

init = tf.global_variables_initializer()
index_input=7
with tf.Session() as sess:
    init.run()
    X_show0 = X_test[index_input, :].reshape(1, n_inputs)
    X_show0 = X_show0.reshape(28, 28)

    X_show = outputs.eval(feed_dict={X_for_test: X_test[index_input, :].reshape(1, n_inputs)})
    X_show = X_show.reshape(28, 28)
    plot_image(X_show0, X_show)

        