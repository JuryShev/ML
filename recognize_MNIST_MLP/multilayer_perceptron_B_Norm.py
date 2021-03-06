import tensorflow as tf
import numpy as np
from datetime import datetime
from functools import partial

##################################################
import os
status = tf.pywrap_tensorflow.TF_NewStatus()
#print(tf.pywrap_tensorflow.GetChildren(tf.compat.as_bytes(os.getcwd()), status))
print(os.listdir(os.getcwdb()))
##################################################
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_hidden3 = 50
n_hidden4 = 50
n_hidden5 = 50
n_outputs = 10

batch_norm_momentum = 0.9

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

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

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    he_init = tf.variance_scaling_initializer()#Инициализатор, способный адаптировать свою шкалу к форме тензоров весов
                                               #                (нормальное распределение)


    my_batch_norm_layer=partial(
        tf.layers.batch_normalization,# создан чтобы по умолчанию индексировать параметры
        training=training,
        momentum=batch_norm_momentum
    )

    my_dense_layer = partial(
        tf.layers.dense,
        kernel_initializer=he_init)


    hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
    bn1=tf.nn.elu(my_batch_norm_layer(hidden1))
    hidden2 = my_dense_layer(bn1, n_hidden2, name="hidden2")
    bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
    logits_before_bn = my_dense_layer(bn2, n_outputs, name="outputs")
    logits=my_batch_norm_layer(logits_before_bn)


with tf.name_scope("dnn2"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hiddenG1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hiddenG2")
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hiddenG3")
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hiddenG4")
    hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hiddenG5")
    logits = tf.layers.dense(hidden5, n_outputs, name="outputs_G")

print("hidden1_G=",hidden1.name)

dropout_rate = 0.5  # == 1 - keep_prob
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

with tf.name_scope("dnn_drop"):
    hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu,
                              name="hidden_drop1")
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, activation=tf.nn.relu,
                              name="hidden_drop2")
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs_drop")

print("hidden1_drop=",logits.name)
with tf.name_scope("loss"):


    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=tf.get_default_graph().get_tensor_by_name("dnn_drop/outputs_drop/BiasAdd:0"))
    loss = tf.reduce_mean(xentropy, name="loss")



with tf.name_scope("train"):
    """___Change speed of learning________________"""

                #η0*10^-t/r

    initial_learning_rate = 0.1#η0
    decay_steps=10000#r
    decay_rate=0.1#10^-1
    global_steps=tf.Variable(0, trainable=False, name="global_step") #t
    learning_rate=tf.train.exponential_decay(initial_learning_rate, global_steps,
                                             decay_steps,decay_rate)

    """___________________________________________"""

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss,global_step=global_steps)#global_steps+1
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(tf.get_default_graph().get_tensor_by_name("dnn_drop/outputs_drop/BiasAdd:0"), y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),name="accuracy")

init = tf.global_variables_initializer()
saver = tf.train.Saver()
n_epochs = 50
batch_size = 30

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            out0=sess.run(logits, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)

    save_path = saver.save(sess, "./B_norm_final.ckpt")
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

file_writer.close()

