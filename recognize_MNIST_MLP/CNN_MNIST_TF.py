import tensorflow as tf
import numpy as np
height=28
width=28
channels=1
n_inputs=height*width
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


conv1_fmaps=32
conv1_stride=1
conv1_ksize=3
conv1_pad='SAME'


conv2_fmaps=64
conv2_stride=2
conv2_ksize=3
conv2_pad='SAME'

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 10
learning_rate=0.01
batch_size=30
n_epochs=100
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)



def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

"""____________________Arhotecture_of CNN__________________________________________________"""

with tf.name_scope("Input"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_rehaped=tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")

    conv1=tf.layers.conv2d(X_rehaped,conv1_fmaps, conv1_ksize,
                           conv1_stride, padding=conv1_pad, activation=tf.nn.relu, name="conv1")
    conv2=tf.layers.conv2d(conv1,conv2_fmaps, conv2_ksize,
                           conv2_stride, padding=conv2_pad, activation=tf.nn.relu, name="conv2")

with tf.name_scope("maxpool"):

    maxpool_1=tf.nn.max_pool(conv2, ksize=[1,2,2,1],
                             strides=[1,2,2,1],padding="VALID")

    pool3_flat = tf.reshape(maxpool_1, shape=[-1, pool3_fmaps * 7 * 7])

with tf.name_scope("fc1"):

    fc1_1=tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1_1")

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1_1, n_outputs, name="outputs")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
"""______________________________________________________________________________________"""

"""_____________TRAIN_MODEL_&_UPDATE_WEIGHT_____________________________________________________"""
with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)  # global_steps+1
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),name="accuracy")
"""______________________________________________________________________________________"""

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            out_0=conv1.eval(feed_dict={X: X_batch, y: y_batch})
            out=sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)




