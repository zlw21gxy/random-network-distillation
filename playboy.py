import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

X = np.random.randn(1000, 100)
y = np.zeros(1000)

X[X[:, 0] > 0] += 3
y[X[:, 0] > 0] += 1

X_test = np.random.randn(1000, 100)
y_test = np.zeros(1000)

X_test[X_test[:, 0] > 0] += 3
y_test[X_test[:, 0] > 0] += 1

# ##
# fig = plt.figure(0, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#
# plt.cla()
# pca = decomposition.PCA(n_components=3)
# pca.fit(X)
# X = pca.transform(X)
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.choose(y.astype(int), [1, 0]))
#
# ##
# fig = plt.figure(1, figsize=(4, 3))
# pca = decomposition.PCA(n_components=2)
# pca.fit(X)
# X = pca.transform(X)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()


# X (batch_size, dim)


def mlp():
    input_ph = tf.placeholder(dtype=float, shape=(None, 100))
    w1 = tf.get_variable(name='w1', shape=(100, 128))
    b1 = tf.get_variable(name='b1', shape=(128, ))
    w2 = tf.get_variable(name='w2', shape=(128, 32))
    b2 = tf.get_variable(name='b2', shape=(32, ))
    w3 = tf.get_variable(name='w3', shape=(32, 2))
    b3 = tf.get_variable(name='b3', shape=(2, ))
    x = input_ph
    x1 = tf.nn.tanh(tf.matmul(x, w1) + b1)
    x2 = tf.nn.tanh(tf.matmul(x1, w2) + b2)
    x3 = tf.matmul(x2, w3) + b3
    output_pred = x3

    return input_ph, output_pred


input_ph, output_pred = mlp()
output_ba = tf.placeholder(dtype='int32', shape=(None,))
output_ph = tf.one_hot(output_ba, 2)
loss = tf.losses.softmax_cross_entropy(output_ph, output_pred)

opt = tf.train.AdamOptimizer().minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
batch_size = 64

for t in range(3000):
    ind = np.random.randint(0, len(X), size=batch_size)
    input_batch = X[ind]
    output_batch = y[ind]
    # print(t)

    _, l_train = sess.run([opt, loss], feed_dict={input_ph:input_batch, output_ba:output_batch})

    if t % 1000 == 0:
        print('{0:04d} loss:{1: .9f}'.format(t, l_train))
        saver.save(sess, './tmp/model.ckpt')

##
y_pre = sess.run(output_pred, feed_dict={input_ph: X_test})
y_pre = np.argmax(y_pre, axis=1)
acc = sum(y_pre==y_test)/len(y_test)
print(acc)



















