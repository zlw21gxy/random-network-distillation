# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = (train_images - 128.0)/128.0
test_images = (test_images - 128.0)/128.0
X = train_images
X_test = test_images
y = train_labels
y_test = test_labels

def mlp(input_size=(28, 28), output_size=10):
    input_ph = tf.placeholder(dtype=float, shape=(None, input_size[0], input_size[1]))
    is_train = tf.placeholder(dtype=bool, name='istrain')
    w1 = tf.get_variable(name='w1', shape=(np.prod(input_size), 128), initializer=tf.keras.initializers.glorot_normal())
    b1 = tf.get_variable(name='b1', shape=(128, ))
    w2 = tf.get_variable(name='w2', shape=(128, 64), initializer=tf.keras.initializers.glorot_normal())
    b2 = tf.get_variable(name='b2', shape=(64, ))
    w3 = tf.get_variable(name='w3', shape=(64, output_size), initializer=tf.keras.initializers.glorot_normal())
    b3 = tf.get_variable(name='b3', shape=(output_size, ))
    x = tf.reshape(input_ph, (-1, np.prod(input_size)))
    x1 = tf.layers.batch_normalization(tf.nn.tanh(tf.matmul(x, w1) + b1), training=is_train)
    x1 = tf.layers.dropout(x1, rate=0.5, training=is_train)
    x2 = tf.layers.batch_normalization(tf.nn.tanh(tf.matmul(x1, w2) + b2), training=is_train)
    x2 = tf.layers.dropout(x2, rate=0.5, training=is_train)
    x3 = tf.matmul(x2, w3) + b3
    output_pred = x3
    return input_ph, output_pred, is_train


input_ph, output_pred, is_train = mlp()
output_ba = tf.placeholder(dtype='int32', shape=(None,))  # label
output_ph = tf.one_hot(output_ba, 10)   # one hot label,ground truth
# loss = tf.losses.softmax_cross_entropy(output_ph, output_pred)  # different between ground truth and predict
loss = tf.losses.cosine_distance(output_ph, output_pred, axis=0)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    opt = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
batch_size = 64
epochs = 5
time_step = int(epochs*len(X))
acc_col_train = []
loss_col = []
acc_col_test = []
for t in range(time_step):
    ind = np.random.randint(0, len(X), size=batch_size)
    input_batch = X[ind]
    output_batch = y[ind]  # label
    # print(t)

    _, l_train, train_pre = sess.run([opt, loss, output_pred], feed_dict={input_ph: input_batch,
                                                  output_ba: output_batch,
                                                  is_train: True})

    if t % 3000 == 0:
        y_pre = sess.run(output_pred, feed_dict={input_ph: X_test, is_train: False})
        y_pre = np.argmax(y_pre, axis=1)
        train_pre = np.argmax(train_pre, axis=1)
        train_acc = sum(train_pre == output_batch) / len(output_batch)
        test_acc = sum(y_pre == y_test) / len(y_test)
        print(t)
        print('loss:{0:.9f} train_accuray:{1:.9f} test_accuracy:{2:.9f}'.format(l_train, train_acc, test_acc))
        saver.save(sess, './tmp/model_cos/model_shirt.ckpt')
        acc_col_test.append(test_acc)
        acc_col_train.append(train_acc)
        loss_col.append(l_train)

##
y_pre = sess.run(output_pred, feed_dict={input_ph: X_test, is_train: False})
y_pre = np.argmax(y_pre, axis=1)
acc = sum(y_pre==y_test)/len(y_test)
print(acc)
plt.figure('loss')
plt.plot(loss_col)
plt.figure('accuracy')
plt.plot(range(len(acc_col_train)), acc_col_train, label='train accuracy')
plt.plot(range(len(acc_col_train)), acc_col_test, label='test accuracy')
plt.legend(loc='upper left')
plt.show()
# plt.figure('test_acc')
# plt.plot(acc_col_test)