import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

fashion = input_data.read_data_sets('data/fashion', one_hot=True)

# print(fashion.train.images.shape)
# print(fashion.train.labels.shape)
# print(fashion.test.images.shape)
# print(fashion.test.labels.shape)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def addclayer(csize, cnum, x, activation_function = None):

    W = tf.Variable(tf.truncated_normal([csize[0], csize[1], csize[2], cnum], stddev=0.1))
    b = tf.Variable(tf.zeros([1, cnum]) + 0.1)
    h_conv2 = conv2d(x, W) + b

    if activation_function is None:
        h_result = h_conv2
    else:
        h_result = activation_function(h_conv2)

    return max_pool_2x2(h_result)

def addflayer(node_in, node_out, x, keep_drop,activation_function=None):

    # W = tf.Variable(tf.random_normal([node_in, node_out], stddev= 0.1, dtype=tf.float32))
    W = tf.Variable(tf.truncated_normal([node_in, node_out], stddev=0.1))
    b = tf.Variable(tf.zeros([1, node_out])+0.1)
    # cnn后矩阵变形
    x = tf.reshape(x, [-1, node_in])
    z = tf.matmul(x, W) + b

    if activation_function is None:
        a = z
    else:
        a = activation_function(z)
    return tf.nn.dropout(a, keep_drop)

x_ = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_drop = tf.placeholder(tf.float32)


batch_size = 50
batch_num = fashion.train.num_examples // batch_size

x_img = tf.reshape(x_, [-1, 28, 28, 1])

layer_c1 = addclayer([3, 3, 1], 20, x_img, activation_function=tf.nn.relu)
layer_c2 = addclayer([3, 3, 20], 30, layer_c1, activation_function=tf.nn.relu)

layer1 = addflayer(7*7*30, 20, layer_c2, keep_drop, activation_function=tf.nn.relu)

layer2 = addflayer(20, 10, layer1, keep_drop, activation_function=None)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer2, labels=y_))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# 判断是否准确
correct_prediction = tf.equal(tf.argmax(layer2, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for times in range(1001):
        for i in range(batch_num + 1):
            batch_x, batch_y = fashion.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x_: batch_x, y_: batch_y, keep_drop: 0.5})
        acc = sess.run(accuracy, feed_dict={x_: fashion.test.images, y_: fashion.test.labels, keep_drop: 1})
        print(times)
        print('Times: ' + str(times) + ',acc: ' + str(acc))
# with tf.Session() as sess:
#     print("start")
#     sess.run(tf.global_variables_initializer())
#     for times in range(101):
#
#         sess.run(train_step, feed_dict={x_: fashion.train.images, y_: fashion.train.labels})
#         acc = sess.run(accuracy, feed_dict={x_: fashion.test.images, y_: fashion.test.labels})
#         print('Epoch: ' + str(times) + ',acc: ' + str(acc))