#coding: utf-8
from sklearn.datasets import load_iris,load_digits
from sklearn.manifold import MDS
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
MSD梯度下降迭代实现(很有趣的一个实验)
本质：2套数据朝一个方向靠拢！！！
"""

def sklearn_mds(n_com=2):
    mds=MDS(n_components=n_com)
    data=load_digits().data
    target=load_digits().target
    data_2d = mds.fit_transform(data)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=target)
    plt.show()

def cal_pairwise_dist(x):
    '''计算pairwise 距离, x是matrix
        (a-b)^2 = a^2 + b^2 - 2*a*b
        '''
    sum_x=np.sum(np.square(x),1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #自己的理解,可以对比一下实验结果
    #dist = np.add(-2 * np.dot(x, x.T), np.add(sum_x[..., np.newaxis], sum_x[np.newaxis, ...]))
    return dist

def tensor_mds(data,n_dims=2,learning_rate=0.1):
    """

    :param data:
    :param n_dims:
    :param learning_rate:
    :return: (n_samples, n_dims)
    """
    n,feature=data.shape
    X_dist=cal_pairwise_dist(data)

    X=tf.placeholder(name="X",dtype=tf.float32,shape=[n,n])
    Y=tf.get_variable(name="Y",shape=[n,n_dims],initializer=tf.random_uniform_initializer())

    sum_y=tf.reduce_sum(tf.square(Y),1)
    Y_dist=tf.add(tf.transpose(tf.add(-2*tf.matmul(Y,tf.transpose(Y)),sum_y)),sum_y)

    loss=tf.log(tf.reduce_sum(tf.square((X-Y_dist))))

    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()

    n_epochs=1000

    with tf.Session() as sess:
        init.run()
        for i in range(n_epochs):
            training_op.run(feed_dict={X:X_dist})
            if i%200==0:
                loss_val=loss.eval(feed_dict={X:X_dist})
                print("loss:",loss_val)

        data_2d=sess.run(Y)
    return data_2d

if __name__=='__main__':
    iris=load_iris()
    data=iris.data
    Y=iris.target
    data_1=tensor_mds(data,learning_rate=0.01)
    data_2=MDS(n_components=2).fit_transform(data)

    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("my_MDS")
    plt.scatter(data_1[:,0],data_1[:,1],c=Y)

    plt.subplot(122)
    plt.title("sklearn_MDS")
    plt.scatter(data_2[:, 0], data_2[:, 1], c=Y)
    plt.savefig("MDS_2.png")
    plt.show()


































