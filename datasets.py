import tensorflow as tf
import numpy as np
import pickle as pkl
from utils import *

def load_mnist(batch_size):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    mnist_train = mnist.train.images.reshape(55000, 28, 28, 1)
    mnist_test = mnist.test.images.reshape(10000, 28, 28, 1)
    
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

    dataset_train = make_dataset(mnist_train, mnist.train.labels, batch_size)
    dataset_test  = make_dataset(mnist_test,  mnist.test.labels,  batch_size)
    
    return dataset_train, dataset_test

def load_mnist_m(batch_size):
    mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
    
    mnistm_train = mnistm['train'].astype(np.float32) / 255
    mnistm_test = mnistm['test'].astype(np.float32) / 255

    dataset_train = make_dataset(mnistm_train, np.zeros((mnistm_train.shape[0], 10)), batch_size)
    dataset_test  = make_dataset(mnistm_test,  np.zeros((mnistm_train.shape[0], 10)), batch_size)
    
    return dataset_train, dataset_test

def get_mnist_vs_mnistm(batch_size=32):
    dataset_s_train, dataset_s_test = load_mnist(batch_size // 2)
    dataset_t_train, dataset_t_test = load_mnist_m(batch_size // 2)

    iter_s_train = dataset_s_train.make_one_shot_iterator()
    iter_s_test =  dataset_s_test.make_one_shot_iterator()
    iter_t_train = dataset_t_train.make_one_shot_iterator()
    iter_t_test =  dataset_t_test.make_one_shot_iterator()

    X_s_train, y_s_train = iter_s_train.get_next()
    X_t_train, y_t_train = iter_t_train.get_next()
    X_s_test, y_s_test = iter_s_test.get_next()
    X_t_test, y_t_test = iter_t_test.get_next()

    X_train = tf.concat([X_s_train, X_t_train], axis=0)
    X_test =  tf.concat([X_s_test,  X_t_test],  axis=0)
    y_train = tf.concat([y_s_train, y_t_train], axis=0)
    y_test =  tf.concat([y_s_test,  y_t_test],  axis=0)

    return X_train, y_train, X_test, y_test

