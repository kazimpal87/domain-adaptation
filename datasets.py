import tensorflow as tf
import numpy as np
import pickle as pkl

def load_mnist(batch_size):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    mnist_train = mnist.train.images.reshape(55000, 28, 28, 1)
    mnist_test = mnist.test.images.reshape(10000, 28, 28, 1)

    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    
    return mnist_train, mnist.train.labels, mnist_test, mnist.test.labels

def load_mnist_m(batch_size):
    mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
    
    mnistm_train = mnistm['train'].astype(np.float32) / 255
    mnistm_test = mnistm['test'].astype(np.float32) / 255
    
    return mnistm_train, mnistm_test

def get_mnist_vs_mnistm(batch_size=32):
    mnist_train, mnist_train_labels, mnist_test, mnist_test_labels = load_mnist(batch_size // 2)
    mnistm_train, mnistm_test = load_mnist_m(batch_size // 2)

    pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))

    mnist_train -= pixel_mean
    mnist_test -= pixel_mean
    mnistm_train -= pixel_mean
    mnistm_test -= pixel_mean

    gen_source_train = batch_generator([mnist_train, mnist_train_labels], batch_size // 2)
    gen_target_train = batch_generator([mnistm_train, mnist_train_labels], batch_size // 2)
    gen_source_test = batch_generator([mnist_test, mnist_test_labels], batch_size // 2)
    gen_target_test = batch_generator([mnistm_test, mnist_test_labels], batch_size // 2)

    return gen_source_train, gen_target_train, gen_source_test, gen_target_test

def shuffle_aligned_list(data):
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]

def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]