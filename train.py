import tensorflow as tf
import datasets

sess = tf.Session()
batch_size = 4

X, y, _, _ = datasets.get_mnist_vs_mnistm()

model = MNISTModel()

lr = tf.placeholder(tf.float32, [])

class_loss = tf.reduce_mean(model.class_loss)
domain_loss = tf.reduce_mean(model.domain_loss)
total_loss = pred_loss + domain_loss

train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(total_loss)

with sess.as_default():
    
