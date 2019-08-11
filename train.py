import numpy as np
import tensorflow as tf
import datasets
from dann_model import DANNModel

sess = tf.Session()
batch_size = 64
epochs = 10
batches_per_epoch = 55000 // batch_size
num_steps = epochs * batches_per_epoch

gen_source_train, gen_target_train, gen_source_test, gen_target_test = datasets.get_mnist_vs_mnistm(batch_size=batch_size)

model = DANNModel(10, batch_size)

class_loss = tf.reduce_mean(model.class_loss)
domain_loss = tf.reduce_mean(model.domain_loss)
total_loss = class_loss + domain_loss

learning_rate = tf.placeholder(tf.float32, [])
train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

correct_class_pred = tf.equal(tf.argmax(model.class_labels, 1), tf.argmax(model.class_prediction, 1))
class_accuracy = tf.reduce_mean(tf.cast(correct_class_pred, tf.float32))

correct_domain_pred = tf.equal(tf.argmax(model.y_dom, 1), tf.argmax(model.domain_prediction, 1))
domain_accuracy = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

with sess.as_default():
    print('running global_variables_initializer')
    sess.run(tf.global_variables_initializer())
    print('global_variables_initializer ... done ...')

    print('starting training ... ')
    print(num_steps)
    for i in range(1, num_steps + 1):
        p = float(i) / num_steps
        lamda = 2. / (1. + np.exp(-1. * p)) - 1
        lr = 0.01 / (1. + 10 * p)**0.75

        X0, y0 = next(gen_source_train)
        X1, y1 = next(gen_target_train)
        X = np.vstack([X0, X1])
        y = np.vstack([y0, y1])

        y_domain = np.zeros((batch_size,2), dtype=np.float32)
        y_domain[:batch_size//2, 0] = 1
        y_domain[batch_size//2:, 1] = 1

        _, loss, loss_c, loss_d, acc_c, acc_d, cp, dp = sess.run(
            [train_op, total_loss, class_loss, domain_loss, class_accuracy, domain_accuracy, model.class_logits, model.domain_logits],
            feed_dict={
                model.X: X,
                model.y: y,
                model.y_dom: y_domain,
                model.train: True,
                model.lamda: lamda,
                learning_rate: lr})
        
        if i % 1 == 0:
            print('Batch {}, Loss {:.3f},{:.3f},{:.3f}, Acc {:.3f} {:.3f}, LR {:.3f}, Lamda {:.3f}'.format(i, loss, loss_c, loss_d, acc_c, acc_d, lr, lamda))
            print(np.amin(cp), np.amax(cp))
            print(np.amin(dp), np.amax(dp))
            print()

    src_acc = 0
    tar_acc = 0
    dom_acc = 0
    for i in range(1, num_steps + 1):
        X0, y0 = next(gen_source_test)
        X1, y1 = next(gen_target_test)
        X = np.vstack([X0, X1])
        y = np.vstack([y0, y1])
        y_domain = np.zeros((batch_size,2), dtype=np.float32)
        y_domain[:batch_size//2, 0] = 1
        y_domain[batch_size//2:, 1] = 1

        src_acc_i = sess.run(class_accuracy, feed_dict={
                                model.X: X0, 
                                model.y: y0,
                                model.train: False})

        tar_acc_i = sess.run(class_accuracy, feed_dict={
                                model.X: X1, 
                                model.y: y1,
                                model.train: False})

        dom_acc_i = sess.run(domain_accuracy, feed_dict={
                                model.X: X,
                                model.y_dom: y_domain, 
                                model.lamda: 1.0})

        src_acc += src_acc_i * 1.0/num_steps
        tar_acc += tar_acc_i * 1.0/num_steps
        dom_acc += dom_acc_i * 1.0/num_steps

    print(src_acc, tar_acc, dom_acc)