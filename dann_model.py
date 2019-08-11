import tensorflow as tf

@tf.custom_gradient
def flip_gradient(x, l):
    def grad(dy):
        return tf.negative(dy) * l, None
    return tf.identity(x), grad

class DANNModel(object):

    def __init__(self, nb_classes, batch_size):
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self._define_model()

    def _define_model(self):
        
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 3])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.y_dom = tf.placeholder(tf.float32, [None, 2])
        self.lamda = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])

        with tf.variable_scope('feature'):
            conv1 = tf.layers.conv2d(self.X, 32, (5,5), padding='same')
            pool1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')

            conv2 = tf.layers.conv2d(pool1, 46, (5,5), padding='same')
            pool2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
            
            self.feature = tf.layers.flatten(pool2)

        with tf.variable_scope('class_prediction'):
            
            self.features_to_classify = tf.cond(
                self.train,
                lambda: tf.slice(self.feature, [0,0], [self.batch_size // 2, -1]),
                lambda: self.feature)

            self.class_labels = tf.cond(
                self.train,
                lambda: tf.slice(self.y, [0,0], [self.batch_size // 2, -1]),
                lambda: self.y)
            
            fc1 = tf.layers.dense(self.features_to_classify, 128, activation='relu')
            fc2 = tf.layers.dense(fc1, 128, activation='relu')
            
            self.class_logits = tf.layers.dense(fc2, self.nb_classes)
            self.class_prediction = tf.nn.softmax(self.class_logits)
            self.class_loss = tf.losses.softmax_cross_entropy(logits=self.class_logits, onehot_labels=self.class_labels)

        with tf.variable_scope('domain_prediction'):

            gr = flip_gradient(self.feature, self.lamda)

            fc1_d = tf.layers.dense(gr, 128, activation='relu')
            
            self.domain_logits = tf.layers.dense(fc1_d, 2)
            self.domain_prediction = tf.nn.softmax(self.domain_logits)
            self.domain_loss = tf.losses.softmax_cross_entropy(logits=self.domain_logits, onehot_labels=self.y_dom)
