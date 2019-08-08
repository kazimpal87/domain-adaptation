import tensorflow as tf

class DANNModel(object):

    def __init__(self, input_shape, nb_classes, batch_size):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self._define_model()

    def _define_model(self):
        
        self.X = tf.placeholder(tf.float32, self.input_shape)
        self.y = tf.placeholder(tf.float32, [None, self.nb_classes])
        self.y_dom = tf.placeholder(tf.float32, [None, 2])
        self.lamda = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])

        with tf.variable_scope('feature'):
            conv1 = tf.layers.conv2d(self.X, 32, (5,5))
            pool1 = tf.layers.max_pooling2d(conv1, (2,2))

            conv2 = tf.layers.conv2d(pool1, 64, (5,5))
            pool2 = tf.layers.max_pooling2d(conv2, (2,2))
            
            self.feature = tf.layers.flatten(pool2)

        with tf.variable_scope('class_prediction'):
            
            self.features_to_classify = tf.cond(
                self.train,
                lambda: tf.slice(self.feature, [0,0], [self.batch_size // 2, -1] ),
                lambda: self.feature)

            self.class_labels = tf.cond(
                self.train,
                lambda: tf.slice(self.y, [0,0], [self.batch_size // 2, -1] ),
                lambda: self.y)
            
            fc1 = tf.layers.dense(self.features_to_classify, 128, activation='relu')
            fc2 = tf.layers.dense(fc1, 128, activation='relu')
            class_logits = tf.layers.dense(fc2, self.nb_classes)

            self.class_prediction = tf.nn.softmax(class_logits)
            self.class_loss = tf.losses.softmax_cross_entropy(self.class_labels, class_logits)

        with tf.variable_scope('domain_prediction'):

            gr = gradient_reversal(self.feature, self.lamda)

            fc1 = tf.layers.dense(gr, 128, activation='relu')
            domain_logits = tf.layers.dense(fc1, 2)

            self.domain_prediction = tf.nn.softmax(domain_logits)
            self.domain_loss = tf.losses.softmax_cross_entropy(self.y_dom, domain_logits)


