import tensorflow as tf

class NetworkBuider:
    def __init__(self):
        pass

    def attach_conv_layer(self, input_layer, output_size=32, feature_size=[5,5],strides=[1,1,1,1], padding="SAME", summary=False):
        with tf.name_scope(name="conv") as scope:
            input_size=input_layer.get_shape().as_list()[-1]
            conv_weights=tf.Variable(tf.random_normal([feature_size[0], feature_size[1], input_size, output_size]), name="conv_weights")
            conv_biases=tf.Variable(tf.random_normal([output_size], name="conv_biases"))
            conv_layer=tf.nn.conv2d(input_layer, conv_weights, strides, padding, name="conv_layer")
            if summary:
                tf.summary.histogram(conv_weights.name,values=conv_weights)
            return conv_layer


    def attach_pooling_layer(self, input_layer, k_size=[1,2,2,1], strides=[1,2,2,1], padding="SAME"):

        with tf.name_scope(name="pooling") as scope:
            return tf.nn.max_pool(input_layer, ksize=k_size, strides=strides, padding=padding, name="pooling_layer")


    def attach_flatten_layer(self,  input_layer):
        with tf.name_scope(name="flatten") as scope:
            input_size = input_layer.get_shape().as_list()
            new_size = input_size[-1]*input_size[-2]*input_size[-3]
            return  tf.reshape(input_layer, [-1, new_size])

    def attach_dense_layer(self, input_layer, output_size=32, summary=False):
        with tf.name_scope("dense") as scope:
            input_size = input_layer.get_shape().as_list()[-1]
            weights=tf.Variable(tf.random_normal([input_size, output_size]), name="dense_weight")
            biases=tf.Variable(tf.random_normal([output_size], name="biases"))
            if summary:
                tf.summary.histogram(weights.name, weights)
            return  tf.matmul(input_layer, weights)+biases
    def attach_relu_layer(self, input_layer):
        with tf.name_scope("Activation") as scope:
            return tf.nn.relu(input_layer, name="relu")
    def attach_softmax_layer(self, input_layer):
        with tf.name_scope("softmax") as scope:
            return tf.nn.softmax(input_layer)

    def attach_sigmoid_layer(self, input_layer):
        with tf.name_scope("sigmoid") as scope:
            return tf.nn.sigmoid(input_layer)
