from neuralnetwork.networkbuilder import NetworkBuider
import tensorflow as tf
from neuralnetwork.DatasetGenerator import DataSetGenerator
import time
import os
import numpy as np

with tf.name_scope("Input_layer") as scope:
    Input = tf.placeholder("float", shape=[None, 128,128,1], name="input")


with tf.name_scope("Target_layer") as scope:
    Target = tf.placeholder("float", shape=[None, 2], name="Target")

nb = NetworkBuider()
with tf.name_scope("mymodel") as scope:
    model=Input
    model = nb.attach_conv_layer(model, output_size=32,summary=True)
    model = nb.attach_relu_layer(model)
    model=nb.attach_conv_layer(model, output_size=32,summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_pooling_layer(model)

    model = nb.attach_conv_layer(model, output_size=64, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_conv_layer(model, output_size=64,summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_pooling_layer(model)

    model = nb.attach_conv_layer(model, output_size=128, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_conv_layer(model, output_size=128,summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_pooling_layer(model)

    model = nb.attach_flatten_layer(input_layer=model)
    model = nb.attach_dense_layer(input_layer=model, output_size=200, summary=True)
    model=nb.attach_dense_layer(model)
    model = nb.attach_dense_layer(model)
    model = nb.attach_dense_layer(model,output_size=2, summary=True)
    model = nb.attach_softmax_layer(model)
    prediction = model

with tf.name_scope("optimizer") as scope:
    global_itr= tf.Variable(0, name="global_itr", trainable=False)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Target, name="softmax_cost_function")
    cost = tf.reduce_mean(cost)
    tf.summary.scalar("cost",cost)

    optimizer = tf.train.AdamOptimizer().minimize(cost, global_step=global_itr)

with tf.name_scope('accuracy') as scope:
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

db = DataSetGenerator("E:/datasets/train/train")

epochs =10
batchSize = 10

saver = tf.train.Saver()
model_save_path = "E:/datasets/train/train"
model_name = "model"
with tf.Session() as sess:
    summaryMerged = tf.summary.merge_all()

    filename = "E:/run/run" + time.strftime("%H+%M+%S")
    tf.global_variables_initializer().run()

    if os.path.exists(model_save_path + 'checkpoint'):
        # saver = tf.train.import_meta_graph('./saved '+modelName+'/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_save_path))
    writer = tf.summary.FileWriter(filename, sess.graph)

    for epoch in range(epochs):
        batches = db.get_mini_batches(batchSize, (128, 128), allchannel=False)
        for imgs, labels in batches:
            imgs = np.divide(imgs, 255)
            error, sumOut, acu, steps, _ = sess.run([cost, summaryMerged, accuracy, global_itr, optimizer],feed_dict={Input: imgs, Target: labels})
            writer.add_summary(sumOut, steps)
            print("epoch=", epoch, "Total Samples Trained=", steps * batchSize, "err=", error, "accuracy=", acu)
