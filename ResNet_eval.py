import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
import data_input
import model
import numpy as np


def evaluate(filename, model_dir, is_test):
    with tf.Graph().as_default():
        test_data_iter = data_input.data_iterator(num_epochs=1, batch_size=24, tf_filename=filename)
        image, label = test_data_iter.get_next()

        # # initialize save to load variables
        # saver = tf.train.Saver()

        # build graph
        logits = model.inference(images=image, tf_training=False)

        # compute softmax & accuracy of logits
        predictions = tf.nn.softmax(logits=logits)
        accuracy = tf.equal(tf.argmax(input=predictions, axis=1), tf.argmax(input=label, axis=1))

        # restore model
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_dir)
            tf.train.Saver().restore(sess=sess, save_path=ckpt.model_checkpoint_path)

            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Model restored at global step = ', global_step)

            step = 0
            output = 0.0
            try:
                while True:
                    result = sess.run(accuracy)
                    output += 100 * (np.sum(result) / 24)
                    step += 1
            except tf.errors.OutOfRangeError:
                pass
            if is_test:
                print('Test accurcy = %.3f' % (output / step))
                print('Number of total steps %d' % step)
            else:
                print('Train accuracy = %.3f' % (output / step))
                print('Number of total steps %d' % step)


evaluate(filename='train.tfrecords', model_dir='train_dir_1', is_test=False)
evaluate(filename='test.tfrecords', model_dir='train_dir_1', is_test=True)
