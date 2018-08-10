import tensorflow as tf
import os
import model
import data_input
import time
from datetime import datetime
import param

num_gpus = param.num_gpus
num_epochs = param.num_epochs
batch_size = param.batch_size


def accuracy(logits, input_labels):
    return tf.equal(tf.argmax(logits, 1), tf.argmax(input_labels, 1))


def test_accuracy(input_images, input_labels, reuse_variables):
    """

    :param input_images:
    :param input_labels:
    :param reuse_variables:
    :return:
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        logits = model.inference(images=input_images, tf_training=False)
    test_prediction = tf.nn.softmax(logits=logits)
    result = accuracy(test_prediction, input_labels)
    return result


def tower_loss(input_images, input_labels):
    """

    :param scope:
    :param images:
    :param labels:
    :return:
    """
    # with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
    logits = model.inference(images=input_images, tf_training=True)
    cross_entropy_loss = model.loss(logits, input_labels)

    return cross_entropy_loss


def average_gradients(tower_grads):
    """
    :param tower_grads:
    :return:
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # only first tower's pointer is return

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        lr = 1e-3

        # optimizer
        opt = tf.train.AdamOptimizer(lr)

        # get images and labels from function
        # images, labels = data_input.load_data(['train_data.mat', 'train_label.mat'])
        # print('Data loaded, data ', images.shape, ', label ', labels.shape)
        # image, label = tf.train.batch([images, labels], batch_size=batch_size, num_threads=2, enqueue_many=True)
        # batch_queue = tfc.slim.prefetch_queue.prefetch_queue([image, label], capacity=2*num_gpus)

        get_data = data_input.data_iterator(num_epochs, 48, param.train_filename)
        image, label = get_data.get_next()

        image_batch = tf.split(axis=0, num_or_size_splits=num_gpus, value=image)
        label_batch = tf.split(axis=0, num_or_size_splits=num_gpus, value=label)

        # calculate gradients
        tower_grads = []
        reuse_variables = tf.AUTO_REUSE

        with tf.variable_scope(tf.get_variable_scope()) as outer:
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (model.TOWER_NAME, i)) as scope:
                        # image_batch, label_batch = get_data.get_next()

                        loss = tower_loss(image_batch[i], label_batch[i])
                        tf.summary.scalar(name=scope+'_loss', tensor=loss)

                        # reuse variables for the next tower
                        tf.get_variable_scope().reuse_variables()

                        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

                        # calculate the gradients for the batch
                        grads = opt.compute_gradients(loss)

                        # add gradients to list
                        tower_grads.append(grads)

        # batch_norm_op = tf.group(*update_ops)
        avg_grads = average_gradients(tower_grads)

        # apply the gradients to adjust the shared variables
        apply_gradient_op = opt.apply_gradients(avg_grads, global_step=tf.train.get_or_create_global_step())
        train_op = tf.group(apply_gradient_op, tf.group(*batchnorm_updates))

        summary_op = tf.summary.merge_all()

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        sess.run(init)

        # summary writer
        summary_writer = tf.summary.FileWriter(logdir='log', graph=sess.graph)

        try:
            step = 0
            while True:
                start_time = time.time()
                _, loss_value, summary_str = sess.run([train_op, loss, summary_op])
                duration = time.time() - start_time

                if step % 10 == 0:
                    # summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary=summary_str, global_step=step)

                if step % 100 == 0:
                    num_examples_per_step = batch_size * num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / num_gpus

                    format_str = ('%s: step %d, loss = %.6f (%.1f examples/sec; %.3f ''sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
                step += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Final loss: {}'.format(loss_value))
        print('Number of total steps %d' % step)

        checkpoint_path = os.path.join(param.mode_dir_name, 'model.ckpt')
        saver.save(sess=sess, save_path=checkpoint_path, global_step=step)


train()
