import tensorflow as tf
import tensorflow.contrib as tfc
import param

num_labels = param.num_labels
TOWER_NAME = param.tower_name


def _variable_on_cpu(name, shape, initializer=tf.glorot_uniform_initializer(seed=0)):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=initializer)
    return var


def batch_norm_wrapper(inputs, is_training, decay=0.999):
    epsilon = 1e-5
    # scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]), ini)
    # beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    # pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    # pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    scale = tf.get_variable(name='scale', shape=inputs.shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
    beta = tf.get_variable(name='beta', shape=inputs.shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

    pop_mean = tf.get_variable(name='pop_mean', shape=inputs.get_shape().as_list()[1:4], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0), trainable=False)
    pop_var = tf.get_variable(name='pop_var', shape=inputs.get_shape().as_list()[1:4], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.0), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)


def conv_block(logit, filter_size, num_filter, stage, block, training, s):
    """
    This is the identity block used in the ResNet. Identity block is used where input and out activation size is same.
    We are implementing 3 hidden layers

    :param logit:           Input logits from previous layers
    :param filter_size:     filter size for middle convolution layer
    :param num_filter:      number of filters for each convolution layer
    :param stage:           layer name according to their position in network
    :param block:           sub-name of the layer
    :param training:        Training = 1, Test = 0
    :param s:               stride
    :return:                +output logit
    """

    F1, F2, F3 = num_filter
    pre_filter = logit.get_shape().as_list()[3]
    shortcut = logit

    with tf.variable_scope(str(stage) + block + '2a'):
        W = _variable_on_cpu(name='W', shape=[1, 1, pre_filter, F1])
        L1 = tf.nn.conv2d(input=logit, filter=W, strides=[1, s, s, 1], padding='VALID', name='Conv')
        L1_bn = tfc.layers.batch_norm(inputs=L1, decay=0.9, is_training=training)
        # L1_bn = batch_norm_wrapper(L1, training)
        L1_act = tf.nn.relu(features=L1_bn, name='ReLU')

    with tf.variable_scope(str(stage) + block + '2b'):
        W = _variable_on_cpu(name='W', shape=[filter_size, filter_size, F1, F2])
        L2 = tf.nn.conv2d(input=L1_act, filter=W, strides=[1, 1, 1, 1], padding='SAME', name='Conv')
        L2_bn = tfc.layers.batch_norm(inputs=L2, decay=0.9, is_training=training)
        # L2_bn = batch_norm_wrapper(L2, training)
        L2_act = tf.nn.relu(features=L2_bn, name='ReLU')

    with tf.variable_scope(str(stage) + block + '2c'):
        W = _variable_on_cpu('W', shape=[1, 1, F2, F3])
        L3 = tf.nn.conv2d(input=L2_act, filter=W, strides=[1, 1, 1, 1], padding='VALID', name='Conv')
        L3_bn = tfc.layers.batch_norm(inputs=L3, decay=0.9, is_training=training)
        # L3_bn = batch_norm_wrapper(L3, training)

    with tf.variable_scope(str(stage) + block + '2d'):
        W = _variable_on_cpu('W', shape=[1, 1, pre_filter, F3])
        L4 = tf.nn.conv2d(input=shortcut, filter=W, strides=[1, s, s, 1], padding='VALID', name='Conv')
        L4_bn = tfc.layers.batch_norm(inputs=L4, decay=0.9, is_training=training)
        # L4_bn = batch_norm_wrapper(L4, training)

    out_sum = tf.add(L4_bn, L3_bn)
    output = tf.nn.relu(features=out_sum)
    return output


def identity_block(logit, filter_size, num_filter, stage, block, training):
    """
    This is the identity block used in the ResNet. Identity block is used where input and out activation size is same.
    We are implementing 3 hidden layers

    :param logit:           Input logits from previous layers
    :param filter_size:     filter size for middle convolution layer
    :param num_filter:      number of filters for each convolution layer
    :param stage:           layer name according to their position in network
    :param block:           sub-name of the layer
    :param training:
    :return:                +output logit
    """

    F1, F2, F3 = num_filter
    pre_filter = logit.get_shape().as_list()[3]
    shortcut = logit

    with tf.variable_scope(str(stage) + block + '2a'):
        W = _variable_on_cpu(name='W', shape=[1, 1, pre_filter, F1])
        L1 = tf.nn.conv2d(input=logit, filter=W, strides=[1, 1, 1, 1], padding='VALID', name='Conv')
        L1_bn = tfc.layers.batch_norm(inputs=L1, decay=0.9, is_training=training)
        # L1_bn = batch_norm_wrapper(L1, training)
        L1_act = tf.nn.relu(features=L1_bn, name='ReLU')

    with tf.variable_scope(str(stage) + block + '2b'):
        W = _variable_on_cpu(name='W', shape=[filter_size, filter_size, F1, F2])
        L2 = tf.nn.conv2d(input=L1_act, filter=W, strides=[1, 1, 1, 1], padding='SAME', name='Conv')
        L2_bn = tfc.layers.batch_norm(inputs=L2, decay=0.9, is_training=training)
        # L2_bn = batch_norm_wrapper(L2, training)
        L2_act = tf.nn.relu(features=L2_bn, name='ReLU')

    with tf.variable_scope(str(stage) + block + '2c'):
        W = _variable_on_cpu(name='W', shape=[1, 1, F2, F3])
        L3 = tf.nn.conv2d(input=L2_act, filter=W, strides=[1, 1, 1, 1], padding='VALID', name='Conv')
        L3_bn = tfc.layers.batch_norm(inputs=L3, decay=0.9, is_training=training)
        # L3_bn = batch_norm_wrapper(L3, training)

    out_sum = tf.add(shortcut, L3_bn)
    output = tf.nn.relu(features=out_sum)
    return output


def inference(images, tf_training):
    """

    :param images:
    :param tf_training:
    :param scope:
    :return:
    """

    with tf.variable_scope('Conv_1'):
        W = _variable_on_cpu(name='W', shape=[7, 7, 1, 64])
        B = _variable_on_cpu(name='B', shape=[64], initializer=tf.constant_initializer(0.0))
        L1 = tf.nn.conv2d(input=images, filter=W, strides=[1, 2, 2, 1], padding='SAME')
        L1_bn = tfc.layers.batch_norm(inputs=L1, decay=0.9, scale=True, epsilon=1e-5, is_training=tf_training)
        L1_b = tf.nn.bias_add(L1_bn, B)
        L1_act = tf.nn.relu(L1_b)
        L1_p = tf.nn.max_pool(value=L1_act, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('Block_2'):
        L2 = conv_block(L1_p, 3, [64, 64, 256], 2, 'a', tf_training, 1)
        L3 = identity_block(L2, 3, [64, 64, 256], 2, 'b', tf_training)
        L4 = identity_block(L3, 3, [64, 64, 256], 2, 'c', tf_training)

    with tf.variable_scope('Block_3'):
        L5 = conv_block(L4, 3, [128, 128, 512], 3, 'a', tf_training, 2)
        L6 = identity_block(L5, 3, [128, 128, 512], 3, 'b', tf_training)
        L7 = identity_block(L6, 3, [128, 128, 512], 3, 'c', tf_training)
        L8 = identity_block(L7, 3, [128, 128, 512], 3, 'd', tf_training)

    with tf.variable_scope('Block_4'):
        L9 = conv_block(L8, 3, [256, 256, 1024], 4, 'a', tf_training, 2)
        L1_0 = identity_block(L9, 3, [256, 256, 1024], 4, 'b', tf_training)
        L1_1 = identity_block(L1_0, 3, [256, 256, 1024], 4, 'c', tf_training)
        L1_2 = identity_block(L1_1, 3, [256, 256, 1024], 4, 'd', tf_training)
        L1_3 = identity_block(L1_2, 3, [256, 256, 1024], 4, 'e', tf_training)
        L1_4 = identity_block(L1_3, 3, [256, 256, 1024], 4, 'f', tf_training)

    with tf.variable_scope('Block_5'):
        L1_5 = conv_block(L1_4, 3, [512, 512, 2048], 5, 'a', tf_training, 2)
        L1_6 = identity_block(L1_5, 3, [512, 512, 2048], 5, 'b', tf_training)
        L1_7 = identity_block(L1_6, 3, [512, 512, 2048], 5, 'c', tf_training)

    with tf.variable_scope('Pool'):
        global_pool = tf.reduce_mean(input_tensor=L1_7, axis=(1, 2))

    with tf.variable_scope('Dense'):
        D1_f = tf.layers.flatten(global_pool)
        W = _variable_on_cpu(name='W', shape=[D1_f.shape[1], num_labels])
        D1 = tf.matmul(D1_f, W)

    return D1


def loss(logits, labels):
    """

    :param logits:
    :param labels:
    :return:
    """
    entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels), name='Loss')
    return entropy_loss
