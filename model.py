import tensorflow as tf

def bottom(input, name, out_c, size=3, stride=1, padding="SAME"):

    in_c = input.get_shape().as_list()[3]

    with tf.variable_scope(name):
        filter1 = tf.Variable(tf.random_normal([size, size, in_c, out_c]))
        filter2 = tf.Variable(tf.random_normal([size, size, out_c, out_c]))
        strides = [1, stride, stride, 1]
        conv1 = tf.nn.conv2d(input, filter1, strides, padding)
        relu1 = tf.nn.relu(conv1, name=name+"relu1")
        bn1 = tf.layers.batch_normalization(relu1, name=name+"bn1")
        conv2 = tf.nn.conv2d(bn1, filter2, strides, padding)
        relu2 = tf.nn.relu(conv2, name= name + "relu2")
        bn2 = tf.layers.batch_normalization(relu2, name=name+"bn2")

        return bn2


def down(input, name, out_c, size=3, stride=1, p_stride=2, padding="SAME", do_pool=True):

    in_c = input.get_shape().as_list()[3]

    with tf.variable_scope(name):
        filter1 = tf.Variable(tf.random_normal([size, size, in_c, out_c]))
        filter2 = tf.Variable(tf.random_normal([size, size, out_c, out_c]))
        strides = [1, stride, stride, 1]
        conv1 = tf.nn.conv2d(input, filter1, strides, padding)
        relu1 = tf.nn.relu(conv1, name=name+"relu1")
        bn1 = tf.layers.batch_normalization(relu1, name=name+"bn1")
        conv2 = tf.nn.conv2d(bn1, filter2, strides, padding)
        relu2 = tf.nn.relu(conv2, name=name+"relu2")
        residual = tf.layers.batch_normalization(relu2, name=name+"bn2")

        if do_pool:
            p_strides = [1, p_stride, p_stride, 1]
            pool = tf.nn.max_pool(residual, p_strides, p_strides, padding=padding, name="name/""pool")
            return pool, residual
        else:
            return residual


def up(input, residual, name, out_c, size=3, stride=2, padding="SAME"):

    in_c = input.get_shape().as_list()[3]
    batch_size = input.get_shape().as_list()[0]
    w = input.get_shape().as_list()[1]
    h = input.get_shape().as_list()[2]

    out_c_t = int(in_c/2)

    with tf.variable_scope(name):
        filter_t = tf.Variable(tf.random_normal([2, 2, out_c_t, in_c]))
        out_shape = [batch_size, 2*w, 2*h, out_c_t]
        strides_t = [1, stride, stride, 1]
        deconv = tf.nn.conv2d_transpose(input, filter_t, out_shape, strides_t, padding="SAME")

        concat = tf.concat([deconv, residual], 3)

        filter1 = tf.Variable(tf.random_normal([size, size, in_c, out_c_t]))
        filter2 = tf.Variable(tf.random_normal([size, size, out_c_t, out_c]))
        strides = [1, 1, 1, 1]

        conv1 = tf.nn.conv2d(concat, filter1, strides, padding)

        relu1 = tf.nn.relu(conv1, name=name+"relu1")
        bn1 = tf.layers.batch_normalization(relu1, name=name+"bn1")
        conv2 = tf.nn.conv2d(bn1, filter2, strides, padding)
        relu2 = tf.nn.relu(conv2, name=name+"relu2")
        bn2 = tf.layers.batch_normalization(relu2, name=name+"bn2")

        return bn2


def make_unet(input, out_c):

    l1, res1 = down(input, "layer1", 64)

    l2, res2 = down(l1, "layer2", 128)

    l3, res3 = down(l2, "layer3", 256)

    l4, res4 = down(l3, "layer4", 512)

    l5 = bottom(l4, "layer5", 1024)

    l6 = up(l5, res4, "layer6", 512)

    l7 = up(l6, res3, "layer7", 256)

    l8 = up(l7, res2, "layer8", 128)

    l9 = up(l8, res1, "layer9", 64)

    out = tf.layers.conv2d(l9, out_c, 1, padding="SAME", activation=tf.nn.sigmoid)

    return out


def generator(x):

    with tf.variable_scope("generator"):

        return make_unet(x, 3)


def discriminator(y, reuse=False):

    with tf.variable_scope("discriminator"):

        if(reuse):
            tf.get_variable_scope().reuse_variables()

        return make_unet(y, 1)

