import model
import tensorflow as tf
from util import image_generator, save_img
from os import listdir

batchsize = 16
w = 512
h = 512
lr = 0.001
img_dim = [w, h]
epochs = 10

images_dir = "/home/aa3250/AML_PROJECT/carvana/test/"
train_image_list = sorted(listdir(images_dir))

corrupted_image = tf.placeholder(tf.float32, shape=(batchsize, w, h, 3))
mask = tf.placeholder(tf.float32, shape=(batchsize, w, h, 1))

fake_image = model.generator(corrupted_image)
produced_mask = model.discriminator(fake_image)

generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

with tf.name_scope("generator_loss"):
    gen_loss = tf.reduce_mean(tf.log(1-produced_mask))

with tf.name_scope("discriminator_loss"):
    dis_loss = tf.reduce_mean(mask*tf.log(1-produced_mask)) + tf.reduce_mean((1-mask)*tf.log(produced_mask))

with tf.name_scope("train_gen"):
    train_gen = tf.train.AdamOptimizer(lr).minimize(gen_loss, var_list=generator_variables)

with tf.name_scope("train_dis"):
    train_dis = tf.train.AdamOptimizer(lr).minimize(dis_loss, var_list=discriminator_variables)

with tf.name_scope("init"):
    init_op = tf.global_variables_initializer()


data_maker = image_generator(images_dir, train_image_list, batchsize, img_dim)

k = 0

with tf.Session() as sess:
    sess.run(init_op)
    print("Setup done!")

    for e in range(epochs):

        for i in range(6254):


            curr_img, curr_mask = next(data_maker)

            #forward pass
            d_tra, g_tra, fak_img, pro_msk, g_loss, d_loss = sess.run([train_dis, train_gen, fake_image, produced_mask,
                                                                       gen_loss, dis_loss],
                                                                      feed_dict={corrupted_image: curr_img,
                                                                                 mask: curr_mask})


            k += 1

            if k % 50 == 0:
                print("[%d/10] [%d/6254] G Loss: %f D Loss: %f" %(e, i, g_loss, d_loss))
                save_img(fak_img[1, :, :, :], "/home/aa3250/AML_PROJECT/novel_gan/pixel-wise-gan/samples/sample_%d" %k)

