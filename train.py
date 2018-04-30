import model
import tensorflow as tf
from util import image_generator, save_img
from os import listdir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, required=True)
parser.add_argument('--saveroot', type=str, required=True)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--img_width', type=int, default=256)
parser.add_argument('--img_height', type=int, default=256)
parser.add_argument('--lrD', type=float, default=0.001)
parser.add_argument('--lrG', type=float, default=0.001)
parser.add_argument('--epc', type=int, default=10)

opt = parser.parse_args()

batchsize = opt.batchsize
w = opt.img_width
h = opt.img_height
lrD = opt.lrD
lrG = opt.lrG
img_dim = [w, h]
epochs = opt.epc

images_dir = opt.dataroot
train_image_list = sorted(listdir(images_dir))

corrupted_image = tf.placeholder(tf.float32, shape=(batchsize, w, h, 3))
mask = tf.placeholder(tf.float32, shape=(batchsize, w, h, 1))

fake_image = model.generator(corrupted_image)
produced_mask = model.discriminator(fake_image)

generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

with tf.name_scope("generator_loss"):
    gen_loss = tf.reduce_mean(tf.log(1-produced_mask+1e-8))

with tf.name_scope("discriminator_loss"):
    dis_loss = tf.reduce_mean(mask*tf.log(1-produced_mask+1e-8)) + tf.reduce_mean((1-mask)*tf.log(produced_mask+1e-8))

with tf.name_scope("train_gen"):
    train_gen = tf.train.AdamOptimizer(lrG).minimize(gen_loss, var_list=generator_variables)

with tf.name_scope("train_dis"):
    train_dis = tf.train.AdamOptimizer(lrD).minimize(dis_loss, var_list=discriminator_variables)

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

            d_tra, g_tra, fak_img, pro_msk, g_loss, d_loss = sess.run([train_dis, train_gen, fake_image, produced_mask,
                                                                       gen_loss, dis_loss],
                                                                      feed_dict={corrupted_image: curr_img,
                                                                                 mask: curr_mask})

            k += 1

            if k % 50 == 0:
                print("[%d/10] [%d/6254] G Loss: %f D Loss: %f" %(e, i, g_loss, d_loss))
                save_img(fak_img[1, :, :, :], opt.saveroot + "sample_%d" %k)

