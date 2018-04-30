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
parser.add_argument('--dstep', type=int, default=2)
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

gen_input = tf.placeholder(tf.float32, shape=(batchsize, w, h, 3))
mask = tf.placeholder(tf.float32, shape=(batchsize, w, h, 1))
disc_input = tf.placeholder(tf.float32, shape=(batchsize, w, h, 3))

fake_image = model.generator(gen_input)
no_gan_mask = model.discriminator(disc_input)
gan_mask = model.discriminator(fake_image, reuse=True)

generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

print(len(generator_variables))
print(len(discriminator_variables))
print(discriminator_variables)

with tf.name_scope("generator_loss"):
    gen_loss = -tf.reduce_mean(tf.log(gan_mask)+1e-8)

with tf.name_scope("discriminator_fake_loss"):
    dis_f_loss = -tf.reduce_mean(tf.log(1-gan_mask+1e-8))

with tf.name_scope("discriminator_real_loss"):
    dis_r_loss = -tf.reduce_mean(((1-mask)*tf.log(1-no_gan_mask+1e-8)) + (mask*tf.log(no_gan_mask+1e-8)))

with tf.name_scope("train_gen"):
    train_gen = tf.train.AdamOptimizer(lrG).minimize(gen_loss, var_list=generator_variables)

with tf.name_scope("train_f_dis"):
    train_dis_fake = tf.train.AdamOptimizer(lrD).minimize(dis_f_loss, var_list=discriminator_variables)

with tf.name_scope("train_r_dis"):
    train_dis_real = tf.train.AdamOptimizer(lrD).minimize(dis_r_loss, var_list=discriminator_variables)

with tf.name_scope("init"):
    init_op = tf.global_variables_initializer()


data_maker = image_generator(images_dir, train_image_list, batchsize, img_dim)

k = 0

with tf.Session() as sess:
    sess.run(init_op)
    print("Setup done!")

    for e in range(epochs):

        for i in range(int(100064/batchsize)):

            for s in range(opt.dstep):
                curr_img, curr_mask = next(data_maker)

                #give real image, train discriminator
                _, _, _, dr_loss = sess.run([train_dis_real, fake_image, no_gan_mask, dis_r_loss],
                                              feed_dict={disc_input: curr_img, mask: curr_mask, gen_input: curr_img})

            curr_img, curr_mask = next(data_maker)

            #train both
            _, _, fak_img, pro_msk, g_loss, df_loss = sess.run([train_gen, train_dis_fake,
                                                               fake_image, gan_mask, gen_loss, dis_f_loss],
                                                               feed_dict={disc_input: curr_img, mask: curr_mask,
                                                                         gen_input: curr_img})

            k += 1

            if k % 500 == 0:
                print("[%d/10] [%d/6254] G Loss: %f DF Loss: %f DR Loss: %f" %(e, i, g_loss, df_loss, dr_loss))
                save_img(255*fak_img[1, :, :, :], opt.saveroot + "sample_%d.jpg" %k)

