import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from os.path import join


def corrupt_img(img, center_num=3, center_dist = "uniform", corrupt_dist = "gaussian", ratio = 0.5, center_spread=0.3):

    img = img/np.max(img)

    w = img.shape[0]
    h = img.shape[1]
    sigma_x = w/16
    sigma_y = h/16

    center_spread_x = center_spread * w
    center_spread_y = center_spread * h

    if center_dist == "uniform":
        centers_x = np.random.randint(0, w, (center_num,))
        centers_y = np.random.randint(0, h, (center_num,))
    else:
        centers_x = np.random.normal(int(w/2), center_spread_x, (center_num,)).astype(int)
        centers_y = np.random.normal(int(h/2), center_spread_y, (center_num,)).astype(int)

    centers_x = centers_x[:,None] * np.ones(int(w * h * ratio))
    centers_y = centers_y[:,None] * np.ones(int(w * h * ratio))

    centers_x[centers_x < 0] = 0
    centers_x[centers_x >= w] = w-1

    centers_y[centers_y < 0] = 0
    centers_y[centers_y >= h] = h - 1

    if corrupt_dist == "gaussian":
        corrpution_x = np.random.normal(centers_x, sigma_x).astype(int)
        corrpution_y = np.random.normal(centers_y, sigma_y).astype(int)
    else:
        corrpution_x = np.random.randint(int(centers_x-w/2), sigma_x, (center_num, w * h * ratio))
        corrpution_y = np.random.randint(int(centers_y-h/w), sigma_y, (center_num, w * h * ratio))

    corrpution_x[corrpution_x < 0] = 0
    corrpution_x[corrpution_x >= w] = w-1

    corrpution_y[corrpution_y < 0] = 0
    corrpution_y[corrpution_y >= h] = h-1

    corrupted_img = img

    corrupted_img[corrpution_x, corrpution_y, :] = 0
    mask = np.ones((w,h))

    mask[corrpution_x, corrpution_y] = 0

    return corrupted_img, mask

def image_generator(images_dir,
                    images,
                    batch_size,
                    img_dim):
    """
    Creates image generator.

    Args:
        images_dir: String. The path pointing to directory containing the images.
        images: List. A list of image names.
        batch_size: Integer. The number of images to use in each batch.
        img_dim: List. A list containing the shape to which original image should be resized.

    Returns:
        A tuple of arrays size batch_size. One array is for images and one for masks.
    """
    while True:

        random_indices = np.random.choice(len(images), batch_size)
        i = []
        m = []

        for index in random_indices:

            img = load_img(join(images_dir, images[index]), target_size=img_dim)
            img, mask = corrupt_img(img)

            img_array = img_to_array(img) / 255
            mask_array = img_to_array(mask) / 255

            i.append(img_array)
            m.append(mask_array[:, :, 0])

        yield np.array(i), np.array(m).reshape(-1, img_dim[1], img_dim[1], 1)


def save_img(img, name):

    encoded = tf.image.encode_jpeg(img, name="save_me")
    f = open(name, "wb+")
    f.write(encoded.eval())
    f.close()

