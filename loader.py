import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt


def set_memory_growth(gpus):
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)


def load_image_dataset(data_dir):
    image_exts = ['jpeg','jpg', 'bmp', 'png']
    for image_class in os.listdir(data_dir): 
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try: 
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts: 
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e: 
                print('Issue with image {}'.format(image_path))
                os.remove(image_path)
    
    data = tf.keras.utils.image_dataset_from_directory(data_dir)
    return data


def show_image_batch(data_iterator, num_images=4, figsize=(20,20)):
    batch = data_iterator.next()
    fig, ax = plt.subplots(ncols=num_images, figsize=figsize)
    for idx, img in enumerate(batch[0][:num_images]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])
