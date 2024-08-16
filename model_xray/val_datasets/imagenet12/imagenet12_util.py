import sys, os, time
from pathlib import Path
from glob import glob

# # Select GPU
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import tensorflow as tf
# import cv2


def humansize(nbytes):
    '''From https://stackoverflow.com/questions/14996453/python-libraries-to-calculate-human-readable-filesize-from-bytes'''
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

ds_path = '/mnt/exdisk1/danigil/datasets/imagenet'

path_imagenet_val_dataset = Path(os.path.join(ds_path, "processed")) # path/to/data/
dir_images = Path(os.path.join(ds_path, "ILSVRC2012_img_val")) # path/to/images/directory
path_labels = Path(os.path.join(ds_path, "ILSVRC2012_validation_ground_truth.txt"))
path_synset_words = Path(os.path.join(ds_path, "synset_words.txt"))
path_meta = Path(os.path.join(ds_path, "meta.mat"))

image_paths = sorted(glob(str(dir_images/"*")))
n_images = len(image_paths)


def load_images(image_paths, returned_shard, n_shards=5, final_im_size=240, resize_size=256):
    """ loads images into memory. It only load and returns images of the 'returned_shard'.
        image_paths: a list of paths to images
        n_shards: number of shards to loaded images be divided.
        returned_shard: the part of images to be returned. 0 <= returned_shard < n_shards
    """
    assert 0 <= returned_shard < n_shards, "The argument returned_shard must be between 0 and n_shards"
    
    shard_size = len(image_paths) // n_shards
    sharded_image_paths = image_paths[returned_shard*shard_size:(returned_shard+1)*shard_size] if returned_shard < n_shards - 1 \
                     else image_paths[returned_shard*shard_size:]
    
    
    images_list = np.zeros((len(sharded_image_paths), final_im_size, final_im_size, 3), dtype=np.uint8)
    
    for i, image_path in enumerate(sharded_image_paths):
        
        # Load (in BGR channel order)
        image = cv2.imread(image_path)
        
        
        # Resize
        height, width, _ = image.shape
        new_height = height * resize_size // min(image.shape[:2])
        new_width = width * resize_size // min(image.shape[:2])
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Crop
        height, width, _ = image.shape
        startx = width//2 - (final_im_size//2)
        starty = height//2 - (final_im_size//2)
        image = image[starty:starty+final_im_size,startx:startx+final_im_size]
        assert image.shape[0] == final_im_size and image.shape[1] == final_im_size, (image.shape, height, width)
        

        images_list[i, ...] = image[..., ::-1]
        
    return images_list

n_shards = 20

def create_imagenet_preprocessed_xs(final_im_size, n_shards=20):
    print(f"Loading images of size {final_im_size}x{final_im_size}")

    for i in range(n_shards):
        
        images = load_images(image_paths, returned_shard=i, n_shards=n_shards, final_im_size=final_im_size, resize_size=final_im_size)
        
        if i == 0:
            print(f"Total memory allocated for loading images ({final_im_size}):", humansize(images.nbytes))
        
        np.save(str(path_imagenet_val_dataset / "x_val_{}.npy".format(i+1)), images)
        
        # if (i + 1) * 100 / n_shards % 5 == 0:
        #     print("{:.0f}% Completed.".format((i+1)/n_shards*100))
        
        images = None

    print("Finished loading images")

def xs_cleanup():
    for x_path in glob(str(path_imagenet_val_dataset / "x_val_*.npy")):
        os.remove(x_path)

def create_imagenet_preprocessed_y():
    meta = scipy.io.loadmat(str(path_meta))
    original_idx_to_synset = {}
    synset_to_name = {}

    for i in range(1000):
        ilsvrc2012_id = int(meta["synsets"][i,0][0][0][0])
        synset = meta["synsets"][i,0][1][0]
        name = meta["synsets"][i,0][2][0]
        original_idx_to_synset[ilsvrc2012_id] = synset
        synset_to_name[synset] = name

    synset_to_keras_idx = {}
    keras_idx_to_name = {}
    with open(str(path_synset_words), "r") as f:
        for idx, line in enumerate(f):
            parts = line.split(" ")
            synset_to_keras_idx[parts[0]] = idx
            keras_idx_to_name[idx] = " ".join(parts[1:])

    convert_original_idx_to_keras_idx = lambda idx: synset_to_keras_idx[original_idx_to_synset[idx]]

    with open(str(path_labels),"r") as f:
        y_val = f.read().strip().split("\n")
        y_val = np.array([convert_original_idx_to_keras_idx(int(idx)) for idx in y_val])

    return y_val

    # np.save(str(path_imagenet_val_dataset/"y_val.npy"), y_val)


def ret_imagnet12_tf_ds(image_size = (256,256)):
    y_val = create_imagenet_preprocessed_y()

    ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=dir_images,
        labels=list(y_val),
        label_mode='int',

        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=image_size,
        shuffle=False,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )

    return ds