from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import os, random
import pickle as pkl
import numpy as np
import skimage
import skimage.io
import skimage.transform

from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTClass():
    def __init__(self):
        self.trainval = MNIST(data_root, train=True, transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor()
        ]), download=True)
        self.trainval = [img.numpy() for img, label in self.trainval]
        self.train = np.array(self.trainval[:-10000])
        self.val = np.array(self.trainval[-10000:])
        self.test = MNIST(data_root, train=False, transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor()
        ]))
        self.test = np.array([img.numpy()  for img, label in self.test])

def get_bsds500():
    BST_PATH = os.path.join(data_root, 'bsds500/BSR_bsds500.tgz')

    f = tarfile.open(BST_PATH)
    train_files = []
    for name in f.getnames():
        if name.startswith('BSR/BSDS500/data/images/train/'):
            train_files.append(name)

    print('Loading BSR training images')
    background_data = []
    for name in train_files:
        try:
            fp = f.extractfile(name)
            bg_img = skimage.io.imread(fp)
            background_data.append(bg_img)
        except:
            continue
    return background_data

def compose_image(digit, background, size=64):
    """Difference-blend a digit and a random patch from a background image."""
    scale = np.random.uniform(0.25, 1)
    digit = skimage.transform.rescale(digit, (scale, scale, 1), anti_aliasing=True)

    w, h, _ = background.shape
    dw, dh, _ = digit.shape
    x = np.random.randint(0, w - size)
    y = np.random.randint(0, h - size)
    bg = background[x:x+size, y:y+size]

    w, h, _ = bg.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)

    bg[x:x+dw, y:y+dh, :] = np.abs(bg[x:x+dw, y:y+dh, :] - digit).astype(np.uint8)
    # make 32x32
    #bg = skimage.transform.rescale(bg, (0.5, 0.5, 1), anti_aliasing=True)
    (x1, y1), (x2, y2) = (x / 2, y / 2), ((x + dw) / 2, (y + dh) / 2)
    return bg, (x1, y1), (x2, y2)


def mnist_to_img(x):
    """Binarize MNIST digit and convert to RGB."""
    x = (x > 0).astype(np.float32)
    d = x.reshape([28, 28, 1]) * 255
    return np.concatenate([d, d, d], 2)


def create_mnistm(X):
    """
    Give an array of MNIST digits, blend random background patches to
    build the MNIST-M dataset as described in
    http://jmlr.org/papers/volume17/15-239/15-239.pdf
    """
    global rand

    X_ = np.zeros([X.shape[0], 3, 64, 64], np.uint8)
    y = np.zeros([X.shape[0], 4])
    for i in range(X.shape[0]):

        if i % 1000 == 0:
            print('Processing example', i)

        bg_img = random.sample(background_data, 1)[0].copy()

        d = mnist_to_img(X[i])
        d, pos, size = compose_image(d, bg_img)
        d = np.transpose(d, (2, 0, 1))
        X_[i] = d
        y[i] = np.array([*pos, *size])

    return X_, y


if __name__ == "__main__":
    filename = {"black": "mnist.pkl", "bsds500": "mnistm.pkl"}
    data_root = os.environ["DATA_ROOT"]
    print(f"Data root at {data_root}")
    random.seed(0)

    for bg_style in ["bsds500", "black"]:
        print(f"Generating images with {bg_style} backgrounds...")
        mnist = MNISTClass()
        background_data = get_bsds500()
        if bg_style == "black":
            background_data = [np.zeros_like(img) for img in background_data]
        print(len(background_data), background_data[0].shape)

        print('Building train set...')
        train, trainlabels = create_mnistm(mnist.train)
        print('Building validation set...')
        val, vallabels = create_mnistm(mnist.val)
        print('Building test set...')
        test, testlabels = create_mnistm(mnist.test)

        # Save dataset as pickle
        os.makedirs(os.path.join(data_root, "mnist_m_r"), exist_ok=True)
        with open(os.path.join(data_root, "mnist_m_r", filename[bg_style]), 'wb') as f:
            pkl.dump({
                'train': train, 'val': val, 'trainval': np.concatenate([train, val]), 'test': test,
                'trainlabels': trainlabels, 'vallabels': vallabels, 'trainvallabels': np.concatenate([trainlabels, vallabels]), 'testlabels': testlabels
                }, f, pkl.HIGHEST_PROTOCOL)
