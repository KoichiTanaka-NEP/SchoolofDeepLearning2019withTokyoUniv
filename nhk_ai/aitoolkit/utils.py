# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing import image
import subprocess

def image_to_tensor(filename, img_height, img_width, color_mode="rgb"):
    if color_mode == "rgb":
        isGrayscale = False
    elif color_mode == "grayscale":
        isGrayscale = True
    img = image.load_img(filename, grayscale=isGrayscale, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # normalization
    x = x / 255.0
    return x

def plot_learning_history(filename):
    epoch_list = []
    val_loss_list = []
    val_acc_list = []

    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            line = line.rstrip()
            cols = line.split('\t')
            assert len(cols) == 5
            epoch = int(cols[0])
            loss = float(cols[1])
            acc = float(cols[2])
            val_loss = float(cols[3])
            val_acc = float(cols[4])
            epoch_list.append(epoch)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

    fig, [ax1,ax2] = plt.subplots(2,1,figsize=(15,9))
    ax1.plot(epoch_list, val_loss_list)
    ax2.plot(epoch_list, val_acc_list)
    plt.title('上：Loss（低いほど良い）、下：Accuracy（何%命中したか）')
