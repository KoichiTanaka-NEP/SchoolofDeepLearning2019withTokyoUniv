# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing import image
import subprocess
from io import BytesIO
import urllib
import urllib.request as urllib

# print "aitoolkit::utilsを読み込みました．"

# utilities
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from IPython.display import Image, display
import PIL.Image

# utility function to show image
def show_img_at_path(path, fmt='jpeg'):
    img = image.load_img(path,target_size=(255,255))
    a = image.img_to_array(img)
    a = np.uint8(np.clip(a, 0, 255))
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def show_img_array(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    #f = StringIO()
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def image_to_tensor(filename, img_height, img_width, color_mode="rgb"):
    if color_mode == "rgb":
        isGrayscale = False
    elif color_mode == "grayscale":
        isGrayscale = True
        
    filename = urllib.urlopen(filename) 
    #img = image.load_img(BytesIO(filename.read()) , grayscale=isGrayscale, target_size=(img_height, img_width))
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
    # sns.plt.title('上：Loss（低いほど良い）、下：Accuracy（何%命中したか）')


def start_app():
    subprocess.check_output(["python", "/home/ec2-user/nes_application/src/server.py"])
