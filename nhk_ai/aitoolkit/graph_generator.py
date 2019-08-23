# -*- coding: utf-8 -*-

import numpy as np
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import seaborn as sns
# from PIL import Image
from utils import image_to_tensor
from IPython.display import Image, display, clear_output

class GraphGenerator(Callback):
    def __init__(self, img_height, img_width, labels, test_img_list=[], color_mode='rgb'):
        self.test_img_list = test_img_list
        self.img_height = img_height
        self.img_width = img_width
        self.labels = labels
        self.now_epoch = 1
        self.color_mode = color_mode
        sns.set(style="whitegrid")


    def on_train_begin(self, logs={}):
        self.verbose = self.params['verbose']


    def on_epoch_begin(self, epoch, logs={}):
        pass


    def on_epoch_end(self, epoch, logs={}):
        index = 1
        clear_output(False)
        print('------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------------------------')
        for image in self.test_img_list:
            if self.color_mode == "rgb":
                input_tensor = image_to_tensor(image, self.img_height, self.img_width)
            elif self.color_mode == "grayscale":
                input_tensor = image_to_tensor(image, self.img_height, self.img_width, color_mode="grayscale")
            detection = self.model.predict(input_tensor)[0]
            print('\nIndex %d' % index)
            print(detection)
            a = np.array(detection)
            print('予測：　　%s' % self.labels[a.argmax(0)])
            display(Image(filename=image, width=300, height=300))
            sns.set(style="white", context="talk")
            f, ax1 = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
            sns.barplot(self.labels, detection, palette="PiYG", ax=ax1)
            ax1.set_ylabel("Value")
            plt.show()

            index = index + 1
        self.now_epoch = self.now_epoch + 1


    def on_train_end(self, logs={}):
        self.now_epoch = 1
