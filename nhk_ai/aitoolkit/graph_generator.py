# -*- coding: utf-8 -*-

import numpy as np
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import seaborn as sns
from vis.utils import utils
import matplotlib.cm as cm
from vis.visualization import visualize_saliency, overlay
from vis.visualization import visualize_cam
from PIL import Image
from utils import image_to_tensor
from IPython.display import display, clear_output

class GraphGenerator(Callback):
    def __init__(self, img_height, img_width, labels, task_name, test_img_list=[], color_mode='rgb', verbose=False):
        self.test_img_list = test_img_list
        self.img_height = img_height
        self.img_width = img_width
        self.labels = labels
        self.now_epoch = 1
        self.color_mode = color_mode
        self.verbose = verbose
        self.task_name = task_name
        sns.set(style="whitegrid")

    def on_train_begin(self, logs={}):
        pass
#         self.verbose = self.params['verbose']


    def on_epoch_begin(self, epoch, logs={}):
        pass


    def on_epoch_end(self, epoch, logs={}):
        clear_output(wait = True)
        index = 1
        print('------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------------------------')
        if self.verbose is True:
            for image in self.test_img_list:
                if self.color_mode == "rgb":
                    input_tensor = image_to_tensor(image, self.img_height, self.img_width)
                elif self.color_mode == "grayscale":
                    input_tensor = image_to_tensor(image, self.img_height, self.img_width, color_mode="grayscale")
                detection = self.model.predict(input_tensor)[0]
                layer_idx = utils.find_layer_idx(self.model, 'predictions')
                test_label = image.split("/")[-2]
                filter_index = self.labels.index(test_label)
                print(filter_index)
                img1 = utils.load_img(image, target_size=(self.img_height, self.img_width))
                grads = visualize_cam(
                    self.model,
                    layer_idx,
                    filter_indices=None, 
                    seed_input=img1, 
                    backprop_modifier='guided'
                )
                print('\nIndex' + str(index))
                print(detection)
                a = np.array(detection)
                print('Estimation：' +  self.labels[a.argmax(0)])
                fig = plt.figure(figsize=(10, 5))
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.tick_params(labelbottom="off",bottom="off")
                ax1.tick_params(labelleft="off",left="off")
                ax1.set_xticklabels([]) 
                ax1.imshow(overlay(grads, img1))
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.tick_params(labelbottom="off",bottom="off")
                ax2.tick_params(labelleft="off",left="off")
                ax2.set_xticklabels([])
                ax2.imshow(Image.open(image))
                plt.show()
                sns.set(style="white", context="talk")
                f, ax1 = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
                sns.barplot(self.labels, detection, palette="PiYG", ax=ax1)
                ax1.set_ylabel("Value")
                plt.show()
                index = index + 1
        if self.now_epoch % 5 == 0 or self.now_epoch == 1:
            _index = str(self.now_epoch)
            if self.now_epoch < 10:
                _index = "0" + _index
            self.model.save("./models/"+ self.task_name+"/"+"epoch_"+ _index +".hdf5")
        self.now_epoch = self.now_epoch + 1


    def on_train_end(self, logs={}):
        self.now_epoch = 1
