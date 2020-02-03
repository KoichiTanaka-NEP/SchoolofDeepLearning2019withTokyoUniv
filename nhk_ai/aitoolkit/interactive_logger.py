# -*- coding: utf-8 -*-

import numpy as np
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from vis.utils import utils
import matplotlib.cm as cm
import json
from utils import image_to_tensor
from IPython.display import display, clear_output


class InteractiveLogger(Callback):
    def __init__(self, img_height, img_width, labels, nb_epoch, steps_per_epoch, log_dirpath, test_img_list=[], color_mode='rgb', verbose=False):
        self.test_img_list = test_img_list
        self.img_height = img_height
        self.img_width = img_width
        self.labels = labels
        self.now_epoch = 1
        self.total_batch = nb_epoch * steps_per_epoch
        self.now_progress_rate = 1
        self.color_mode = color_mode
        self.verbose = verbose
        self.nb_epoch = nb_epoch
        self.log_dirpath = log_dirpath
        self.loss_history = []
        self.acc_history = []
        self.epoch_losses = []
        self.epoch_accs = []


    def on_train_begin(self, logs={}):
        pass
    

    def on_epoch_begin(self, epoch, logs={}):
        pass
    

    def on_batch_begin(self, batch, logs={}):
        pass
    

    def on_batch_end(self, batch, logs={}):
        self.now_progress_rate = self.now_progress_rate + 1
        self.loss_history.append(float(logs.get('loss')))
        self.acc_history.append(float(logs.get('acc')))
        t = int((float(self.now_progress_rate) / float(self.total_batch) ) * 100.)
        output_dict = {
            "losses": self.loss_history,
            "accs": self.acc_history,
            "epoch_losses": self.epoch_losses,
            "acc_losses": self.epoch_accs,
            "epoch": self.now_epoch,
            "progress_rate": t
        }
        with open(self.log_dirpath + "interactiveLogger.json", "w") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    def on_epoch_end(self, epoch, logs={}):
        self.now_epoch = self.now_epoch + 1
        self.loss_history = []
        self.acc_history = []
        self.epoch_losses.append(float(logs.get('loss')))
        self.epoch_accs.append(float(logs.get('acc')))


    def on_train_end(self, logs={}):
        self.now_epoch = 1
        self.now_progress_rate = 1
        self.loss_history = []
        self.acc_history = []
        output_dict = {
            "losses": [],
            "accs": [],
            "epoch": "",
            "progress_rate": ""
        }
        with open(self.log_dirpath + "interactiveLogger.json", "w") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
