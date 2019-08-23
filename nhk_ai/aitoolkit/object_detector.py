# -*- coding: utf-8 -*-

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from graph_generator import GraphGenerator
from utils import image_to_tensor
import os
import glob
import tensorflow as tf

class ObjectDetector:
    def __init__(self, train_data_dir = 'data/train', validation_data_dir = 'data/validation', result_data_dir = 'results/object_detection'):
        self.is_created_datasets = False
        self.is_created_model = False
        self.is_trained = False
        self.img_width = 200
        self.img_height = 200
        self.train_data_dir = train_data_dir
        dir_labels = glob.glob(self.train_data_dir+'/*')
        self.labels = []
        for dir_label in dir_labels:
            tmp_label = dir_label.split('/')[-1]
            self.labels.append(tmp_label)
        print("ラベル : ")
        print(self.labels)
        self.num_of_detection = len(dir_labels)
        self.validation_data_dir = validation_data_dir
        self.result_data_dir = result_data_dir
        self.model = None
        self.batch_size = 32

    def create_datasets(self, labels=None, color_mode='rgb', batch_size=32, is_whitening = False, is_augmenting = True):
        if is_augmenting:
            train_datagen = ImageDataGenerator(rescale=1.0 / 255, width_shift_range=0.25, height_shift_range=0.25, \
                                           rotation_range=45, shear_range=0.2, zoom_range=0.2, channel_shift_range=20, \
                                           horizontal_flip=True, vertical_flip=True, zca_whitening=is_whitening)
        else:
            train_datagen = ImageDataGenerator(rescale=1.0 / 255, zca_whitening=is_whitening)
        validation_datagen = ImageDataGenerator(rescale=1.0 / 255, zca_whitening=is_whitening)
        self.train_generator = train_datagen.flow_from_directory(self.train_data_dir, \
            target_size=(self.img_height, self.img_width), batch_size=batch_size, \
            class_mode='categorical', classes=labels, shuffle=True, color_mode=color_mode)
        self.validation_generator = validation_datagen.flow_from_directory(self.validation_data_dir,\
            target_size=(self.img_height, self.img_width), batch_size=batch_size, \
            class_mode='categorical', classes=labels, shuffle=True, color_mode=color_mode)
        self.is_created_datasets = True
        if labels is not None:
            self.labels = labels
        self.color_mode = color_mode
        self.batch_size = batch_size
        

    def create_model(self, sgd_lr=0.01, sgd_momentum=0.1, original_model=None, channel_num = 3, batch_norm = True, dropout = True):
        input_tensor = Input(shape=(self.img_height, self.img_width, channel_num))
        if original_model is not None:
            self.input_model = original_model
            self.img_height = original_model.input_shape[1]
            self.img_width = original_model.input_shape[2]
        else:
            self.input_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
            self.img_height = self.input_model.input_shape[1]
            self.img_width = self.input_model.input_shape[2]
        # Fine-tuning Model
        tuning_model = Sequential()
        tuning_model.add(Flatten(input_shape=self.input_model.output_shape[1:]))
        if dropout:
            tuning_model.add(Dropout(0.5))
        tuning_model.add(Dense(64, activation='relu'))
        if batch_norm:
            tuning_model.add(BatchNormalization())
        if dropout:
            tuning_model.add(Dropout(0.5))
        tuning_model.add(Dense(64, activation='relu'))
        if batch_norm:
            tuning_model.add(BatchNormalization())
        if dropout:
            tuning_model.add(Dropout(0.5))
        tuning_model.add(Dense(self.num_of_detection, activation='softmax'))
        tuning_model.summary()
        self.model = Model(inputs=self.input_model.input, outputs=tuning_model(self.input_model.output))
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=sgd_lr, momentum=sgd_momentum), metrics=['accuracy'])
        self.is_created_model = True

    def get_model_summary(self):
        if self.is_created_model is None or self.is_created_model is False:
            print("モデルをまだ作れていないよ。")
        else:
            self.model.summary()

    def fit(self, steps_per_epoch = 2000, validation_steps = 40, nb_epoch = 20, log_dir='./tb_logs', use_reduce_lr=True):
        if self.is_created_datasets is None or self.is_created_datasets is False:
            print("データセットを先に作ってね。")
        else:
            if self.is_created_model is None or self.is_created_model is False:
                print("モデルを先に作ってね。")
            else:
                self.test_img_list = []
                for label in self.labels:
                    tmp_query = self.validation_data_dir + label +'/'+ '*.jpg'
                    tmp_glob = glob.glob(tmp_query)
                    for tmp_path in tmp_glob:
                        self.test_img_list.append(tmp_path)
                # Callbacks
                graph_generator = GraphGenerator(img_height = self.img_height, img_width = self.img_width,\
                                                 labels = self.labels, test_img_list=self.test_img_list, color_mode=self.color_mode)
                tensorboard = TensorBoard(log_dir=log_dir)
                callbacks = [graph_generator,tensorboard]
                if use_reduce_lr:
                    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
                    callbacks.append(reduce_lr)
                self.history = self.model.fit_generator(self.train_generator, steps_per_epoch=steps_per_epoch/(self.batch_size), \
                                                        nb_epoch=nb_epoch, validation_data=self.validation_generator, \
                                                        validation_steps=validation_steps, callbacks=callbacks)

    def detect(self, filename):
        input_tensor = image_to_tensor(filename, self.img_height, self.img_width)
        detection = self.model.predict(input_tensor)[0]
        a = np.array(detection)
        detect_label = self.labels[a.argmax(0)]
        print(detect_label)
        print(detection)

    def output_history(self, result_file):
        result = os.path.join(self.result_data_dir, result_file)
        history = self.history
        loss = history.history['loss']
        acc = history.history['acc']
        val_loss = history.history['val_loss']
        val_acc = history.history['val_acc']
        nb_epoch = len(acc)
        with open(result, "w") as f:
            f.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
            for i in range(nb_epoch):
                f.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))

    def deactivate_layer(self, layer_num):
        layer = self.model.layers[layer_num]
        layer.trainable = False

    def dump_model_weights(self, weights_path):
        self.model.save_weights(weights_path)

    def read_model_weights(self, weights_path):
        json_string = self.model.to_json()
        self.model.load_weights(weights_path)
