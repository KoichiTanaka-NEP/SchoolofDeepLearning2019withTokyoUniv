# -*- coding: utf-8 -*-

import numpy as np
import urllib
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from graph_generator import GraphGenerator
from interactive_logger import InteractiveLogger
from keras.models import load_model
from IPython.display import Image, display, clear_output
import PIL.Image
import seaborn as sns
import matplotlib.pyplot as plt
import vis.utils.utils as vis_utils
import matplotlib.cm as cm
from vis.visualization import visualize_saliency, overlay
from vis.visualization import visualize_cam
from vis.visualization import visualize_activation
from vis.visualization import get_num_filters

from keras import activations

from utils import image_to_tensor, show_img_array
import os
import glob


# print "aitoolkit::ObjectDetectorを読み込みました．"


class ObjectDetector:
    def __init__(self, train_data_dir = 'data/train', validation_data_dir = 'data/validation', result_data_dir = 'results/object_detection', task_name="task1"):
        self.is_created_datasets = False
        self.is_created_model = False
        self.is_trained = False
        self.img_width = 150
        self.img_height = 150
        self.train_data_dir = train_data_dir
        dir_labels = glob.glob(self.train_data_dir+'/*')
        if len(dir_labels) == 0:
            raise ValueError('train_data_dirのパスが間違っています．')
        if len(glob.glob(validation_data_dir+'/*')) == 0:
            raise ValueError('validation_data_dirのパスが間違っています．')
        if len(glob.glob(result_data_dir+'/*')) == 0:
            raise ValueError('result_data_dirのパスが間違っています．')
        self.labels = []
        for dir_label in dir_labels:
            tmp_label = dir_label.split('/')[-1]
            self.labels.append(tmp_label)
        self.labels.sort()
        # print "ラベル : "
        # print self.labels
        self.num_of_detection = len(dir_labels)
        self.validation_data_dir = validation_data_dir
        self.result_data_dir = result_data_dir
        self.model = None
        self.nb_epoch = 0
        self.task_name = task_name
        if os.path.exists('./models/'+self.task_name+"/"):
            pass
        else:
            os.mkdir('./models/'+self.task_name+"/")


    def create_datasets(self, labels=None, color_mode='rgb', batch_size=32, is_whitening = False, is_augmenting = True):
        if is_augmenting:
            train_datagen = ImageDataGenerator(rescale=1.0 / 255, width_shift_range=0.25, height_shift_range=0.25, \
                                           rotation_range=45, shear_range=0.2, zoom_range=0.2, channel_shift_range=20, \
                                           horizontal_flip=True, vertical_flip=True, zca_whitening=is_whitening)
        else:
            train_datagen = ImageDataGenerator(rescale=1.0 / 255, zca_whitening=is_whitening)
        validation_datagen = ImageDataGenerator(rescale=1.0 / 255, zca_whitening=is_whitening)
        if is_whitening is True:
            ## Warning : This function has not been implemented yet!!
            train_datagen.fit()
            validation_datagen.fit()
        else:
            pass
        ######隠しディレクトリを消去（現状）######
        hidden_dirs = glob.glob(self.train_data_dir + ".*/")
        for h_dir in hidden_dirs:
            os.rmdir(h_dir)
        hidden_dirs = glob.glob(self.validation_data_dir + ".*/")
        for h_dir in hidden_dirs:
            os.rmdir(h_dir)
        ##############################
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
        print("ラベル:")
        print(self.labels)


    def create_model(self, original_model=None, channel_num = 3, batch_norm = True, dropout = True):
        input_tensor = Input(shape=(self.img_height, self.img_width, channel_num))
        if original_model is not None:
            self.input_model = original_model
            self.img_height = original_model.input_shape[1]
            self.img_width = original_model.input_shape[2]
        else:
            self.input_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
        # Fine-tuning Model
        x = Flatten()(self.input_model.output)
        if dropout:
            x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        if batch_norm:
            x = BatchNormalization()(x)
        if dropout:
            x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        if batch_norm:
            x = BatchNormalization()(x)
        if dropout:
            x = Dropout(0.5)(x)
        y = Dense(self.num_of_detection, activation='softmax', name='predictions')(x)
        # setup model and compile
        self.model = Model(inputs=self.input_model.input, outputs=y)
        self.is_created_model = True
        print("create_modelメソッドが完了しました．")

    def compile_model(self, sgd_lr=0.01, sgd_momentum=0.1):
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=sgd_lr, momentum=sgd_momentum),
                   metrics=['accuracy'])
        self.is_model_compiled = True
        print("compile_modelメソッドが完了しました．")


    def get_model_summary(self):
        if self.is_created_model is None or self.is_created_model is False:
            print("モデルをまだ作れていないよ。")
        else:
            self.model.summary()


    def fit(self, nb_epoch = 20, batch_size=32, use_reduce_lr=True, verbose=False, steps_per_epoch = 100):
        self.nb_epoch = nb_epoch
        if self.is_created_datasets is None or self.is_created_datasets is False:
            print("データセットを先に作ってね。")
        else:
            if self.is_created_model is None or self.is_created_model is False:
                print("モデルを先に作ってね。")
            else:
                self.test_img_list = []

                nb_train_samples = len(glob.glob(self.train_data_dir + '/*/*.jpg'))
                nb_validation_samples = len(glob.glob(self.validation_data_dir + '/*/*.jpg'))
                

                steps_per_epoch = steps_per_epoch
                validation_steps = int(nb_validation_samples/batch_size)

                print("Validation Sample Number : " +  str(nb_validation_samples))
                for label in self.labels:
                    tmp_query = self.validation_data_dir + label +'/'+ '*.jpg'
                    tmp_glob = glob.glob(tmp_query)
                    self.test_img_list.append(tmp_glob[0])


                # Callbacks
                graph_generator = GraphGenerator(img_height = self.img_height, img_width = self.img_width,\
                                                 labels = self.labels, test_img_list=self.test_img_list, color_mode=self.color_mode, verbose=verbose, task_name = self.task_name)
                interactive_logger = InteractiveLogger(img_height = self.img_height, img_width = self.img_width, \
                                                 labels = self.labels, nb_epoch = nb_epoch, steps_per_epoch=steps_per_epoch, \
                                                 log_dirpath = "./logs/", test_img_list=self.test_img_list, color_mode=self.color_mode, verbose=verbose)
                model_checkpoint = ModelCheckpoint("./models/"+ self.task_name+"/"+"epoch_{epoch:02d}.hdf5", monitor='loss')
                callbacks = [graph_generator, interactive_logger]
                if use_reduce_lr:
                    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
                    callbacks.append(reduce_lr)
                # Run
                print()
                self.history = self.model.fit_generator(self.train_generator,steps_per_epoch=steps_per_epoch, \
                                                        epochs=nb_epoch, validation_data=self.validation_generator, \
                                                         validation_steps=validation_steps, \
                                                        callbacks=callbacks)
                print("Training Finished!! ", "Total Epoch :::" + str(nb_epoch))


    def detect(self, filename, answer_text, verbose=False, layer_index = -1, mode="local"):
        if mode == "local":
            input_tensor = image_to_tensor(filename, self.img_height, self.img_width)
        else:
#             if True:
            try:
                input_tensor = image_to_tensor(filename, self.img_height, self.img_width)
            except:
                raise ValueError("This URL is not supported!! Use other one.")
        detection = self.model.predict(input_tensor)[0]
        a = np.array(detection)
        detect_label = self.labels[a.argmax(0)]
        if verbose is True:
            print("結果 .... " + str(answer_text[detect_label]))
            img1 = vis_utils.load_img(filename, target_size=(self.img_height, self.img_width))
            # Swap softmax with linear
            layer_idx = vis_utils.find_layer_idx(self.model, 'predictions')
            self.model.layers[layer_idx].activation = activations.linear
            vis_model = vis_utils.apply_modifications(self.model)
            filter_index = a.argmax(0)
            grads = visualize_cam(
                vis_model,
                layer_idx,
                filter_index, #クラス番号
                img1[:, :, :],
                backprop_modifier='guided'
            )
            a = np.array(detection)
            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.tick_params(labelbottom="off",bottom="off")
            ax1.grid(False)
            ax1.tick_params(labelleft="off",left=False)
            plt.yticks(color="None")
            ax1.set_xticklabels([])
            ax1.imshow(img1)
            ax1.imshow(grads,cmap='jet', alpha = 0.6)
            ax1.set_title("Heat Map")
            sns.set(style="white", context="talk")
            f, ax1 = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
            sns.barplot(self.labels, detection, palette="PiYG", ax=ax1)
            ax1.set_ylabel("Value")
            plt.tick_params(length=0)
            plt.grid(False)
            plt.show()
        else:
            print(detect_label)
            print(detection)
        print("detectメソッドが完了しました．")

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

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, epoch_num=0, silent=False):
        if epoch_num < 10:
            epoch_num = "0" + str(epoch_num)
        else:
            epoch_num = str(epoch_num)
        self.model = load_model("./models/"+self.task_name+"/"+"epoch_"+epoch_num+".hdf5")

        if silent is False:
            print("load_modelメソッドが完了しました．")


    def get_layer_filters(self, layer_idx, filters):
        vis_images = []
        vert_img = ""
        hori_img = ""
        for idx in filters:
            img = visualize_activation(self.model, layer_idx, filter_indices=idx)
            if hori_img == "":
                hori_img = img
            else:
                hori_img = np.concatenate((hori_img, img), axis = 1)
            if (idx + 1) % 3 == 0:
                if vert_img == "":
                    vert_img = hori_img
                else:
                    vert_img = np.concatenate((vert_img, hori_img), axis = 0)
                hori_img = ""
        return vert_img


    def show_layer_activation(self, layer_name, filter_indices):
        layer_idx = vis_utils.find_layer_idx(self.model, layer_name)

        self.model.layers[layer_idx].activation = activations.linear
        vis_model = vis_utils.apply_modifications(self.model)

        img = visualize_activation(vis_model, layer_idx, filter_indices=filter_indices)
        show_img_array(img)


    def show_filters_evolution(self, num_epoch, conv_layer_idxs = [3, 6, 9, 13, 17]):
        fig,axes = plt.subplots(nrows=5,ncols=5,figsize=(18,18))
        two_ep = int(num_epoch/4)
        thr_ep = int(num_epoch/2)
        four_ep = int(num_epoch*3/4)
        model_epochs = [1, two_ep, thr_ep, four_ep, num_epoch]
        for idx, epoch in enumerate(model_epochs):
            self.load_model(epoch, silent=True)
            for jdx, layer_idx in enumerate(conv_layer_idxs):
                filter_img = self.get_layer_filters(layer_idx, range(9))
                axes[idx,jdx].axis('off')
                axes[idx,jdx].imshow(filter_img)
                axes[idx,jdx].set_title(str(epoch)+"Epoch   "+ str(layer_idx) +"Layer")
        plt.show()

    def show_attention_evolution(self, num_epoch, class_name, image_path):
        if class_name not in self.labels:
            print("ラベル  が間違っているよ" + str(class_name))

        class_index = self.labels.index(class_name)

        two_ep = int(num_epoch/4)
        thr_ep = int(num_epoch/2)
        four_ep = int(num_epoch*3/4)
        model_epochs = [1, two_ep, thr_ep, four_ep, num_epoch]
        for idx, epoch in enumerate(model_epochs):

            print("epoch" + str(epoch))
            self.load_model(epoch, silent=True)
            layer_idx = vis_utils.find_layer_idx(self.model, 'predictions')

            # Swap softmax with linear
            self.model.layers[layer_idx].activation = activations.linear
            vis_model = vis_utils.apply_modifications(self.model)

            x = vis_utils.load_img(image_path, target_size=(self.img_width, self.img_height))
            grads = visualize_cam(vis_model, layer_idx, filter_indices=class_index,
                              seed_input=x, backprop_modifier="guided")
            show_img_array(grads)
            show_img_array(np.squeeze(x))
        
