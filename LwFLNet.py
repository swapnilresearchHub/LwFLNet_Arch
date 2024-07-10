#!/usr/bin/env python
# coding: utf-8

# The below code is for the LightWeight Face liveness Network (LwFLNet) that is proposed for generalization of 2D and 3D face spoofing attacks.
# Kindly Download the datasets from the links provided in the Readme file and then proceed with use of this code

import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Dense, Flatten,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.inputs.keras import PlotLossesCallback
from keras.models import Sequential
from keras.regularizers import l2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.layers import Input, Dense, concatenate,Add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

# Model A
common_inp = Input(shape=(64,64,3), name="Input_a")
al_1 = Conv2D(32, kernel_size=(3,3),activation = "relu")(common_inp)
al_2 = MaxPooling2D(2, 2)(al_1)
al_3 = Conv2D(64, kernel_size=(3,3), activation = "relu")(al_2)
al_4 = MaxPooling2D(2, 2)(al_3)
al_5 = Conv2D(128, kernel_size=(3,3), activation = "relu")(al_4)
al_6 = MaxPooling2D(2, 2)(al_5)
al_7 = Flatten()(al_6)
al_8 = Dense(128, activation="relu")(al_7)
al_9 = Dropout(0.4)(al_8)
al_10 = Dense(256, activation="relu")(al_9)
al_11 = Dropout(0.2)(al_10)
# al_12 = Dense(2, activation="sigmoid",name ="a_output_layer")(al_11)

#Model B
b_ip_img = Input(shape=(64,64,3), name="Input_b")
bl_1 = Conv2D(32, kernel_size=(3,3), activation = "relu")(common_inp)
bl_2 = MaxPooling2D(2, 2)(bl_1)
bl_3 = Conv2D(64, kernel_size=(3,3), activation = "relu")(bl_2)
bl_4 = MaxPooling2D(2, 2)(bl_3)
bl_5 = Flatten()(bl_4)
bl_6 = Dense(128, activation="relu")(bl_5)
bl_7 = Dropout(0.4)(bl_6)
bl_8 = Dense(256, activation="relu")(bl_7)
# bl_8 = Dense(2, activation = "sigmoid",name ="b_output_layer")(bl_7)

#Merging model A and B
a_b = Add()([al_11,bl_8])

#Final Layer
output_layer = Dense(2, activation = "softmax", name = "output_layer")(a_b)

#Model Definition 
merged_new_model = Model(inputs=common_inp,outputs=output_layer, name = "merged_model")

#Model Details
# keras.utils.plot_model(merged_new_model, "output/architecture.png", show_shapes=True)


merged_new_model.summary()

from tensorflow.keras.utils import plot_model
plot_model(merged_new_model, to_file='Combined_CNN_architecturenew.jpg', show_shapes=False)


#training_dir = '..\\Database_Replay\\Train\\'
#validation_dir = '..\\Database_Replay\\Val\\' 
#test_dir='..\\Database_Replay\\Test\\'

# Cross Dataset validation sets
#training_dir = '..\\3dmaddataset\\train\\'
#validation_dir= '..\\NUAADataset\\Dataset\\val'
#test_dir='..\\3dmaddataset\\test\\'


training_dir = '..\\NUAADataset\\Dataset\\train'
test_dir = '..\\NUAADataset\\Dataset\\test'
validation_dir='..\\NUAADataset\\Dataset\\val'


class_subset=['Fake','Real']

training_datagen = ImageDataGenerator(
                                      width_shift_range=0.5,
                                      height_shift_range=0.5,
                                      horizontal_flip=True,
                                      
                                    )
test_generator = ImageDataGenerator(
                                      width_shift_range=0.5,
                                      height_shift_range=0.5,
                                      
                                      )

training_data = training_datagen.flow_from_directory(training_dir, 
                                      target_size=(64, 64), 
                                      batch_size=32,
                                      class_mode='binary',
                                      classes=class_subset,
                                      shuffle=True,
                                      seed=42) 
testgen = test_generator.flow_from_directory(test_dir,
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary',
                                             shuffle=False,
                                             seed=42)

valid_data = training_datagen.flow_from_directory(validation_dir, 
                                      target_size=(64, 64), 
                                      batch_size=32,
                                      class_mode='binary',
                                      shuffle=True,
                                      classes=class_subset,
                                      seed=42
                                      ) 




#pip install livelossplot
plot_loss_1 = PlotLossesCallback()

tl_checkpoint_1 = ModelCheckpoint(filepath='NUAA_RGB_CNN_CombinedModelEpoch30.weights.best.hdf5',
                                  save_best_only=True,
                                  verbose=1)
# EarlyStopping
early_stop = EarlyStopping(monitor='loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')

merged_new_model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])

merged_new_model.fit(training_data,
                          epochs=30, 
                          verbose=1, 
                          validation_data= valid_data,
                          callbacks=[tl_checkpoint_1,early_stop,plot_loss_1])


merged_new_model.load_weights('NUAA_RGB_CNN_CombinedModelEpoch30.weights.best.hdf5') # initialize the best trained weights


true_classes = testgen.classes
class_indices = training_data.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())
print('test class',len(true_classes))
vgg_preds = merged_new_model.predict(testgen)
vgg_pred_classes = np.argmax(vgg_preds, axis=1)
print('pred_classes',len(vgg_pred_classes))

##Parameter Estimation


vgg_acc_Epoch = accuracy_score(true_classes, vgg_pred_classes)
print("Resnet50 Accuracy without Fine-Tuning: {:.2f}%".format(vgg_acc_Epoch * 100))


cm=confusion_matrix(true_classes,vgg_pred_classes)
print(cm)
[[tn, fp], [fn, tp]] = cm

#Attack Presentation classification error rate
APCER = fp/ (fp+tn) *100

#Bonafide Presentation classification error rate
BPCER = fn/(fn+tp) *100

#half total error rate
HTER = (APCER+BPCER) / 2 


print(f'APCER obtianed is {APCER}, BPCER obtained is {BPCER} and Half Total Error rate is {HTER}')

#Cross dataset validation

test_dirreplay='D:\\FacePapers2021\\Liveliness Detection\\Database_Replay\\Test\\'
testgen1 = test_generator.flow_from_directory(test_dirreplay,
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary',
                                             shuffle=False,
                                             seed=42)

import pandas as pd
import numpy as np

merged_new_model.load_weights('NUAA_RGB_CNN_CombinedModelEpoch30.weights.best.hdf5') # initialize the best trained weights


true_classes = testgen1.classes
class_indices = training_data.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())
print('test class',len(true_classes))
vgg_preds = merged_new_model.predict(testgen1)
vgg_pred_classes = np.argmax(vgg_preds, axis=1)
print('pred_classes',len(vgg_pred_classes))

##Parameter Estimation


vgg_acc_Epoch = accuracy_score(true_classes, vgg_pred_classes)
print("Resnet50 Accuracy without Fine-Tuning: {:.2f}%".format(vgg_acc_Epoch * 100))


cm=confusion_matrix(true_classes,vgg_pred_classes)
print(cm)
[[tn, fp], [fn, tp]] = cm

#Attack Presentation classification error rate
APCER = fp/ (fp+tn) *100

#Bonafide Presentation classification error rate
BPCER = fn/(fn+tp) *100

#half total error rate
HTER = (APCER+BPCER) / 2 


print(f'APCER obtianed is {APCER}, BPCER obtained is {BPCER} and Half Total Error rate is {HTER}')



#Cross dataset validation
test_dir3d = 'C:\\Users\\Convergytics\\3dmaddataset\\test\\'


testgen2 = test_generator.flow_from_directory(test_dir3d,
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary',
                                             shuffle=False,
                                             seed=42)


merged_new_model.load_weights('NUAA_RGB_CNN_CombinedModelEpoch30.weights.best.hdf5') # initialize the best trained weights


true_classes = testgen2.classes
class_indices = training_data.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())
print('test class',len(true_classes))
vgg_preds = merged_new_model.predict(testgen2)
vgg_pred_classes = np.argmax(vgg_preds, axis=1)
print('pred_classes',len(vgg_pred_classes))

##Parameter Estimation


vgg_acc_Epoch = accuracy_score(true_classes, vgg_pred_classes)
print("Resnet50 Accuracy without Fine-Tuning: {:.2f}%".format(vgg_acc_Epoch * 100))


cm=confusion_matrix(true_classes,vgg_pred_classes)
print(cm)
[[tn, fp], [fn, tp]] = cm

#Attack Presentation classification error rate
APCER = fp/ (fp+tn) *100

#Bonafide Presentation classification error rate
BPCER = fn/(fn+tp) *100

#half total error rate
HTER = (APCER+BPCER) / 2 


print(f'APCER obtianed is {APCER}, BPCER obtained is {BPCER} and Half Total Error rate is {HTER}')

