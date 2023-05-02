# -*- coding: utf-8 -*-

import os
import zipfile
import random
import shutil
from shutil import copyfile
from os import getcwd

#Download Dataset dari Google Drive
!wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Asnswg3iJOemPg0Mb5x_Z5T3ASXz****' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Asnswg3iJOemPg0Mb5x_Z5T3ASXz****" \
    -O mendeley-parasite-dataset.zip && rm -rf /tmp/cookies.txt

local_zip = 'mendeley-parasite-dataset.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

#Baca Isi Dataset dari Google Drive
print(len(os.listdir('/tmp/Mendeley Parasite Dataset/Babesia/')))
print(len(os.listdir('/tmp/Mendeley Parasite Dataset/Leishmania/')))
print(len(os.listdir('/tmp/Mendeley Parasite Dataset/Plasmodium/')))
print(len(os.listdir('/tmp/Mendeley Parasite Dataset/Toxoplasma/')))
print(len(os.listdir('/tmp/Mendeley Parasite Dataset/Trichomonad/')))
print(len(os.listdir('/tmp/Mendeley Parasite Dataset/Trypanosome/')))

#Membuat Directory Train-Test
try:
    os.mkdir('/tmp/mendeley-parasite-dataset/')

    os.mkdir('/tmp/mendeley-parasite-dataset/training/')
    os.mkdir('/tmp/mendeley-parasite-dataset/testing/')

    os.mkdir('/tmp/mendeley-parasite-dataset/training/Babesia/')
    os.mkdir('/tmp/mendeley-parasite-dataset/training/Leishmania/')
    os.mkdir('/tmp/mendeley-parasite-dataset/training/Plasmodium/')
    os.mkdir('/tmp/mendeley-parasite-dataset/training/Toxoplasma/')
    os.mkdir('/tmp/mendeley-parasite-dataset/training/Trichomonad/')
    os.mkdir('/tmp/mendeley-parasite-dataset/training/Trypanosome/')

    os.mkdir('/tmp/mendeley-parasite-dataset/testing/Babesia/')
    os.mkdir('/tmp/mendeley-parasite-dataset/testing/Leishmania/')
    os.mkdir('/tmp/mendeley-parasite-dataset/testing/Plasmodium/')
    os.mkdir('/tmp/mendeley-parasite-dataset/testing/Toxoplasma/')
    os.mkdir('/tmp/mendeley-parasite-dataset/testing/Trichomonad/')
    os.mkdir('/tmp/mendeley-parasite-dataset/testing/Trypanosome/')

except OSError:
    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
  all_files = []
    
  for file_name in os.listdir(SOURCE):
      file_path = SOURCE + file_name

      if os.path.getsize(file_path):
            all_files.append(file_name)
      else:
          print('{} is zero length, so ignoring'.format(file_name))
    
  n_files = len(all_files)
  split_point = int(n_files * SPLIT_SIZE)
    
  shuffled = random.sample(all_files, n_files)
    
  train_set = shuffled[:split_point]
  test_set = shuffled[split_point:]
    
  for file_name in train_set:
        copyfile(SOURCE + file_name, TRAINING + file_name)
        
  for file_name in test_set:
        copyfile(SOURCE + file_name, TESTING + file_name)

BABESIA_SOURCE_DIR = "/tmp/Mendeley Parasite Dataset/Babesia/"
TRAINING_BABESIA_DIR = "/tmp/mendeley-parasite-dataset/training/Babesia/"
TEST_BABESIA_DIR = "/tmp/mendeley-parasite-dataset/testing/Babesia/"

LEISHMANIA_SOURCE_DIR = "/tmp/Mendeley Parasite Dataset/Leishmania/"
TRAINING_LEISHMANIA_DIR = "/tmp/mendeley-parasite-dataset/training/Leishmania/"
TEST_LEISHMANIA_DIR = "/tmp/mendeley-parasite-dataset/testing/Leishmania/" 

PLASMODIUM_SOURCE_DIR = "/tmp/Mendeley Parasite Dataset/Plasmodium/"
TRAINING_PLASMODIUM_DIR = "/tmp/mendeley-parasite-dataset/training/Plasmodium/"
TEST_PLASMODIUM_DIR = "/tmp/mendeley-parasite-dataset/testing/Plasmodium/"

TOXOPLASMA_SOURCE_DIR = "/tmp/Mendeley Parasite Dataset/Toxoplasma/"
TRAINING_TOXOPLASMA_DIR = "/tmp/mendeley-parasite-dataset/training/Toxoplasma/"
TEST_TOXOPLASMA_DIR = "/tmp/mendeley-parasite-dataset/testing/Toxoplasma/"

TRICHOMONAD_SOURCE_DIR = "/tmp/Mendeley Parasite Dataset/Trichomonad/"
TRAINING_TRICHOMONAD_DIR = "/tmp/mendeley-parasite-dataset/training/Trichomonad/"
TEST_TRICHOMONAD_DIR = "/tmp/mendeley-parasite-dataset/testing/Trichomonad/"

TRYPANOSOME_SOURCE_DIR = "/tmp/Mendeley Parasite Dataset/Trypanosome/"
TRAINING_TRYPANOSOME_DIR = "/tmp/mendeley-parasite-dataset/training/Trypanosome/"
TEST_TRYPANOSOME_DIR = "/tmp/mendeley-parasite-dataset/testing/Trypanosome/"

split_size = .9 #Bagi data 90% Train, 10% Test
split_data(BABESIA_SOURCE_DIR, TRAINING_BABESIA_DIR, TEST_BABESIA_DIR, split_size)
split_data(LEISHMANIA_SOURCE_DIR, TRAINING_LEISHMANIA_DIR, TEST_LEISHMANIA_DIR, split_size)
split_data(PLASMODIUM_SOURCE_DIR, TRAINING_PLASMODIUM_DIR, TEST_PLASMODIUM_DIR, split_size)
split_data(TOXOPLASMA_SOURCE_DIR, TRAINING_TOXOPLASMA_DIR, TEST_TOXOPLASMA_DIR, split_size)
split_data(TRICHOMONAD_SOURCE_DIR, TRAINING_TRICHOMONAD_DIR, TEST_TRICHOMONAD_DIR, split_size)
split_data(TRYPANOSOME_SOURCE_DIR, TRAINING_TRYPANOSOME_DIR, TEST_TRYPANOSOME_DIR, split_size)

print('Babesia')
print('Train Data : ', (len(os.listdir('/tmp/mendeley-parasite-dataset/training/Babesia/'))))
print('Test Data : ', (len(os.listdir('/tmp/mendeley-parasite-dataset/testing/Babesia/'))))

print('Leishmania')
print('Train Data : ', (len(os.listdir('/tmp/mendeley-parasite-dataset/training/Leishmania/'))))
print('Test Data : ', (len(os.listdir('/tmp/mendeley-parasite-dataset/testing/Leishmania/'))))

print('Plasmodium')
print('Train Data : ', (len(os.listdir('/tmp/mendeley-parasite-dataset/training/Plasmodium/'))))
print('Test Data : ', (len(os.listdir('/tmp/mendeley-parasite-dataset/testing/Plasmodium/'))))

print('Toxoplasma')
print('Train Data : ', (len(os.listdir('/tmp/mendeley-parasite-dataset/training/Toxoplasma/'))))
print('Test Data : ', (len(os.listdir('/tmp/mendeley-parasite-dataset/testing/Toxoplasma/'))))

print('Trichomonad')
print('Train Data : ',(len(os.listdir('/tmp/mendeley-parasite-dataset/training/Trichomonad/'))))
print('Test Data : ', (len(os.listdir('/tmp/mendeley-parasite-dataset/testing/Trichomonad/'))))

print('Trypanosome')
print('Train Data : ', (len(os.listdir('/tmp/mendeley-parasite-dataset/training/Trypanosome/'))))
print('Test Data : ', (len(os.listdir('/tmp/mendeley-parasite-dataset/testing/Trypanosome/'))))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import *
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, MaxPooling2D, Flatten, Dropout

import datetime
global_start = datetime.datetime.now();

from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = "/tmp/mendeley-parasite-dataset/training/"

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      brightness_range=[0.2,1.2],
      fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=32,
    shuffle=True,
    class_mode='categorical',
    target_size=(150, 150))

TESTING_DIR = "/tmp/mendeley-parasite-dataset/testing/"

test_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      brightness_range=[0.2,1.2],
      fill_mode='nearest')

test_generator =  test_datagen.flow_from_directory(
    TESTING_DIR,
    batch_size=1,
    shuffle=False,
    class_mode='categorical',
    target_size=(150, 150)
)

classes = list(train_generator.class_indices.keys())
print('Classes: '+str(classes))
num_classes  = len(classes)

"""InceptionResNetV2"""

pre_trained_model = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                             weights = "imagenet",
                                                             input_shape=(150, 150, 3))

pre_trained_model.summary()

# for layer in pre_trained_model.layers:
#   layer.trainable = True
    
# last_layer = pre_trained_model.get_layer("activation_74")
# print('last layer output shape: ', last_layer.output_shape)
# last_output = last_layer.output

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.97):
      print("\nReached 97.0% accuracy so cancelling training!")
      self.model.stop_training = True

adam = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)


x = pre_trained_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x) # FC layer 1
x = Dense(64,activation='relu')(x)   # FC layer 2
out = Dense(6, activation='softmax')(x) #final layer with softmax activation

model = Model(inputs=pre_trained_model.input,outputs=out)

pre_trained_model.trainable = False

model.compile(optimizer = adam, 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

model.summary()

callbacks = myCallback()
history = model.fit(train_generator,
                    epochs=15,
                    validation_data = test_generator,
                    steps_per_epoch=200,
                    verbose=1)

print("\n")
print('Total Running Time: ', datetime.datetime.now()-global_start)

score = model.evaluate_generator(train_generator)
print('\n')
print('Train Accuracy:', score[1])
print('Train Loss:', score[0])

score = model.evaluate_generator(test_generator)
print('\n')
print('Test Accuracy:', score[1])
print('Test Loss:', score[0])

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Accuracy History')
plt.legend(loc=0)
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Loss History')
plt.legend(loc=0)
plt.figure()

import itertools

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Some reports
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_generator)#, nb_test_samples // BATCH_SIZE, workers=1)
y_pred = np.argmax(Y_pred, axis=1)
target_names = classes

#Confution Matrix
cm = confusion_matrix(test_generator.classes, y_pred)
plot_confusion_matrix(cm, target_names, normalize=False, title='Confusion Matrix - InceptionResNet(Adam)')
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=target_names))